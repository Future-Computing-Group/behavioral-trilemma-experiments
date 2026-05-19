"""Ollama logprob client: OpenAI-compatible /v1/chat/completions with logprobs.

Mirrors src/ollama_client.py's generation regime (same requests HTTP lib,
same DEFAULT_URL, same temperature/seed/max_tokens defaults, same OllamaError,
same 120s timeout, no retry/backoff) and additionally requests per-token
logprobs so the L.1 logprob-confidence equation can be evaluated.

GitHub main reconcile: no coauthor logprob/inference client present on
https://github.com/lloven/behavioral-trilemma-experiments main (checked
2026-05-19); only src/ollama_client.py exists there. This is the canonical
logprob client; confidence is delegated to analysis.logprob_confidence.

Single-responsibility (L.3): this client does NOT parse ANSWER:/CONFIDENCE:
from the completion -- answer/abstention parsing is a downstream concern.

Transport-parity rationale: see REGIME_NOTE below (consumed by the
cross-model figure's honest-disclosure caption).
"""
import json
from dataclasses import dataclass

import requests

from analysis.logprob_confidence import logprob_confidence

# Same default host as src/ollama_client.py (mirror, do not diverge).
DEFAULT_URL = "http://localhost:11434"

# Quotable for the cross-model figure's honest-disclosure caption.
# Phase-1 transport-parity root cause (review CHANGES-REQUIRED):
#
#   1. Endpoint is FORCED, not arbitrary. Per-token logprobs are returned
#      ONLY by Ollama's OpenAI-compatible /v1/chat/completions; the native
#      /api/generate endpoint (used by src/ollama_client.py) does not emit
#      them. The L.1 logprob-confidence equation therefore cannot be
#      evaluated off /v1. This client cannot switch endpoints.
#
#   2. Within-figure comparability holds BY CONSTRUCTION. All three
#      figure models go through THIS one client at THIS one endpoint, so
#      cross-model comparison within the figure is regime-invariant
#      regardless of any base-client transport question.
#
#   3. Effective-input relationship to the base client is NEGLIGIBLE.
#      src/ollama_client.py posts the caller's prompt verbatim to
#      /api/generate and never sets "raw" (raw absent => Ollama applies
#      the model's Modelfile chat template); it injects no system prompt,
#      prefix, suffix or wrapper. The caller's prompt already contains the
#      full instruction+task block (scripts/generate_tasks.py). This
#      client likewise sends that prompt verbatim as the SOLE user
#      message and injects no system message. Both endpoints apply the
#      model chat template, so the effective text reaching the model is a
#      faithful near-equivalent. This is a documented near-equivalence,
#      NOT a real regime divergence.
REGIME_NOTE = (
    "Logprob transport regime: per-token logprobs are available only via "
    "Ollama's OpenAI-compatible /v1/chat/completions endpoint, so the /v1 "
    "endpoint is forced (the native /api/generate used by the base "
    "src/ollama_client.py does not return logprobs). All figure models go "
    "through this one client at this one endpoint, so within-figure "
    "cross-model comparability holds by construction. Relationship to the "
    "base client is negligible: the base /api/generate path posts the "
    "caller's prompt verbatim and never sets raw (raw absent => Ollama "
    "applies the model chat template) and injects no system prompt; this "
    "client sends the same prompt verbatim as the sole user message with "
    "no system message, and /v1 also applies the model chat template, so "
    "the effective input is a faithful near-equivalent (a documented "
    "near-equivalence, not a real regime divergence)."
)


class OllamaError(Exception):
    """Raised when the Ollama API call fails (mirrors src.ollama_client)."""
    pass


@dataclass
class LogprobCompletion:
    """Return container for generate_with_logprobs.

    content: raw completion text (choices[0].message.content), unparsed.
    token_logprobs: selected-token logprob at every position, ALL tokens.
    confidence: analysis.logprob_confidence(token_logprobs).
    """
    content: str
    token_logprobs: list[float]
    confidence: float


def generate_with_logprobs(
    prompt: str,
    model: str = "qwen2.5:7b",
    temperature: float = 0.8,
    seed: int | None = None,
    max_tokens: int = 512,
    base_url: str = DEFAULT_URL,
) -> LogprobCompletion:
    """Generate one completion with per-token logprobs via /v1/chat/completions.

    Same model/temperature/seed/max_tokens contract as
    src.ollama_client.generate_completions (mirrored, only the endpoint and
    logprobs flags differ); the prompt is wrapped as a single OpenAI-compat
    user message. Raises OllamaError on transport/HTTP failure; raises
    (fail-loud) if logprobs are absent so a missing-logprob response can
    never be silently coerced into a fabricated confidence.
    """
    # Content parity with src/ollama_client.py (review CHANGES-REQUIRED):
    # the base /api/generate path posts the caller's prompt verbatim with
    # NO system prompt / prefix / suffix / wrapper. Reproduce that exactly
    # -- the caller's prompt is the SOLE message and carries the role
    # "user"; never inject a system message the base path lacks. See
    # REGIME_NOTE for the full transport-parity rationale.
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Additional vs the base client: request per-token logprobs.
        "logprobs": True,
        "top_logprobs": 1,
    }
    if seed is not None:
        body["seed"] = seed

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(body),
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise OllamaError(f"Ollama call failed: {e}") from e

    choice = data["choices"][0]
    content = choice["message"]["content"]

    logprobs = choice.get("logprobs")
    if logprobs is None or logprobs.get("content") is None:
        # Fail loud: a logprob-less response cannot yield a real confidence;
        # coercing it would mask an upstream/config bug (no logprobs:true
        # support, wrong endpoint, ...). No sentinel, no fabrication.
        raise OllamaError(
            "response has no logprobs.content: cannot compute "
            "logprob-confidence from a logprob-less completion."
        )

    # We consume content[*].logprob (the selected/realized token's natural-log
    # probability) at EVERY position, with no filtering -- this is exactly the
    # per-token probability in the logprob-confidence equation. The
    # top_logprobs alternatives are requested only to satisfy the API
    # (top_logprobs:1) and are deliberately NOT used.
    token_logprobs = [c["logprob"] for c in logprobs["content"]]

    # logprob_confidence([]) fails loud on a zero-token completion (L.1).
    confidence = logprob_confidence(token_logprobs)

    return LogprobCompletion(
        content=content,
        token_logprobs=token_logprobs,
        confidence=confidence,
    )
