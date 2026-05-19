"""Logprob-confidence: all-token geometric mean, clipped to [0.01, 1.0].

Pure function implementing the manuscript's logprob-confidence equation. No
network, no file I/O, no logging, no argparse, no global state. Stdlib only.

The confidence of a completion is the geometric mean of the per-token
probabilities of its selected (realized) tokens, taken over EVERY content
token of the completion. The manuscript deliberately uses the all-token set
(no answer-span selection, no role/template-token exclusion): this dilutes
confidence with low-probability format tokens, and that dilution is the
intended behavior, not a defect. The geometric mean is computed in
log-space (``exp(mean(token_logprobs))``) for numerical stability rather
than by multiplying raw probabilities, then clipped to the closed interval
[0.01, 1.0].
"""

import math

__all__ = ["logprob_confidence"]

_LO = 0.01
_HI = 1.0


def logprob_confidence(token_logprobs: list[float]) -> float:
    """Geometric mean of per-token probabilities, clipped to [0.01, 1.0].

    ``token_logprobs`` are the per-token natural-log probabilities of the
    selected tokens of a completion (one float per token, each <= 0.0),
    over ALL completion tokens with no filtering.

    Geometric mean over T tokens = (prod p_t)^(1/T) = exp(mean(logprobs)),
    computed via the log-space form for numerical stability. The result is
    then clipped: < 0.01 -> 0.01; > 1.0 -> 1.0.
    """
    if not token_logprobs:
        # Zero tokens is an upstream generation failure; coercing it to a
        # confidence number would mask a bug -- fail loud, no sentinel.
        raise ValueError(
            "token_logprobs is empty: a zero-token completion is an "
            "upstream generation failure, not a confidence of 0."
        )
    geomean = math.exp(sum(token_logprobs) / len(token_logprobs))
    return min(_HI, max(_LO, geomean))
