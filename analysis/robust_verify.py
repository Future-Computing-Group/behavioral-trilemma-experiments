"""Charitable answer verifier for cross-model logprob CSV re-analysis (L.5).

Why this exists
---------------
The original ``src.orchestrator._verify_answer`` is exact-match: it compares
``answer.strip().lower() == gold`` (with a numeric-equality branch for
``arithmetic`` tasks). It was tuned for the qwen-7B 540-config gated runs,
where qwen reliably emits a bare number / bare token after ``ANSWER:``. It
is preserved unchanged for the H1--H6 Tables (backward compatibility).

For the cross-model logprob figure, other open-weight families
(granite, phi, gemma2, mistral, yi, ...) frequently wrap the answer in
verbose prose, e.g. "The sum of 664 and 124 is 788." or "18,564". The
original verifier scores these as 0 even when the model produced the right
integer. That bias confounds H (helpfulness) by model VERBOSITY rather than
COMPETENCE, which is exactly the thing the cross-model placement is meant
to compare.

What this does
--------------
Category-appropriate matching (charitable for prose; structural cases
remain strict):

* arithmetic_easy / arithmetic_hard:
  Strip commas from both sides; regex-extract all signed integers
  (``-?\\d+``) from the answer; pass if the gold integer is in that set.
  Recovers "788.", "18,564", "-734\\nThe subtraction of 769 from 35 is
  -734" without admitting "790" for gold "788" (integer-equality, not
  substring).
* factual (verification == "exact"):
  Case-insensitive substring match (after light trailing punctuation
  strip). Handles "The capital of Finland is Helsinki." vs "Helsinki".
* code (verification == "code"):
  Delegates verbatim to ``src.orchestrator._verify_answer`` — structural
  correctness cannot be fuzzed.
* Unknown category / unknown verification: fall back to exact lowercase
  match (conservative).
* ``answer is None`` / empty string -> 0 for every branch.
"""
from __future__ import annotations

import re
import string


_INT_RE = re.compile(r"-?\d+")
# Categories that should use the charitable any-integer match. The
# task-set uses these two names for arithmetic; "verification": "arithmetic"
# is the SSoT signal but we also accept the category fallback.
_ARITHMETIC_CATEGORIES = {"arithmetic_easy", "arithmetic_hard"}
# Categories that should use substring/factual match. The task-set uses
# "factual_common" and "factual_obscure"; verification is "exact".
_FACTUAL_CATEGORIES = {"factual_common", "factual_obscure", "factual"}

# Punctuation we strip from the trailing end of a factual answer before
# the exact-after-strip check (substring check ignores this anyway, but
# it makes the equality fallback more forgiving).
_TRAILING_PUNCT = string.punctuation + " \t\n\r"


def _arithmetic_match(answer: str, gold: str) -> int:
    """Any-integer-match: pass iff gold (as integer) appears as a standalone
    signed-integer token in ``answer`` after comma-strip on both sides.
    """
    a_stripped = answer.replace(",", "")
    g_stripped = gold.replace(",", "").strip()
    # Gold must itself be a valid integer for this branch to apply at all.
    try:
        gold_int = int(g_stripped)
    except ValueError:
        # Gold isn't an integer (shouldn't happen for arithmetic tasks);
        # fall back to exact-lowercase match.
        return 1 if a_stripped.strip().lower() == g_stripped.lower() else 0

    # Standalone integer tokens in the answer (after comma strip). The regex
    # naturally avoids substring-of-larger-number matches because \d+ is
    # greedy and integers are delimited from each other.
    found = _INT_RE.findall(a_stripped)
    if not found:
        return 0
    try:
        ints = {int(s) for s in found}
    except ValueError:
        return 0
    return 1 if gold_int in ints else 0


def _factual_match(answer: str, gold: str) -> int:
    """Case-insensitive substring match, with trailing-punct equality
    fallback. Returns 1 if ``gold`` (lowercased, stripped) is contained in
    ``answer`` (lowercased, stripped), or if their exact-after-trailing-
    punct-strip lowercased forms match.
    """
    a_low = answer.strip().lower()
    g_low = gold.strip().lower()
    if not g_low:
        # Degenerate gold: only match the exact empty string.
        return 1 if a_low == "" else 0
    if g_low in a_low:
        return 1
    # Trailing punctuation tolerance: "Helsinki." == "Helsinki".
    a_tail_stripped = a_low.rstrip(_TRAILING_PUNCT)
    g_tail_stripped = g_low.rstrip(_TRAILING_PUNCT)
    if a_tail_stripped == g_tail_stripped:
        return 1
    return 0


def robust_verify(answer: str | None, task: dict) -> int:
    """Return 1 if ``answer`` matches ``task['ground_truth']`` under the
    category-appropriate matching rule, else 0.

    See module docstring for the per-category contract.
    """
    if answer is None:
        return 0
    if not isinstance(answer, str) or answer == "":
        return 0

    gold = str(task.get("ground_truth", "")).strip()
    category = task.get("category", "")
    verification = task.get("verification", "")

    # Arithmetic — charitable any-integer-match.
    if verification == "arithmetic" or category in _ARITHMETIC_CATEGORIES:
        return _arithmetic_match(answer, gold)

    # Code — strictly delegate to the original structural verifier.
    if verification == "code" or category.startswith("code"):
        # Import lazily to avoid an unconditional dependency at module
        # import time (and to keep the test that mocks _verify_answer
        # working correctly).
        from src.orchestrator import _verify_answer
        return _verify_answer(task, answer)

    # Factual / exact — substring + case-insensitive (charitable for prose).
    if verification == "exact" or category in _FACTUAL_CATEGORIES:
        return _factual_match(answer, gold)

    # Unknown — conservative exact-lowercase fallback.
    return 1 if answer.strip().lower() == gold.lower() else 0
