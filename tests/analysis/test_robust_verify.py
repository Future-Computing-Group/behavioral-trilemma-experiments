"""Tests for analysis.robust_verify (L.5 robust answer verifier).

TDD RED: written before the implementation. The robust verifier is charitable
to verbose prose for arithmetic (any-integer-match after comma-strip) and
factual (substring), but does NOT relax structural correctness for code
(delegates to the original exact-match check).

Critical: a charitable verifier MUST NOT mark wrong answers correct.
Negative tests are first-class here (e.g. "790" must not pass for gold "788").
"""
import pytest

from analysis.robust_verify import robust_verify


# ---------------------------------------------------------------------------
# Arithmetic (charitable any-integer-match; comma-strip)
# ---------------------------------------------------------------------------

def _arith(gold: str) -> dict:
    return {
        "id": "arith_x",
        "category": "arithmetic_easy",
        "ground_truth": gold,
        "verification": "arithmetic",
    }


def test_arithmetic_granite_prose_with_integer_passes():
    # Granite-style verbose prose containing the right integer somewhere.
    assert robust_verify(
        "The sum of 664 and 124 is 788.", _arith("788")
    ) == 1


def test_arithmetic_comma_thousands_in_answer_passes():
    # Phi-style "18,564" should match gold "18564" after comma strip.
    assert robust_verify("18,564", _arith("18564")) == 1


def test_arithmetic_negative_with_leading_number_and_prose_passes():
    # "-734\nThe subtraction of 769 from 35 results in -734" vs "-734".
    text = "-734\nThe subtraction of 769 from 35 results in -734"
    assert robust_verify(text, _arith("-734")) == 1


def test_arithmetic_wrong_answer_returns_0():
    # "790" should not pass for gold "788" — charitable, not blind.
    assert robust_verify("790", _arith("788")) == 0


def test_arithmetic_wrong_comma_position_returns_0():
    # "185,64" -> "18564" after comma-strip is correct;
    # but "18,5,64" with extra commas should still strip to "18564" -> 1.
    # The real failure to test: a number that isn't the gold at all.
    assert robust_verify("123", _arith("18564")) == 0


def test_arithmetic_partial_integer_match_within_larger_number_returns_0():
    # gold "788" should NOT match "78800" — regex must find the integer
    # as a standalone signed-integer token, not as a substring.
    assert robust_verify("78800", _arith("788")) == 0


def test_arithmetic_plain_integer_answer_passes():
    assert robust_verify("788", _arith("788")) == 1


def test_arithmetic_none_returns_0():
    assert robust_verify(None, _arith("788")) == 0


def test_arithmetic_empty_returns_0():
    assert robust_verify("", _arith("788")) == 0


# ---------------------------------------------------------------------------
# Factual (substring, case-insensitive; exact-after-strip)
# ---------------------------------------------------------------------------

def _fact(gold: str) -> dict:
    return {
        "id": "fact_x",
        "category": "factual_common",
        "ground_truth": gold,
        "verification": "exact",
    }


def test_factual_substring_passes():
    assert robust_verify(
        "The capital of Finland is Helsinki.", _fact("Helsinki")
    ) == 1


def test_factual_case_insensitive():
    assert robust_verify("HELSINKI", _fact("Helsinki")) == 1


def test_factual_wrong_returns_0():
    assert robust_verify(
        "The capital of Finland is Stockholm.", _fact("Helsinki")
    ) == 0


def test_factual_exact_match_passes():
    assert robust_verify("Helsinki", _fact("Helsinki")) == 1


def test_factual_trailing_punctuation_ok():
    assert robust_verify("Helsinki.", _fact("Helsinki")) == 1


def test_factual_none_returns_0():
    assert robust_verify(None, _fact("Helsinki")) == 0


# ---------------------------------------------------------------------------
# Code (structural — exact preserved, no relaxation)
# ---------------------------------------------------------------------------

def _code_task() -> dict:
    """A code task with a trivial test case: a function adding two numbers."""
    return {
        "id": "code_x",
        "category": "code_simple",
        "ground_truth": "def add(a, b):\n    return a + b",
        "verification": "code",
        "test_cases": [
            {"input": [1, 2], "expected": 3, "function": "add"},
            {"input": [10, 5], "expected": 15, "function": "add"},
        ],
    }


def test_code_correct_passes():
    """A correct code answer (passes the test_cases) returns 1.

    Uses orchestrator._verify_answer's structural semantics — robust_verify
    must not relax this path.
    """
    task = _code_task()
    correct = "def add(a, b):\n    return a + b"
    # The orchestrator wires test_cases through src.tasks.verify_code; if
    # that's available the result is 1, else (defensive) 0. Either way the
    # call must match orchestrator._verify_answer behavior — assert parity.
    from src.orchestrator import _verify_answer
    expected = _verify_answer(task, correct)
    assert robust_verify(correct, task) == expected


def test_code_wrong_fails():
    """A wrong code answer returns 0 (structural correctness untouched)."""
    task = _code_task()
    wrong = "def add(a, b):\n    return a - b"
    from src.orchestrator import _verify_answer
    expected = _verify_answer(task, wrong)
    assert robust_verify(wrong, task) == expected
    # And that expected is 0 (sanity).
    assert expected == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_unknown_category_falls_back_to_exact():
    task = {
        "id": "weird",
        "category": "??",
        "ground_truth": "foo",
        "verification": "??",
    }
    assert robust_verify("foo", task) == 1
    assert robust_verify("bar", task) == 0


def test_none_answer_any_category_returns_0():
    for ver in ("arithmetic", "exact", "code"):
        task = {"id": "x", "category": "y",
                "ground_truth": "z", "verification": ver}
        assert robust_verify(None, task) == 0
