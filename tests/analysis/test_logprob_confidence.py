"""Tests for analysis.logprob_confidence — all-token geomean confidence.

TDD RED step: written before analysis/logprob_confidence.py exists. Pure
function, no fixtures / no I/O. Verifies the manuscript logprob-confidence
equation: geometric mean of per-token probabilities over ALL completion
tokens, clipped to [0.01, 1.0]; empty input raises ValueError.
"""

import math

import pytest

from analysis.logprob_confidence import logprob_confidence


def test_two_equal_tokens_geomean_half():
    # exp((log0.5 + log0.5)/2) = 0.5
    assert logprob_confidence([math.log(0.5), math.log(0.5)]) == pytest.approx(0.5)


def test_two_distinct_tokens_hand_computable():
    # exp((log0.9 + log0.4)/2) = sqrt(0.9 * 0.4) = sqrt(0.36) = 0.6
    assert logprob_confidence([math.log(0.9), math.log(0.4)]) == pytest.approx(0.6)


def test_single_token_returns_its_probability():
    assert logprob_confidence([math.log(0.73)]) == pytest.approx(0.73)


def test_lower_clip_to_point_zero_one():
    # Geomean prob far below 0.01 (exp(-20) ~ 2e-9) must clip to exactly 0.01.
    result = logprob_confidence([-20.0, -20.0, -20.0])
    assert result == 0.01


def test_upper_clip_does_not_exceed_one():
    # logprob 0.0 => p = 1.0; geomean = 1.0, must return exactly 1.0.
    result = logprob_confidence([0.0, 0.0])
    assert result == 1.0
    assert result <= 1.0


def test_all_token_dilution_invariant():
    # Manuscript all-token-dilution invariant: the function MUST use every
    # element. A mix of "answer-like" high-confidence tokens and
    # "format-like" low-confidence template tokens; filtering any subset
    # (e.g. dropping the format tokens) would yield a different, higher
    # number. We assert the result equals the ALL-token geomean exactly.
    answer_like = [math.log(0.95), math.log(0.92), math.log(0.97)]
    format_like = [math.log(0.20), math.log(0.15)]
    all_tokens = answer_like + format_like
    expected_all = math.exp(sum(all_tokens) / len(all_tokens))
    # Sanity: a filtered (answer-only) geomean would differ markedly.
    filtered = math.exp(sum(answer_like) / len(answer_like))
    assert expected_all != pytest.approx(filtered)
    assert logprob_confidence(all_tokens) == pytest.approx(expected_all)


def test_empty_input_raises_value_error():
    with pytest.raises(ValueError):
        logprob_confidence([])
