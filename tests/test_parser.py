"""Tests for response parser."""
import pytest
from src.parser import parse_response


def test_standard_format():
    text = "CONFIDENCE: 0.85\nANSWER: 42"
    r, a = parse_response(text)
    assert r == 0.85
    assert a == "42"


def test_extra_whitespace():
    text = "  CONFIDENCE:  0.7  \n  ANSWER:  hello world  "
    r, a = parse_response(text)
    assert r == 0.7
    assert a == "hello world"


def test_multiline_answer():
    text = "CONFIDENCE: 0.9\nANSWER: def foo():\n    return 42"
    r, a = parse_response(text)
    assert r == 0.9
    assert "def foo():" in a


def test_confidence_as_percentage():
    """If model writes 85% instead of 0.85, parser should handle it."""
    text = "CONFIDENCE: 85%\nANSWER: yes"
    r, a = parse_response(text)
    assert r == 0.85


def test_confidence_clamped_to_0_1():
    text = "CONFIDENCE: 1.5\nANSWER: something"
    r, a = parse_response(text)
    assert r == 1.0


def test_confidence_negative_clamped():
    text = "CONFIDENCE: -0.3\nANSWER: something"
    r, a = parse_response(text)
    assert r == 0.0


def test_missing_confidence_returns_none():
    text = "I think the answer is 42"
    r, a = parse_response(text)
    assert r is None


def test_missing_answer_returns_none():
    text = "CONFIDENCE: 0.9"
    r, a = parse_response(text)
    assert r == 0.9
    assert a is None


def test_case_insensitive():
    text = "confidence: 0.6\nanswer: blue"
    r, a = parse_response(text)
    assert r == 0.6
    assert a == "blue"


def test_with_reasoning_before():
    text = "Let me think...\nThe answer is probably 7.\nCONFIDENCE: 0.75\nANSWER: 7"
    r, a = parse_response(text)
    assert r == 0.75
    assert a == "7"
