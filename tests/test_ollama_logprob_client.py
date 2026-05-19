"""Tests for Ollama logprob client (mocked HTTP, no network).

Mirrors tests/test_ollama_client.py: patches the SAME library
(src.ollama_logprob_client.requests.post) with a realistic
/v1/chat/completions OpenAI-compat JSON fixture.
"""
import json
import math

import pytest
from unittest.mock import patch, MagicMock

from src.ollama_logprob_client import generate_with_logprobs, OllamaError


# Hand-chosen logprobs over a 4-token completion. Geometric mean of the
# per-token probabilities is exp(mean(logprobs)); computed by hand below.
_FIXTURE_LOGPROBS = [-0.10536051565782628, -0.2231435513142097,
                     -0.6931471805599453, -0.35667494393873245]
# Probabilities: 0.90, 0.80, 0.50, 0.70.
# mean(logprobs) = (-0.10536051565782628 - 0.2231435513142097
#                   - 0.6931471805599453 - 0.35667494393873245) / 4
#                = -1.3783261614707137 / 4 = -0.3445815403676784
# geomean = exp(-0.3445815403676784) = 0.7086757731211741  (within [0.01, 1.0])
_EXPECTED_CONFIDENCE = math.exp(sum(_FIXTURE_LOGPROBS) / len(_FIXTURE_LOGPROBS))


def _mock_response(content, logprobs_content, status=200):
    """Build a MagicMock mimicking a requests Response for /v1 chat."""
    resp = MagicMock()
    resp.status_code = status
    payload = {
        "choices": [
            {
                "message": {"content": content},
                "logprobs": (
                    {"content": logprobs_content}
                    if logprobs_content is not None
                    else None
                ),
            }
        ]
    }
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    return resp


def _lp_content(logprobs):
    """OpenAI-compat logprobs.content array (selected token + top_logprobs)."""
    return [
        {
            "token": f"t{i}",
            "logprob": lp,
            "top_logprobs": [{"token": f"t{i}", "logprob": lp}],
        }
        for i, lp in enumerate(logprobs)
    ]


@patch("src.ollama_logprob_client.requests.post")
def test_happy_path_content_logprobs_confidence(mock_post):
    mock_post.return_value = _mock_response(
        "ANSWER: 42\nCONFIDENCE: 0.8", _lp_content(_FIXTURE_LOGPROBS)
    )
    result = generate_with_logprobs("What is 6*7?", model="qwen2.5:7b")

    # content extracted verbatim, NOT parsed
    assert result.content == "ANSWER: 42\nCONFIDENCE: 0.8"
    # token_logprobs == ALL selected-token logprobs, in order, no filtering
    assert result.token_logprobs == _FIXTURE_LOGPROBS
    # confidence == hand-computed logprob_confidence of them
    assert result.confidence == pytest.approx(_EXPECTED_CONFIDENCE)


@patch("src.ollama_logprob_client.requests.post")
def test_request_body_contract(mock_post):
    """Body must request logprobs and mirror temp/seed/max_tokens contract."""
    mock_post.return_value = _mock_response(
        "ok", _lp_content(_FIXTURE_LOGPROBS)
    )
    generate_with_logprobs(
        "test",
        model="qwen2.5:7b",
        temperature=0.8,
        seed=42,
        max_tokens=512,
    )
    body = json.loads(mock_post.call_args[1]["data"])
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 1
    assert body["model"] == "qwen2.5:7b"
    assert body["temperature"] == 0.8
    assert body["seed"] == 42
    assert body["max_tokens"] == 512
    # OpenAI-compat: prompt wrapped as a user message
    assert body["messages"] == [{"role": "user", "content": "test"}]
    # hits the /v1/chat/completions endpoint
    assert mock_post.call_args[0][0].endswith("/v1/chat/completions")


@patch("src.ollama_logprob_client.requests.post")
def test_default_base_url_and_endpoint(mock_post):
    mock_post.return_value = _mock_response(
        "ok", _lp_content(_FIXTURE_LOGPROBS)
    )
    generate_with_logprobs("test")
    url = mock_post.call_args[0][0]
    assert url == "http://localhost:11434/v1/chat/completions"


@patch("src.ollama_logprob_client.requests.post")
def test_missing_logprobs_fails_loud(mock_post):
    """logprobs None -> fail loud, no fabricated confidence."""
    mock_post.return_value = _mock_response("some text", None)
    with pytest.raises((ValueError, OllamaError)):
        generate_with_logprobs("test")


@patch("src.ollama_logprob_client.requests.post")
def test_empty_logprobs_content_fails_loud(mock_post):
    """Empty logprobs.content -> logprob_confidence([]) raises ValueError."""
    mock_post.return_value = _mock_response("", [])
    with pytest.raises((ValueError, OllamaError)):
        generate_with_logprobs("test")


@patch("src.ollama_logprob_client.requests.post")
def test_http_error_raises(mock_post):
    mock_post.return_value = _mock_response("", None, status=500)
    with pytest.raises(OllamaError):
        generate_with_logprobs("test")


@patch("src.ollama_logprob_client.requests.post")
def test_timeout_raises(mock_post):
    mock_post.side_effect = Exception("Connection timed out")
    with pytest.raises(OllamaError):
        generate_with_logprobs("test")
