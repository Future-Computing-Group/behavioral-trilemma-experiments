"""Tests for Ollama client (mocked HTTP)."""
import json
import pytest
from unittest.mock import patch, MagicMock

from src.ollama_client import generate_completions, OllamaError


def _mock_response(text, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"response": text}
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    return resp


@patch("src.ollama_client.requests.post")
def test_generate_single_completion(mock_post):
    mock_post.return_value = _mock_response("CONFIDENCE: 0.8\nANSWER: 42")
    results = generate_completions("What is 6*7?", n=1, model="qwen2.5:7b")
    assert len(results) == 1
    assert "CONFIDENCE" in results[0]


@patch("src.ollama_client.requests.post")
def test_generate_multiple_completions(mock_post):
    mock_post.return_value = _mock_response("CONFIDENCE: 0.9\nANSWER: yes")
    results = generate_completions("Is the sky blue?", n=4, model="qwen2.5:7b")
    assert len(results) == 4
    assert mock_post.call_count == 4


@patch("src.ollama_client.requests.post")
def test_temperature_passed(mock_post):
    mock_post.return_value = _mock_response("CONFIDENCE: 0.5\nANSWER: maybe")
    generate_completions("test", n=1, model="qwen2.5:7b", temperature=0.8)
    call_body = json.loads(mock_post.call_args[1]["data"])
    assert call_body["options"]["temperature"] == 0.8


@patch("src.ollama_client.requests.post")
def test_seed_passed(mock_post):
    mock_post.return_value = _mock_response("CONFIDENCE: 0.5\nANSWER: ok")
    generate_completions("test", n=1, model="qwen2.5:7b", seed=42)
    call_body = json.loads(mock_post.call_args[1]["data"])
    assert call_body["options"]["seed"] == 42


@patch("src.ollama_client.requests.post")
def test_seeds_vary_per_draw(mock_post):
    """Each draw in a batch must get a distinct seed for diversity."""
    mock_post.return_value = _mock_response("CONFIDENCE: 0.5\nANSWER: ok")
    generate_completions("test", n=4, model="qwen2.5:7b", seed=100)
    seeds_used = [
        json.loads(call[1]["data"])["options"]["seed"]
        for call in mock_post.call_args_list
    ]
    assert seeds_used == [100, 101, 102, 103]


@patch("src.ollama_client.requests.post")
def test_http_error_raises(mock_post):
    mock_post.return_value = _mock_response("", status=500)
    with pytest.raises(OllamaError):
        generate_completions("test", n=1, model="qwen2.5:7b")


@patch("src.ollama_client.requests.post")
def test_timeout_raises(mock_post):
    mock_post.side_effect = Exception("Connection timed out")
    with pytest.raises(OllamaError):
        generate_completions("test", n=1, model="qwen2.5:7b")
