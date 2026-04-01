"""Ollama HTTP client for generating completions."""
import json
import requests

DEFAULT_URL = "http://localhost:11434"


class OllamaError(Exception):
    """Raised when Ollama API call fails."""
    pass


def generate_completions(
    prompt: str,
    n: int = 1,
    model: str = "qwen2.5:7b",
    temperature: float = 0.8,
    seed: int | None = None,
    max_tokens: int = 512,
    base_url: str = DEFAULT_URL,
) -> list[str]:
    """Generate n completions from Ollama.

    Returns a list of response strings. Raises OllamaError on failure (L39).
    """
    results = []
    for i in range(n):
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if seed is not None:
            # Vary seed per draw to ensure diverse completions.
            # Same base seed + draw index = reproducible but distinct.
            body["options"]["seed"] = seed + i

        try:
            resp = requests.post(
                f"{base_url}/api/generate",
                data=json.dumps(body),
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            results.append(data.get("response", ""))
        except Exception as e:
            raise OllamaError(f"Ollama call failed: {e}") from e

    return results
