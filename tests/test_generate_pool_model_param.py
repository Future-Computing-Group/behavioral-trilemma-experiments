"""R4: --model plumbing in Stage A (EXPERIMENT-PLAN §11.E-B).

The fake preserves the contract under test (L88): it records the model
parameter and varies its output by model and seed, so a dead seam
(parameter discarded) fails the distinct-answer assertions.
"""
import pathlib
import sys
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.generate_pool as gp  # noqa: E402


class FakeClient:
    def __init__(self):
        self.calls = []

    def __call__(self, *, prompt, model, temperature, seed, max_tokens):
        self.calls.append({"model": model, "seed": seed})
        # Model-sensitive content: distinct models => distinct tokens (L88).
        return SimpleNamespace(
            content=f"CONFIDENCE: 0.9\nANSWER: {model}-{seed}",
            token_logprobs=[-0.1 - 0.001 * seed],
        )


def test_model_param_reaches_client_and_paths(monkeypatch, tmp_path):
    fake = FakeClient()
    monkeypatch.setattr(gp, "generate_with_logprobs", fake)
    task = {"id": "arith_easy_01", "category": "arithmetic_easy",
            "prompt": "p", "ground_truth": "42", "verification": "exact"}
    rec = gp.generate_pool_record_for_task(task=task, seed=42, n_max=2,
                                           model="gemma2:9b")
    assert {c["model"] for c in fake.calls} == {"gemma2:9b"}
    assert [c["seed"] for c in fake.calls] == [42, 43]        # L65 schedule
    # Distinct per-completion seeds produced distinct answers (seam alive).
    assert rec["completions"][0]["a"] != rec["completions"][1]["a"]
    # Slug-derived paths:
    assert gp.pool_path(42, tmp_path, model_slug="gemma2_9b").name == \
        "pool_gemma2_9b_seed42_N32.jsonl"


def test_cli_model_flag_reaches_client(monkeypatch, capsys):
    fake = FakeClient()
    monkeypatch.setattr(gp, "generate_with_logprobs", fake)
    rc = gp.main(["--smoke", "--model", "gemma2:9b"])
    assert rc == 0
    assert {c["model"] for c in fake.calls} == {"gemma2:9b"}


def test_default_model_slug_paths_unchanged(tmp_path):
    # Regression: the qwen2.5:7b default keeps the legacy filenames.
    assert gp.pool_path(0, tmp_path).name == "pool_qwen2.5_7b_seed0_N32.jsonl"
    assert gp.pool_meta_path(tmp_path).name == "pool_meta_qwen2.5_7b_N32.json"


def test_model_slug_rule_matches_eval_logprob():
    from scripts.eval_logprob import _model_slug
    for tag in ("qwen2.5:7b", "gemma2:9b", "mistral:7b-instruct-q4_K_M"):
        assert gp.model_slug(tag) == _model_slug(tag)


def test_write_pool_meta_records_actual_model(tmp_path):
    path = gp.write_pool_meta(tmp_path, model="gemma2:9b",
                              model_slug="gemma2_9b")
    import json
    meta = json.loads(path.read_text())
    assert meta["model"] == "gemma2:9b"
    assert path.name == "pool_meta_gemma2_9b_N32.json"
