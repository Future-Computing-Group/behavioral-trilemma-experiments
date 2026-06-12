"""R5 (L69): Stage-A pool resume — append + skip-completed + salvage.

Real temp files, no filesystem mocks. Resume must be IDEMPOTENT: a rerun
after a crash never truncates completed work and never duplicates records.
"""
import json
import pathlib
import sys
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.generate_pool as gp  # noqa: E402


def _tasks(n):
    return [{"id": f"t{i:02d}", "category": "arithmetic_easy", "prompt": "p",
             "ground_truth": "42", "verification": "exact"} for i in range(n)]


def _fake(calls):
    def fake(**kw):
        calls.append(kw["prompt"])
        return SimpleNamespace(content="CONFIDENCE: 0.9\nANSWER: 42",
                               token_logprobs=[-0.1])
    return fake


def _prior_lines(tasks):
    return [json.dumps({"task_id": t["id"], "category": t["category"],
                        "seed": 42, "confidence_source": "logprob",
                        "completions": []}) for t in tasks]


def test_resume_skips_completed_and_preserves_bytes(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(gp, "generate_with_logprobs", _fake(calls))
    tasks = _tasks(4)
    # Simulate a prior partial run: tasks t00, t01 already on disk.
    out = gp.pool_path(42, tmp_path, model_slug="qwen2.5_7b")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(_prior_lines(tasks[:2])) + "\n")
    before = out.read_text()

    gp.write_pool_for_seed(42, tasks, tmp_path, n_max=1, progress=False)

    text = out.read_text()
    assert text.startswith(before)                  # originals byte-preserved
    ids = [json.loads(l)["task_id"] for l in text.splitlines()]
    assert ids == ["t00", "t01", "t02", "t03"]      # exactly once each
    assert len(calls) == 2                          # only the 2 remaining ran


def test_resume_after_completion_is_idempotent(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(gp, "generate_with_logprobs", _fake(calls))
    tasks = _tasks(3)
    gp.write_pool_for_seed(42, tasks, tmp_path, n_max=1, progress=False)
    out = gp.pool_path(42, tmp_path, model_slug="qwen2.5_7b")
    first = out.read_text()
    n_calls_first = len(calls)

    gp.write_pool_for_seed(42, tasks, tmp_path, n_max=1, progress=False)

    assert out.read_text() == first                 # no truncation, no dupes
    assert len(calls) == n_calls_first              # zero new client calls


def test_trailing_partial_line_salvaged_and_rerun(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(gp, "generate_with_logprobs", _fake(calls))
    tasks = _tasks(3)
    out = gp.pool_path(42, tmp_path, model_slug="qwen2.5_7b")
    out.parent.mkdir(parents=True, exist_ok=True)
    good = _prior_lines(tasks[:2])
    # Crash mid-write: t02's record was cut off (no newline, invalid JSON).
    out.write_text("\n".join(good) + "\n" + '{"task_id": "t02", "comp')

    gp.write_pool_for_seed(42, tasks, tmp_path, n_max=1, progress=False)

    lines = out.read_text().splitlines()
    ids = [json.loads(l)["task_id"] for l in lines]
    assert ids == ["t00", "t01", "t02"]             # garbage dropped, t02 rerun
    assert len(calls) == 1                          # only t02 ran
