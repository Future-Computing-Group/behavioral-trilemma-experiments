"""R7 (L69): Phase-0 per-task persistence, resume, per-model out path.

Real temp files, no filesystem mocks; resume must be idempotent.
"""
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.orchestrator as orch  # noqa: E402


def _tasks(n):
    return [{"id": f"t{i}", "category": "arithmetic_easy", "prompt": f"p{i}",
             "ground_truth": "42", "verification": "exact"} for i in range(n)]


def test_phase0_out_path_resume_and_incremental(monkeypatch, tmp_path):
    calls = []

    def fake(prompt, n, model, temperature, seed):
        calls.append((prompt, seed))
        return ["CONFIDENCE: 0.9\nANSWER: 42"]
    monkeypatch.setattr(orch, "generate_completions", fake)
    tasks = _tasks(3)
    out = tmp_path / "phase0_gemma2_9b.csv"
    # Prior partial run: t0 already done.
    out.write_text("task_id,p_hat,binding_0.5,binding_0.7,binding_0.9\n"
                   "t0,1.0,0,0,0\n")
    res = orch.run_phase0_calibration(tasks, tmp_path, model="gemma2:9b",
                                      seeds=[1000, 1001], out_path=out)
    assert res == out
    rows = list(csv.DictReader(out.open()))
    assert [r["task_id"] for r in rows] == ["t0", "t1", "t2"]
    assert all(p.startswith("p1") or p.startswith("p2") for p, _ in calls)


def test_phase0_rerun_after_completion_is_idempotent(monkeypatch, tmp_path):
    calls = []

    def fake(prompt, n, model, temperature, seed):
        calls.append(prompt)
        return ["CONFIDENCE: 0.9\nANSWER: 42"]
    monkeypatch.setattr(orch, "generate_completions", fake)
    tasks = _tasks(2)
    out = tmp_path / "phase0_gemma2_9b.csv"
    orch.run_phase0_calibration(tasks, tmp_path, seeds=[1000], out_path=out)
    first = out.read_text()
    n_first = len(calls)

    orch.run_phase0_calibration(tasks, tmp_path, seeds=[1000], out_path=out)

    assert out.read_text() == first        # no truncation, no duplicates
    assert len(calls) == n_first           # zero new client calls


def test_phase0_default_out_path_is_legacy(monkeypatch, tmp_path):
    monkeypatch.setattr(orch, "generate_completions",
                        lambda **kw: ["CONFIDENCE: 0.9\nANSWER: 42"])
    res = orch.run_phase0_calibration(_tasks(1), tmp_path, seeds=[1000])
    assert res == tmp_path / "phase0_calibration.csv"
    assert res.exists()


def test_phase0_rows_persisted_per_task_not_end_of_run(monkeypatch, tmp_path):
    # Crash after the 2nd task: rows for t0 and t1 must already be on disk.
    out = tmp_path / "phase0.csv"
    n_seen = 0

    def fake(prompt, n, model, temperature, seed):
        nonlocal n_seen
        n_seen += 1
        if n_seen == 3:                    # first call of the 3rd task
            raise RuntimeError("simulated crash")
        return ["CONFIDENCE: 0.9\nANSWER: 42"]
    monkeypatch.setattr(orch, "generate_completions", fake)
    try:
        orch.run_phase0_calibration(_tasks(3), tmp_path, seeds=[1000],
                                    out_path=out)
    except RuntimeError:
        pass
    rows = list(csv.DictReader(out.open()))
    assert [r["task_id"] for r in rows] == ["t0", "t1"]
