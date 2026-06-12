"""R6 (L63): unconditional per-task status emitter.

The STATUS plumbing is under test, not the client; the stub client still
preserves the contract (distinct tasks => distinct records, L88).
"""
import json
import pathlib
import sys
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.status import write_status  # noqa: E402


def test_write_status_atomic_and_readable(tmp_path):
    p = tmp_path / "STATUS.json"
    write_status(p, stage="stageA", model="gemma2:9b", seed=42,
                 task_idx=3, total_tasks=100, completions_done=96,
                 parse_fail_count=1, elapsed_s=12.5, eta_s=400.0)
    d = json.loads(p.read_text())
    assert d["task_idx"] == 3 and d["parse_fail_count"] == 1
    assert "last_update" in d
    assert not list(tmp_path.glob("*.tmp"))         # tmp+rename, no leftovers


def test_stage_a_emits_per_task(monkeypatch, tmp_path):
    import scripts.generate_pool as gp
    seen = []
    real = gp.write_status
    monkeypatch.setattr(gp, "write_status",
                        lambda p, **kw: (seen.append(kw), real(p, **kw)))
    monkeypatch.setattr(gp, "generate_with_logprobs", lambda **kw:
        SimpleNamespace(content=f"CONFIDENCE: 0.9\nANSWER: {kw['seed']}",
                        token_logprobs=[-0.1]))
    tasks = [{"id": f"t{i}", "category": "arithmetic_easy", "prompt": "p",
              "ground_truth": "42", "verification": "exact"} for i in range(3)]
    pools = tmp_path / "logprob-x" / "pools"
    gp.write_pool_for_seed(0, tasks, pools, n_max=1, progress=False)
    assert [kw["task_idx"] for kw in seen] == [1, 2, 3]   # one per task
    # The status FILE exists at the output root and reflects the last task.
    status = json.loads((tmp_path / "logprob-x" / "STATUS.json").read_text())
    assert status["task_idx"] == 3 and status["total_tasks"] == 3
    assert status["stage"] == "stageA"


def test_phase0_emits_per_task(monkeypatch, tmp_path):
    import src.orchestrator as orch
    seen = []
    real = orch.write_status
    monkeypatch.setattr(orch, "write_status",
                        lambda p, **kw: (seen.append(kw), real(p, **kw)))
    monkeypatch.setattr(orch, "generate_completions",
                        lambda **kw: [f"CONFIDENCE: 0.9\nANSWER: {kw['seed']}"])
    tasks = [{"id": f"t{i}", "category": "arithmetic_easy", "prompt": f"p{i}",
              "ground_truth": "42", "verification": "exact"} for i in range(3)]
    orch.run_phase0_calibration(tasks, tmp_path, seeds=[1000, 1001])
    assert [kw["task_idx"] for kw in seen] == [1, 2, 3]
    status = json.loads((tmp_path / "STATUS.json").read_text())
    assert status["stage"] == "phase0" and status["task_idx"] == 3
