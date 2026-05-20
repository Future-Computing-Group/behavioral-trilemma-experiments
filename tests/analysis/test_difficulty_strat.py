"""TDD tests for analysis.difficulty_strat (L.5 decoupled analysis #2).

Per-task difficulty = 1 - cross-model mean correctness on committed; harder
task -> fewer models commit and get it right. Bin into difficulty tertiles,
then compute (H, C, A) per (model, bin) — competence-controlled by holding
the task-difficulty bin fixed.
"""

import csv
import math
import pathlib

import pytest

LOGPROB_HEADER = ["task", "category", "seed", "r_logprob", "answer", "y",
                  "acted"]


def _write_csv(path: pathlib.Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOGPROB_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({
                "task": r.get("task"),
                "category": r.get("category", "arithmetic_easy"),
                "seed": r.get("seed", 1),
                "r_logprob": r.get("r_logprob", ""),
                "answer": r.get("answer", "a"),
                "y": r.get("y", ""),
                "acted": r.get("acted", 0),
            })
    return str(path)


def _r(task, r, y, acted):
    return {"task": task, "r_logprob": r, "answer": "x" if acted else "",
            "y": y, "acted": acted}


@pytest.fixture
def two_model_six_task(tmp_path):
    """2 models x 6 tasks: 2 easy, 2 medium, 2 hard.

    Easy (t0,t1): both models commit & correct (y=1)
    Medium (t2,t3): mA correct, mB wrong; both commit
    Hard (t4,t5): both commit & wrong (y=0)
    """
    runs = tmp_path / "logprob_xmodel"
    # Model A
    a_rows = [
        _r("t0", 0.9, 1, 1), _r("t1", 0.9, 1, 1),
        _r("t2", 0.8, 1, 1), _r("t3", 0.8, 1, 1),
        _r("t4", 0.4, 0, 1), _r("t5", 0.4, 0, 1),
    ]
    for s in (1, 2, 3, 4, 5):
        _write_csv(runs / "mA" / f"mA_s{s}.csv", list(a_rows))
    # Model B
    b_rows = [
        _r("t0", 0.9, 1, 1), _r("t1", 0.9, 1, 1),
        _r("t2", 0.6, 0, 1), _r("t3", 0.6, 0, 1),
        _r("t4", 0.5, 0, 1), _r("t5", 0.5, 0, 1),
    ]
    for s in (1, 2, 3, 4, 5):
        _write_csv(runs / "mB" / f"mB_s{s}.csv", list(b_rows))
    return runs


def test_task_difficulty_assigns_correctly(two_model_six_task):
    from analysis.difficulty_strat import task_difficulty
    diff = task_difficulty(str(two_model_six_task))
    # All 6 tasks present
    assert set(diff) == {"t0", "t1", "t2", "t3", "t4", "t5"}
    # Easy: both right -> mean correct 1.0 -> difficulty 0
    assert diff["t0"] == pytest.approx(0.0)
    assert diff["t1"] == pytest.approx(0.0)
    # Medium: 1 of 2 -> diff 0.5
    assert diff["t2"] == pytest.approx(0.5)
    assert diff["t3"] == pytest.approx(0.5)
    # Hard: 0 of 2 -> diff 1.0
    assert diff["t4"] == pytest.approx(1.0)
    assert diff["t5"] == pytest.approx(1.0)


def test_stratified_coords_returns_per_bin_per_model(two_model_six_task):
    from analysis.difficulty_strat import stratified_coords
    out = stratified_coords(str(two_model_six_task), n_bins=3)
    assert set(out) == {"mA", "mB"}
    # Each model has an entry per bin (some may be empty).
    for mid, by_bin in out.items():
        # We require keys for each tertile, even if empty (caller filters).
        assert set(by_bin.keys()) >= {"easy", "medium", "hard"}


def test_stratified_coords_easy_bin_perfect_for_both(two_model_six_task):
    from analysis.difficulty_strat import stratified_coords
    out = stratified_coords(str(two_model_six_task), n_bins=3)
    # In the easy bin, both models are 100% correct -> H == 1.0.
    assert out["mA"]["easy"]["H"] == pytest.approx(1.0)
    assert out["mB"]["easy"]["H"] == pytest.approx(1.0)
    # n_tasks_in_bin counts DISTINCT tasks in that bin (t0, t1 -> 2).
    assert out["mA"]["easy"]["n_tasks_in_bin"] == 2
    assert out["mB"]["easy"]["n_tasks_in_bin"] == 2


def test_stratified_coords_medium_bin_diverges(two_model_six_task):
    from analysis.difficulty_strat import stratified_coords
    out = stratified_coords(str(two_model_six_task), n_bins=3)
    # Medium bin: mA correct on both medium tasks, mB wrong on both.
    assert out["mA"]["medium"]["H"] == pytest.approx(1.0)
    assert out["mB"]["medium"]["H"] == pytest.approx(0.0)


def test_stratified_bin_counts_partition_total(two_model_six_task):
    from analysis.difficulty_strat import stratified_coords
    out = stratified_coords(str(two_model_six_task), n_bins=3)
    for mid, by_bin in out.items():
        total = sum(b["n_tasks_in_bin"] for b in by_bin.values())
        # 6 distinct tasks total
        assert total == 6


def test_model_with_zero_acted_in_bin_returns_nan_safe(tmp_path):
    """A model with 0 acted rows in a bin -> nan-safe coords + n_acted=0."""
    from analysis.difficulty_strat import stratified_coords
    runs = tmp_path / "logprob_xmodel"
    # mC: abstains on all hard tasks; commits & correct on easy.
    c_rows = [
        _r("t0", 0.9, 1, 1), _r("t1", 0.9, 1, 1),    # easy: acted
        _r("t2", 0.0, "", 0), _r("t3", 0.0, "", 0),  # medium: abstained
        _r("t4", 0.0, "", 0), _r("t5", 0.0, "", 0),  # hard: abstained
    ]
    # mD: commits & correct on everything (defines difficulty).
    d_rows = [
        _r("t0", 0.9, 1, 1), _r("t1", 0.9, 1, 1),
        _r("t2", 0.9, 0, 1), _r("t3", 0.9, 0, 1),
        _r("t4", 0.9, 0, 1), _r("t5", 0.9, 0, 1),
    ]
    for s in (1, 2, 3, 4, 5):
        _write_csv(runs / "mC" / f"mC_s{s}.csv", list(c_rows))
        _write_csv(runs / "mD" / f"mD_s{s}.csv", list(d_rows))
    out = stratified_coords(str(runs), n_bins=3)
    # mC abstained on every hard-bin row -> n_acted == 0 and C is nan.
    hard_c = out["mC"]["hard"]
    assert hard_c["n_acted"] == 0
    assert math.isnan(hard_c["C"])
    assert hard_c["A"] == pytest.approx(0.0)
