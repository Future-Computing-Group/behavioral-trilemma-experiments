"""TDD tests for analysis.tau_sweep (L.5 competence-decoupled analysis #1).

The tau-sweep re-classifies predictions with r_logprob < tau as abstained
and recomputes (H, C, A) at each tau, tracing a per-model trajectory
through (H, C, A) space with task difficulty held fixed (same model, same
rows, only the abstention rule moves).

Schema (logprob_xmodel): task, category, seed, r_logprob, answer, y, acted.
"""

import csv
import math
import pathlib

import pytest


LOGPROB_HEADER = ["task", "category", "seed", "r_logprob", "answer", "y",
                  "acted"]


def _write_logprob_csv(path: pathlib.Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOGPROB_HEADER)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({
                "task": r.get("task", f"t{i:03d}"),
                "category": r.get("category", "arithmetic_easy"),
                "seed": r.get("seed", 2000),
                "r_logprob": r.get("r_logprob", ""),
                "answer": r.get("answer", ""),
                "y": r.get("y", ""),
                "acted": r.get("acted", 0),
            })
    return str(path)


def _row(task, r, y, acted):
    return {"task": task, "r_logprob": r, "answer": "a" if acted else "",
            "y": y, "acted": acted}


@pytest.fixture
def hand_fixture(tmp_path):
    """4-row hand CSV with tightly known (H, C, A) at each tau.

    Rows:
      r=0.9, y=1, acted=1      (high-conf correct)
      r=0.3, y=0, acted=1      (low-conf wrong)
      r=0.7, y=1, acted=1      (medium-conf correct)
      r=0.0, y=0, acted=0      (originally abstained / parse-fail)

    At tau=0: 3 acted, A=0.75, H=2/4=0.5, briers=[0.01,0.09,0.09],
              C = 1 - 0.19/3 = 0.9367
    At tau=1: 0 acted, A=0, H=0, C=nan
    """
    csv_path = tmp_path / "hand.csv"
    _write_logprob_csv(csv_path, [
        _row("t0", 0.9, 1, 1),
        _row("t1", 0.3, 0, 1),
        _row("t2", 0.7, 1, 1),
        _row("t3", 0.0, 0, 0),
    ])
    return str(csv_path)


def test_tau_zero_reproduces_baseline(hand_fixture):
    from analysis.tau_sweep import tau_sweep_model
    from analysis.model_points import _logprob_point_metrics
    import csv as _csv
    with open(hand_fixture) as f:
        rows = list(_csv.DictReader(f))
    h, a, c, n_acted = _logprob_point_metrics(rows)

    out = tau_sweep_model([hand_fixture], taus=[0.0, 1.0])
    e0 = out[0]
    assert e0["tau"] == 0.0
    assert e0["H"] == pytest.approx(h)
    assert e0["A"] == pytest.approx(a)
    assert e0["C"] == pytest.approx(c)
    assert e0["n_acted"] == n_acted


def test_tau_one_gives_all_abstained(hand_fixture):
    from analysis.tau_sweep import tau_sweep_model
    # tau exactly 1.0: only r >= 1.0 survives -> none in the hand fixture.
    # But r=0.9 < 1.0; treat tau=1.0 as a strict threshold that abstains all.
    out = tau_sweep_model([hand_fixture], taus=[1.0])
    e = out[0]
    assert e["A"] == pytest.approx(0.0)
    assert e["H"] == pytest.approx(0.0)
    assert math.isnan(e["C"])
    assert e["n_acted"] == 0


def test_tau_05_threshold(hand_fixture):
    """At tau=0.5: r=0.9 (acted), r=0.7 (acted), r=0.3 (now abstained),
    r=0.0 (originally abstained). So A=2/4=0.5, H=2/4=0.5,
    briers=[0.01, 0.09], C = 1 - 0.05 = 0.95."""
    from analysis.tau_sweep import tau_sweep_model
    out = tau_sweep_model([hand_fixture], taus=[0.5])
    e = out[0]
    assert e["A"] == pytest.approx(0.5)
    assert e["H"] == pytest.approx(0.5)
    assert e["C"] == pytest.approx(0.95)
    assert e["n_acted"] == 2


def test_monotonicity_A_and_H_nonincreasing(hand_fixture):
    """As tau increases, A and H are monotone non-increasing."""
    from analysis.tau_sweep import tau_sweep_model
    taus = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    out = tau_sweep_model([hand_fixture], taus=taus)
    a_vals = [e["A"] for e in out]
    h_vals = [e["H"] for e in out]
    for i in range(1, len(a_vals)):
        assert a_vals[i] <= a_vals[i - 1] + 1e-12, (
            f"A not monotone at tau={taus[i]}: {a_vals}"
        )
        assert h_vals[i] <= h_vals[i - 1] + 1e-12, (
            f"H not monotone at tau={taus[i]}: {h_vals}"
        )


def test_C_monotone_nondecreasing_for_well_calibrated(tmp_path):
    """For a well-calibrated fixture (low-conf wrong, high-conf right),
    C(tau) is monotone non-decreasing — raising the threshold drops the
    least confident (and likely-wrong) acted rows, improving C."""
    from analysis.tau_sweep import tau_sweep_model
    # Well-calibrated in the sense the test wants: high-conf rows have
    # BETTER (lower) brier than low-conf rows, so raising tau (dropping
    # low-conf rows) strictly improves the surviving-rows brier mean and
    # therefore C. r=0.99,y=1 (brier 0.0001) > r=0.55,y=0 (brier 0.3025).
    csv_path = tmp_path / "wc.csv"
    _write_logprob_csv(csv_path, [
        _row("t0", 0.99, 1, 1),   # brier 0.0001 — well-calibrated correct
        _row("t1", 0.95, 1, 1),   # brier 0.0025
        _row("t2", 0.55, 0, 1),   # brier 0.3025 — barely-confident wrong
        _row("t3", 0.35, 0, 1),   # brier 0.1225 — low-conf wrong
    ])
    taus = [0.0, 0.4, 0.6, 0.97]
    out = tau_sweep_model([str(csv_path)], taus=taus)
    c_vals = [e["C"] for e in out]
    # Drop nan from comparison (only happens when no acted rows).
    prev = -math.inf
    for c, t in zip(c_vals, taus):
        if math.isnan(c):
            continue
        assert c + 1e-12 >= prev, (
            f"C not monotone non-decreasing at tau={t}: {c_vals}"
        )
        prev = c


def test_originally_abstained_stays_abstained(tmp_path):
    """A row with acted==0 in the CSV must NOT become acted at any tau,
    even if r_logprob >= tau. (Parse-fail rows can never be re-acted.)"""
    from analysis.tau_sweep import tau_sweep_model
    csv_path = tmp_path / "absten.csv"
    # Row with acted=0 but high r_logprob — must stay abstained at every tau.
    _write_logprob_csv(csv_path, [
        _row("t0", 0.99, 1, 0),   # originally abstained; high r
        _row("t1", 0.99, 1, 0),
        _row("t2", 0.5, 1, 1),
        _row("t3", 0.5, 0, 1),
    ])
    out = tau_sweep_model([str(csv_path)], taus=[0.0, 0.3, 0.6, 1.0])
    # n_acted at tau=0 should be 2, not 4
    assert out[0]["n_acted"] == 2
    # at tau=0.6 the r=0.5 rows are dropped too -> 0 acted, NOT 2 (the
    # originally-abstained rows must never count).
    e_06 = next(e for e in out if e["tau"] == 0.6)
    assert e_06["n_acted"] == 0


def test_default_taus_resolution():
    """The 12-point default tau ladder is exposed for the figure code."""
    from analysis.tau_sweep import DEFAULT_TAUS
    assert DEFAULT_TAUS == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                            0.9, 0.95, 1.0]


def test_all_logprob_tau_sweeps_groups_by_subdir(tmp_path):
    """all_logprob_tau_sweeps groups CSVs by <model>/ subdirectory."""
    from analysis.tau_sweep import all_logprob_tau_sweeps
    runs = tmp_path / "logprob_xmodel"
    block_a = [_row("t", 0.9, 1, 1), _row("t", 0.3, 0, 1)] * 60
    for s in (1, 2, 3, 4, 5):
        _write_logprob_csv(runs / "lpA" / f"lpA_s{s}.csv", list(block_a))
    block_b = [_row("t", 0.8, 1, 1)] * 120
    for s in (1, 2):
        _write_logprob_csv(runs / "lpB" / f"lpB_s{s}.csv", list(block_b))
    out = all_logprob_tau_sweeps(str(runs), taus=[0.0, 0.5, 1.0])
    assert set(out) == {"lpA", "lpB"}
    for mid in out:
        traj = out[mid]
        assert len(traj) == 3
        assert traj[0]["tau"] == 0.0
        assert traj[-1]["tau"] == 1.0
