"""Tests for the §11 amendment 3 (E-C.2) no-oracle proxy selector.

Manuscript §exp-protocol: V_proxy = -w_C * r * (1 - r) + w_A * 1{r >= r_min}
(Bernoulli variance, NOT a proper scoring rule; y-independent), "included
only as a control quantifying the oracle/no-oracle gap". This wires the
long-defined src.scorer.proxy_payoff into Stage B as an argmax selector.

Pre-stated properties under test (EXPERIMENT-PLAN §11 E-C.2):
  (i)   formula drift-locked to src.scorer.proxy_payoff;
  (ii)  y-independent: flipping outcomes never changes the selection;
  (iii) argmax with first-index tie-breaking, over completions[:N];
  (iv)  19-column schema with payoff_mode='proxy', selection_mode='argmax'.
"""
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.select_from_pool import (  # noqa: E402
    SELECTOR_CHOICES,
    _SELECTORS,
    select_for_config,
)
from src.scorer import proxy_payoff  # noqa: E402


def _rec(completions):
    return {"task_id": "t1", "category": "arith_easy", "seed": 42,
            "confidence_source": "logprob", "completions": completions}


def _c(idx, r, y, a="x"):
    return {"idx": idx, "r": r, "y": y, "a": a}


def test_proxy_in_selector_registry():
    assert "proxy" in _SELECTORS
    assert "proxy" in SELECTOR_CHOICES


def test_drift_lock_against_src_scorer():
    # (i) the Stage-B proxy payoff must equal src.scorer.proxy_payoff on
    # a seeded grid (y must be ignored entirely).
    rng = random.Random(20260719)
    fn = _SELECTORS["proxy"]
    for _ in range(500):
        r = round(rng.uniform(0.01, 1.0), 6)
        y = rng.randint(0, 1)
        w_C = rng.choice([0.5, 1.0, 2.0])
        w_A = w_C * rng.choice([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
        r_min = rng.choice([0.5, 0.7, 0.9])
        assert fn(r, y, w_C, w_A, r_min) == proxy_payoff(r, w_C, w_A, r_min)


def test_y_independence_of_selection():
    # (ii) same r profile, all-y-flipped: identical selection.
    rng = random.Random(7)
    for _ in range(50):
        n = rng.randint(2, 32)
        rs = [round(rng.uniform(0.01, 1.0), 6) for _ in range(n)]
        a = _rec([_c(i, r, 1) for i, r in enumerate(rs)])
        b = _rec([_c(i, r, 0) for i, r in enumerate(rs)])
        ra = select_for_config(a, N=n, w_C=1.0, w_A=2.0, r_min=0.7,
                               w_ratio=2.0, selector="proxy")
        rb = select_for_config(b, N=n, w_C=1.0, w_A=2.0, r_min=0.7,
                               w_ratio=2.0, selector="proxy")
        assert ra["selected_index"] == rb["selected_index"]


def test_hand_built_fixture_prefers_extreme_gate_cleared():
    # r_min=0.7, w_C=1, w_A=2:
    #   idx0: r=0.75 y=1  V_proxy = -0.1875 + 2 = 1.8125
    #   idx1: r=0.95 y=0  V_proxy = -0.0475 + 2 = 1.9525  <- argmax
    #   idx2: r=0.40 y=1  V_proxy = -0.24        = -0.24
    # Proxy picks the extreme (wrong) report idx1; oracle picks idx0.
    rec = _rec([_c(0, 0.75, 1), _c(1, 0.95, 0), _c(2, 0.40, 1)])
    row = select_for_config(rec, N=3, w_C=1.0, w_A=2.0, r_min=0.7,
                            w_ratio=2.0, selector="proxy")
    assert row["selected_index"] == 1
    assert row["V_selected"] == proxy_payoff(0.95, 1.0, 2.0, 0.7)
    assert row["payoff_mode"] == "proxy"
    assert row["selection_mode"] == "argmax"
    assert row["y"] == 0
    assert row["brier"] == (0.95 - 0.0) ** 2
    row_o = select_for_config(rec, N=3, w_C=1.0, w_A=2.0, r_min=0.7,
                              w_ratio=2.0)
    assert row_o["selected_index"] == 0


def test_first_index_tie_breaking_and_first_N():
    # (iii) equal r (hence equal proxy score): first index wins; and only
    # completions[:N] participate.
    rec = _rec([_c(0, 0.9, 0), _c(1, 0.9, 1), _c(2, 1.0, 1)])
    row = select_for_config(rec, N=2, w_C=1.0, w_A=2.0, r_min=0.7,
                            w_ratio=2.0, selector="proxy")
    assert row["selected_index"] == 0  # idx2 (r=1.0) is outside N=2


def test_cli_accepts_proxy_selector(tmp_path):
    import csv
    from scripts.select_from_pool import main
    rc = main(["--config", "qwen2.5_7b_N4_w1.0_r0.7_s42",
               "--selector", "proxy", "--out-dir", str(tmp_path)])
    assert rc == 0
    out = tmp_path / "qwen2.5_7b_N4_w1.0_r0.7_s42.csv"
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 100
    assert all(r["payoff_mode"] == "proxy" for r in rows)
    assert all(r["selection_mode"] == "argmax" for r in rows)
