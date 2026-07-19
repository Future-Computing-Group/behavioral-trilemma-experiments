"""Tests for the EXPERIMENT-PLAN §11.E-A.1 comonotone-compliant selector.

V_c equals the oracle selection payoff (manuscript eq.
(eq:selection-payoff)) below the gate, and replaces the
realized outcome y with the constant 1 on {r >= r_min}, making the score
strictly increasing in r on the cleared region regardless of y — so
prop:bon hypothesis (c) holds surely.
"""
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.select_from_pool import (  # noqa: E402
    comonotone_payoff,
    _oracle_payoff,
    select_for_config,
)


def _rec(completions):
    return {"task_id": "t1", "category": "arith_easy", "seed": 42,
            "confidence_source": "logprob", "completions": completions}


def _c(idx, r, y, a="x"):
    return {"idx": idx, "r": r, "y": y, "a": a}


def test_comonotone_payoff_y_free_and_increasing_above_gate():
    # Above the gate: y-independent and strictly increasing in r.
    rs = [0.70, 0.75, 0.80, 0.95, 0.999]
    for y in (0, 1):
        vals = [comonotone_payoff(r, y, w_C=1.0, w_A=2.0, r_min=0.7) for r in rs]
        assert vals == sorted(vals) and len(set(vals)) == len(vals)
    assert comonotone_payoff(0.8, 0, 1.0, 2.0, 0.7) == \
           comonotone_payoff(0.8, 1, 1.0, 2.0, 0.7)


def test_comonotone_payoff_equals_oracle_below_gate():
    for r, y in [(0.10, 0), (0.69, 1), (0.30, 1)]:
        assert comonotone_payoff(r, y, 1.0, 2.0, 0.7) == \
               _oracle_payoff(r, y, 1.0, 2.0, 0.7)


def test_comonotone_monotone_on_randomized_candidate_sets():
    # §11.E-A.1: V_c strictly increasing in r on the cleared region,
    # for every (w_C, w_A, r_min) cell and ANY y labelling. ≥200 seeded
    # random draws.
    rng = random.Random(20260612)
    for _ in range(250):
        w_C = rng.uniform(0.1, 4.0)
        w_A = rng.choice([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
        r_min = rng.choice([0.5, 0.7, 0.9])
        # Two distinct cleared reports with arbitrary outcomes.
        r_lo = rng.uniform(r_min, 0.999)
        r_hi = rng.uniform(r_lo + 1e-6, 1.0)
        y_lo, y_hi = rng.randint(0, 1), rng.randint(0, 1)
        v_lo = comonotone_payoff(r_lo, y_lo, w_C, w_A, r_min)
        v_hi = comonotone_payoff(r_hi, y_hi, w_C, w_A, r_min)
        assert v_hi > v_lo, (r_lo, r_hi, y_lo, y_hi, w_C, w_A, r_min)


def test_selector_flag_changes_selection():
    # Oracle picks idx 1 (r=0.75, y=1: V=-0.0625+wA beats idx 0's -0.81+wA);
    # comonotone must pick idx 0 (higher r above gate, y ignored).
    rec = _rec([_c(0, 0.90, 0), _c(1, 0.75, 1)])
    row_o = select_for_config(rec, N=2, w_C=1.0, w_A=2.0, r_min=0.7,
                              w_ratio=2.0)
    row_c = select_for_config(rec, N=2, w_C=1.0, w_A=2.0, r_min=0.7,
                              w_ratio=2.0, selector="comonotone")
    assert row_o["selected_index"] == 1
    assert row_c["selected_index"] == 0
    assert row_c["payoff_mode"] == "comonotone"
    assert row_o["payoff_mode"] == "oracle"


def test_selector_default_is_oracle_byte_compatible():
    rec = _rec([_c(0, 0.90, 0), _c(1, 0.75, 1)])
    assert select_for_config(rec, N=2, w_C=1.0, w_A=2.0, r_min=0.7,
                             w_ratio=2.0)["payoff_mode"] == "oracle"
