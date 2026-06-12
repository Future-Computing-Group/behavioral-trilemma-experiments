"""Tests for the §11 E-A amendment 1 (E-A.1b) comonotone REARRANGEMENT selector.

Per (cell, task) pool: take the multiset of ORACLE Eq.-11 scores of the
gate-cleared candidates, sort the cleared candidates by r (ascending, ties
by original index), sort the scores ascending, reassign rank-to-rank;
uncleared candidates keep their oracle scores; argmax over the whole pool
with first-index tie-breaking.

Pre-stated properties under test (EXPERIMENT-PLAN §11 E-A amendment 1):
  (i)   prop:bon hypothesis (c) holds pairwise on the cleared set;
  (ii)  the cleared-set score multiset is preserved exactly;
  (iii) uncleared candidates are untouched;
  (iv)  deterministic under the tie rules.
"""
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.select_from_pool import (  # noqa: E402
    _oracle_payoff,
    rearrangement_scores,
    select_for_config,
)


def _rec(completions):
    return {"task_id": "t1", "category": "arith_easy", "seed": 42,
            "confidence_source": "logprob", "completions": completions}


def _c(idx, r, y, a="x"):
    return {"idx": idx, "r": r, "y": y, "a": a}


def _random_pool(rng):
    """Random candidate pool with clip atoms at 0.01/1.0 to exercise ties."""
    n = rng.randint(2, 32)
    completions = []
    for i in range(n):
        u = rng.random()
        if u < 0.15:
            r = 1.0          # clip atom (ties at the top)
        elif u < 0.25:
            r = 0.01         # clip atom (ties at the bottom)
        else:
            r = round(rng.uniform(0.01, 1.0), rng.choice([2, 3, 6]))
        completions.append(_c(i, r, rng.randint(0, 1)))
    w_C = 1.0
    w_A = w_C * rng.choice([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    r_min = rng.choice([0.5, 0.7, 0.9])
    return completions, w_C, w_A, r_min


def test_property_suite_on_seeded_random_pools():
    # >= 200 seeded random pools; all four pre-stated properties.
    rng = random.Random(20260612)
    n_pools_with_cleared_pairs = 0
    for _ in range(250):
        completions, w_C, w_A, r_min = _random_pool(rng)
        oracle = [_oracle_payoff(c["r"], c["y"], w_C, w_A, r_min)
                  for c in completions]
        scores = rearrangement_scores(completions, w_C=w_C, w_A=w_A,
                                      r_min=r_min)
        assert len(scores) == len(completions)
        cleared = [i for i, c in enumerate(completions)
                   if c["r"] >= r_min]
        uncleared = [i for i in range(len(completions)) if i not in cleared]

        # (i) hypothesis (c): strictly higher r never gets strictly
        # lower score, pairwise on the cleared set.
        for a in cleared:
            for b in cleared:
                if completions[a]["r"] > completions[b]["r"]:
                    assert scores[a] >= scores[b], (a, b, w_A, r_min)
        if len(cleared) >= 2:
            n_pools_with_cleared_pairs += 1

        # (ii) cleared-set score multiset preserved EXACTLY.
        assert sorted(scores[i] for i in cleared) == \
               sorted(oracle[i] for i in cleared)

        # (iii) uncleared untouched (exact float identity, per index).
        for i in uncleared:
            assert scores[i] == oracle[i]

        # (iv) deterministic: identical output on a second call.
        assert rearrangement_scores(completions, w_C=w_C, w_A=w_A,
                                    r_min=r_min) == scores
    # The property suite must actually have exercised (i) non-vacuously.
    assert n_pools_with_cleared_pairs >= 100


def test_tie_rule_equal_r_assigns_by_original_index():
    # Two cleared candidates with EQUAL r: the earlier original index gets
    # the smaller rank, hence the smaller (or equal) score.
    completions = [_c(0, 0.8, 0), _c(1, 0.8, 1)]  # oracle: ~1.36 vs 1.96
    v0 = _oracle_payoff(0.8, 0, 1.0, 2.0, 0.7)  # the smaller score
    v1 = _oracle_payoff(0.8, 1, 1.0, 2.0, 0.7)  # the larger score
    assert v0 < v1
    scores = rearrangement_scores(completions, w_C=1.0, w_A=2.0, r_min=0.7)
    assert scores == [v0, v1]
    # And twice more for determinism under the tie rule.
    assert rearrangement_scores(completions, w_C=1.0, w_A=2.0,
                                r_min=0.7) == [v0, v1]


def test_whole_pool_argmax_first_index_tie_breaking():
    # Two cleared candidates, same r, same y -> identical assigned scores;
    # the selection must take the FIRST index.
    rec = _rec([_c(0, 0.9, 1), _c(1, 0.9, 1)])
    row = select_for_config(rec, N=2, w_C=1.0, w_A=2.0, r_min=0.7,
                            w_ratio=2.0, selector="rearrangement")
    assert row["selected_index"] == 0


def test_hand_built_fixture_known_expected_selection():
    # r_min=0.7, w_C=1, w_A=2.
    #   idx0: r=0.75 y=1 cleared  V_oracle = 2 - 0.0625 = 1.9375
    #   idx1: r=0.95 y=0 cleared  V_oracle = 2 - 0.9025 = 1.0975
    #   idx2: r=0.80 y=1 cleared  V_oracle = 2 - 0.04   = 1.96
    #   idx3: r=0.40 y=0 uncleared V_oracle = -0.16
    # r-order (asc): idx0, idx2, idx1; scores (asc): 1.0975, 1.9375, 1.96
    # -> idx0=1.0975, idx2=1.9375, idx1=1.96; argmax = idx1 (max-r cleared).
    rec = _rec([_c(0, 0.75, 1), _c(1, 0.95, 0), _c(2, 0.80, 1),
                _c(3, 0.40, 0)])
    v0 = _oracle_payoff(0.75, 1, 1.0, 2.0, 0.7)   # 1.9375
    v1 = _oracle_payoff(0.95, 0, 1.0, 2.0, 0.7)   # 1.0975
    v2 = _oracle_payoff(0.80, 1, 1.0, 2.0, 0.7)   # 1.96
    v3 = _oracle_payoff(0.40, 0, 1.0, 2.0, 0.7)   # -0.16
    assert v1 < v0 < v2 and v3 < 0
    scores = rearrangement_scores(rec["completions"], w_C=1.0, w_A=2.0,
                                  r_min=0.7)
    assert scores == [v1, v2, v0, v3]
    row = select_for_config(rec, N=4, w_C=1.0, w_A=2.0, r_min=0.7,
                            w_ratio=2.0, selector="rearrangement")
    assert row["selected_index"] == 1
    assert row["r_selected"] == 0.95
    assert row["y"] == 0
    assert row["V_selected"] == 1.96
    assert row["brier"] == (0.95 - 0.0) ** 2
    assert row["payoff_mode"] == "rearrangement"
    # Oracle on the same pool picks idx2 — the rearrangement differs.
    row_o = select_for_config(rec, N=4, w_C=1.0, w_A=2.0, r_min=0.7,
                              w_ratio=2.0)
    assert row_o["selected_index"] == 2


def test_uncleared_can_still_win_at_w_zero():
    # The cleared-vs-uncleared margin is untouched: with w_A=0 an
    # uncleared near-perfect-confidence wrong answer (V ~ -0.0001) beats
    # a mediocre cleared candidate (V = -(1-0.75)^2 = -0.0625).
    rec = _rec([_c(0, 0.75, 1), _c(1, 0.01, 0)])
    row = select_for_config(rec, N=2, w_C=1.0, w_A=0.0, r_min=0.7,
                            w_ratio=0.0, selector="rearrangement")
    assert row["selected_index"] == 1
    assert row["gate_cleared"] == 0


def test_cli_accepts_rearrangement_selector(tmp_path):
    # Regression for the argparse choices list: one real-pool cell.
    # NOTE: a_selected fields can embed newlines (quoted CSV), so logical
    # rows must be counted with csv.DictReader, not splitlines().
    import csv
    from scripts.select_from_pool import main
    rc = main(["--config", "qwen2.5_7b_N4_w1.0_r0.7_s42",
               "--selector", "rearrangement",
               "--out-dir", str(tmp_path)])
    assert rc == 0
    out = tmp_path / "qwen2.5_7b_N4_w1.0_r0.7_s42.csv"
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 100
    assert all(r["payoff_mode"] == "rearrangement" for r in rows)
    assert all(r["selection_mode"] == "argmax" for r in rows)
