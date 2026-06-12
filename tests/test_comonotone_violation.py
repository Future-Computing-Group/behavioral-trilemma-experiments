"""Tests for analysis/comonotone_violation.py (EXPERIMENT-PLAN §11.E-A.2).

Pair-level violation of prop:bon hypothesis (c) under the ORIGINAL Eq. 11
oracle payoff: among gate-cleared candidates, a pair with strictly higher
report and strictly lower score.
"""
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.comonotone_violation import violation_counts  # noqa: E402


def _c(r, y):
    return {"r": r, "y": y, "a": "x"}


def test_single_violating_pair():
    # G = {r=0.9 (y=0), r=0.8 (y=1)}; V(.9,0)=-0.81 < V(.8,1)=-0.04
    # within-gate (w_A constant in G): higher report, strictly lower score.
    comps = [_c(0.9, 0), _c(0.8, 1), _c(0.3, 1)]   # third is below gate
    out = violation_counts(comps, N=3, r_min=0.7, w_C=1.0, w_A=2.0)
    assert out == {"n_cleared": 2, "n_comparable_pairs": 1,
                   "n_violations": 1, "event": 1, "eligible": 1}


def test_no_violation_when_aligned():
    comps = [_c(0.9, 1), _c(0.8, 1)]   # V increasing in r when y=1
    out = violation_counts(comps, N=2, r_min=0.7, w_C=1.0, w_A=2.0)
    assert out["n_violations"] == 0 and out["event"] == 0 and out["eligible"] == 1


def test_ineligible_when_fewer_than_two_clear():
    comps = [_c(0.9, 0), _c(0.3, 1)]
    out = violation_counts(comps, N=2, r_min=0.7, w_C=1.0, w_A=2.0)
    assert out["eligible"] == 0 and out["n_comparable_pairs"] == 0


def test_equal_reports_not_comparable():
    comps = [_c(0.8, 0), _c(0.8, 1)]   # tie in r: (c) says "strictly higher"
    out = violation_counts(comps, N=2, r_min=0.7, w_C=1.0, w_A=2.0)
    assert out["n_comparable_pairs"] == 0 and out["event"] == 0


def test_w_A_invariance_within_gate():
    comps = [_c(0.9, 0), _c(0.8, 1)]
    a = violation_counts(comps, N=2, r_min=0.7, w_C=1.0, w_A=0.0)
    b = violation_counts(comps, N=2, r_min=0.7, w_C=1.0, w_A=4.0)
    assert a == b   # documented structural fact, §11.E-A.2


def test_candidate_set_is_first_N_only():
    # The 4th completion would form a violating pair but N=3 excludes it.
    comps = [_c(0.8, 1), _c(0.75, 1), _c(0.3, 0), _c(0.9, 0)]
    out = violation_counts(comps, N=3, r_min=0.7, w_C=1.0, w_A=2.0)
    assert out["n_cleared"] == 2 and out["n_violations"] == 0


def test_known_counts_on_handbuilt_three_cleared():
    # G = {(0.95,0), (0.85,1), (0.75,1)}; pairs (hi,lo) all comparable: 3.
    # V(.95,0)=-0.9025, V(.85,1)=-0.0225, V(.75,1)=-0.0625 (w_A cancels).
    # Violations: (.95 vs .85): -0.9025 < -0.0225 -> yes;
    #             (.95 vs .75): -0.9025 < -0.0625 -> yes;
    #             (.85 vs .75): -0.0225 > -0.0625 -> no.   Total = 2.
    comps = [_c(0.95, 0), _c(0.85, 1), _c(0.75, 1)]
    out = violation_counts(comps, N=3, r_min=0.7, w_C=1.0, w_A=1.0)
    assert out == {"n_cleared": 3, "n_comparable_pairs": 3,
                   "n_violations": 2, "event": 1, "eligible": 1}


def test_randomized_never_violates_when_all_y_equal():
    # With identical y across the cleared set, Eq. 11 ranks by |r - y|;
    # for y=1 that is increasing in r, so violations are impossible.
    rng = random.Random(20260612)
    for _ in range(200):
        n = rng.randint(2, 8)
        comps = [_c(round(rng.uniform(0.7, 1.0), 6), 1) for _ in range(n)]
        out = violation_counts(comps, N=n, r_min=0.7,
                               w_C=rng.uniform(0.1, 4.0),
                               w_A=rng.choice([0.0, 1.0, 4.0]))
        assert out["n_violations"] == 0


# ---------------------------------------------------------------------------
# Driver: aggregation over the §5.1 grid (tiny synthetic pools)
# ---------------------------------------------------------------------------

def test_driver_aggregates_event_and_pair_rates(tmp_path):
    import json
    from scripts.compute_comonotone_violation import compute_violation_summary

    # One-seed grid; 2 tasks. Task A has a guaranteed violating pair at
    # r_min=0.5/0.7 once N >= 2; task B never clears with 2 candidates.
    recs = [
        {"task_id": "a", "category": "arithmetic_easy", "seed": 42,
         "confidence_source": "logprob",
         "completions": [_c(0.9, 0), _c(0.8, 1)]},
        {"task_id": "b", "category": "arithmetic_easy", "seed": 42,
         "confidence_source": "logprob",
         "completions": [_c(0.95, 1), _c(0.3, 0)]},
    ]
    pools = tmp_path / "pools"
    pools.mkdir()
    with (pools / "pool_qwen2.5_7b_seed42_N32.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    out = compute_violation_summary(
        pools_dir=pools, n_values=[1, 2], w_ratios=[0, 4.0],
        r_mins=[0.7, 0.9], seeds=[42], w_C=1.0,
    )
    g = out["grid_540_view"]
    # Eligible (cell, task) pairs: only N=2, r_min=0.7, task a -> 2 w-levels.
    assert g["n_eligible"] == 2
    assert g["n_excluded"] == 14            # 2x2x2x2 = 16 total - 2 eligible
    assert g["event_rate"] == 1.0           # both eligible cells violate
    assert g["pair_rate"] == 1.0
    c = out["collapsed_90_view"]            # w dimension dropped
    assert c["n_eligible"] == 1 and c["event_rate"] == 1.0
