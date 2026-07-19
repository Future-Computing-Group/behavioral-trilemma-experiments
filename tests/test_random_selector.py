"""Tests for the §11 amendment 3 (E-C.1) uniform-random control selector.

The manuscript's third control (§sec:exp-controls): uniform random
selection among the N pooled completions — preserves the sampling
distribution while removing the optimization pressure. Deterministically
seeded from the config/seed fields only (SHA-256 over the descriptor
`random|<task_id>|s<seed>|N<N>|w<w>|r<r>`), so re-runs are byte-stable.

Pre-stated properties under test (EXPERIMENT-PLAN §11 E-C.1):
  (i)   deterministic: identical inputs -> identical selection, always;
  (ii)  payoff-blind: y values (and hence every payoff) never affect
        the selection;
  (iii) uniform over the first-N candidate set (all indices reachable,
        empirical frequencies consistent with uniformity);
  (iv)  config-sensitive: the draw stream differs across w/r/N cells so
        weight contrasts are noisy nulls, not degenerate 0/0 contrasts;
  (v)   recorded V_selected is the ORACLE score of the drawn completion
        (comparability only; unused for selection).
"""
import collections
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.select_from_pool import (  # noqa: E402
    SELECTOR_CHOICES,
    _oracle_payoff,
    random_index,
    select_for_config,
)


def _rec(completions, task_id="t1", seed=42):
    return {"task_id": task_id, "category": "arith_easy", "seed": seed,
            "confidence_source": "logprob", "completions": completions}


def _c(idx, r, y, a="x"):
    return {"idx": idx, "r": r, "y": y, "a": a}


def test_random_in_selector_choices():
    assert "random" in SELECTOR_CHOICES


def test_deterministic_repeat_calls():
    # (i) identical inputs -> identical index, on repeated calls.
    for task in ("arith_easy_01", "code_algo_07", "fact_obscure_15"):
        for seed in (42, 123, 0):
            first = random_index(task_id=task, seed=seed, N=32,
                                 w_ratio=1.0, r_min=0.7, n_candidates=32)
            for _ in range(3):
                assert random_index(task_id=task, seed=seed, N=32,
                                    w_ratio=1.0, r_min=0.7,
                                    n_candidates=32) == first


def test_known_regression_values_are_byte_stable():
    # Pin three concrete draws: any change to the seeding scheme (hash,
    # key format, RNG) breaks byte-stability of the archived control CSVs
    # and MUST show up here.
    pinned = [
        random_index(task_id="arith_easy_01", seed=42, N=32, w_ratio=1.0,
                     r_min=0.7, n_candidates=32),
        random_index(task_id="code_algo_07", seed=123, N=8, w_ratio=0,
                     r_min=0.5, n_candidates=8),
        random_index(task_id="fact_obscure_15", seed=0, N=4, w_ratio=4.0,
                     r_min=0.9, n_candidates=4),
    ]
    # Recompute independently (the contract, spelled out).
    import hashlib
    import random as _random
    expect = []
    for task, seed, N, w, r, n in [
        ("arith_easy_01", 42, 32, 1.0, 0.7, 32),
        ("code_algo_07", 123, 8, 0, 0.5, 8),
        ("fact_obscure_15", 0, 4, 4.0, 0.9, 4),
    ]:
        w_str = "0" if w == 0 else str(w)
        key = f"random|{task}|s{seed}|N{N}|w{w_str}|r{r}"
        seed64 = int.from_bytes(
            hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
        expect.append(_random.Random(seed64).randrange(n))
    assert pinned == expect


def test_payoff_blind_y_never_matters():
    # (ii) flipping every y leaves the selection unchanged.
    comps_a = [_c(i, 0.1 + 0.028 * i, 1) for i in range(32)]
    comps_b = [_c(i, 0.1 + 0.028 * i, 0) for i in range(32)]
    for w_ratio in (0, 0.25, 4.0):
        ra = select_for_config(_rec(comps_a), N=32, w_C=1.0,
                               w_A=1.0 * w_ratio, r_min=0.7,
                               w_ratio=w_ratio, selector="random")
        rb = select_for_config(_rec(comps_b), N=32, w_C=1.0,
                               w_A=1.0 * w_ratio, r_min=0.7,
                               w_ratio=w_ratio, selector="random")
        assert ra["selected_index"] == rb["selected_index"]


def test_uniform_over_first_N():
    # (iii) all indices 0..N-1 reachable; frequencies roughly uniform
    # over 2000 distinct (task, seed) keys at N=4 (expected 500 each;
    # binomial 3-sigma half-width ~65).
    counts = collections.Counter(
        random_index(task_id=f"task_{k:04d}", seed=s, N=4, w_ratio=1.0,
                     r_min=0.7, n_candidates=4)
        for k in range(400) for s in (42, 123, 456, 789, 0)
    )
    assert set(counts) == {0, 1, 2, 3}
    for idx in range(4):
        assert 400 <= counts[idx] <= 600, counts


def test_config_sensitivity_across_w_levels():
    # (iv) across 100 tasks, the draws at w=0 and w=4.0 must differ on at
    # least one task (they are independent streams; collision prob of
    # perfect agreement at N=32 is 32^-100).
    diffs = sum(
        random_index(task_id=f"task_{k:04d}", seed=42, N=32, w_ratio=0,
                     r_min=0.7, n_candidates=32)
        != random_index(task_id=f"task_{k:04d}", seed=42, N=32,
                        w_ratio=4.0, r_min=0.7, n_candidates=32)
        for k in range(100)
    )
    assert diffs > 0


def test_row_schema_and_recorded_oracle_score():
    # (v) payoff_mode/selection_mode markers; V_selected = oracle score
    # of the drawn completion; brier/gate fields consistent.
    comps = [_c(i, 0.05 + 0.03 * i, i % 2) for i in range(32)]
    row = select_for_config(_rec(comps, task_id="arith_easy_01", seed=42),
                            N=32, w_C=1.0, w_A=2.0, r_min=0.7,
                            w_ratio=2.0, selector="random")
    assert row["payoff_mode"] == "random"
    assert row["selection_mode"] == "uniform_random"
    i = row["selected_index"]
    assert 0 <= i < 32
    chosen = comps[i]
    assert row["r_selected"] == chosen["r"]
    assert row["y"] == chosen["y"]
    assert row["V_selected"] == _oracle_payoff(chosen["r"], chosen["y"],
                                               1.0, 2.0, 0.7)
    assert row["brier"] == (chosen["r"] - chosen["y"]) ** 2
    assert row["gate_cleared"] == (1 if chosen["r"] >= 0.7 else 0)


def test_respects_first_N_subset():
    # Selection must come from completions[:N] only. With N=2 the index
    # is in {0, 1} for every config.
    comps = [_c(i, 0.5 + 0.01 * i, 1) for i in range(32)]
    for w_ratio in (0, 1.0, 4.0):
        for r_min in (0.5, 0.7, 0.9):
            row = select_for_config(
                _rec(comps, task_id="t9", seed=789), N=2, w_C=1.0,
                w_A=w_ratio, r_min=r_min, w_ratio=w_ratio,
                selector="random")
            assert row["selected_index"] in (0, 1)
            assert row["n_candidates"] == 2


def test_cli_accepts_random_selector(tmp_path):
    # One real-pool cell; logical rows via csv.DictReader (a_selected can
    # embed newlines). Byte-stability: a second run is identical.
    import csv
    from scripts.select_from_pool import main
    rc = main(["--config", "qwen2.5_7b_N4_w1.0_r0.7_s42",
               "--selector", "random", "--out-dir", str(tmp_path / "a")])
    assert rc == 0
    out = tmp_path / "a" / "qwen2.5_7b_N4_w1.0_r0.7_s42.csv"
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 100
    assert all(r["payoff_mode"] == "random" for r in rows)
    assert all(r["selection_mode"] == "uniform_random" for r in rows)
    rc2 = main(["--config", "qwen2.5_7b_N4_w1.0_r0.7_s42",
                "--selector", "random", "--out-dir", str(tmp_path / "b")])
    assert rc2 == 0
    assert out.read_bytes() == (
        tmp_path / "b" / "qwen2.5_7b_N4_w1.0_r0.7_s42.csv").read_bytes()
