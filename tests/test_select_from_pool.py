"""Tests for scripts/select_from_pool.py — Stage-B selector reconstruction.

The selector consumes pre-generated pool jsonl records (one per (task, seed),
each carrying 32 completions with logprob-confidence + parsed answer + y)
and emits the 19-column CSV row that the manuscript's Table-1 pipeline
expects.

These tests pin the selector's contract:
  - subset = first N completions (not random)
  - V = -w_C*(r-y)^2 + w_A*1{r >= r_min}, the manuscript's selection payoff (eq. (eq:selection-payoff))
  - argmax with first-index tie-breaking
  - n_parsed counts completions[:N] where `a is not None`
  - a_selected = "" when a is None
  - schema is the exact 19-column order found in shipped CSVs
  - regenerated rows are byte-equal to the shipped CSVs (the load-bearing
    invariant — anything else is a contract violation we want to catch).
"""
import csv
import json
import pathlib

import pytest

import sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.select_from_pool import (
    select_for_config,
    SHIPPED_FIELDS,
)


# ---------------------------------------------------------------------------
# Pure-function tests (hand-computed examples)
# ---------------------------------------------------------------------------

def _mk_pool(task_id="t1", seed=0, completions=None):
    """Construct a pool record with the schema observed in the real data."""
    return {
        "task_id": task_id,
        "category": "arithmetic_easy",
        "seed": seed,
        "confidence_source": "logprob",
        "completions": completions or [],
    }


def _mk_completion(idx, r, y, a="42", raw=None):
    return {
        "idx": idx,
        "raw": raw if raw is not None else f"CONFIDENCE: {r}\nANSWER: {a}",
        "r": r,
        "r_verbalized": 1.0,
        "r_logprob": r,
        "mean_logprob": -1e-5,
        "a": a,
        "y": y,
    }


def test_n1_w0_correct_high_confidence():
    # N=1, w_C=1, w_A=0, r_min=0.5
    # one completion: r=0.9, y=1
    # V = -(0.9-1)^2 + 0 = -0.01; brier = 0.01; gate_cleared = 1 (0.9>=0.5)
    pool = _mk_pool(completions=[_mk_completion(0, r=0.9, y=1, a="42")])
    row = select_for_config(pool, N=1, w_C=1.0, w_A=0.0, r_min=0.5, w_ratio=0)
    assert row["n_candidates"] == 1
    assert row["n_parsed"] == 1
    assert row["selected_index"] == 0
    assert row["r_selected"] == 0.9
    assert row["a_selected"] == "42"
    assert row["y"] == 1
    assert row["V_selected"] == pytest.approx(-0.01)
    assert row["brier"] == pytest.approx(0.01)
    assert row["gate_cleared"] == 1


def test_subset_is_first_N_not_all_32():
    # 4 completions; N=2 must consider only idx 0,1 (NOT idx 2,3 even if they
    # have higher V). This guards against re-implementing as "best of all 32"
    # which would silently change the manuscript's N-axis interpretation.
    completions = [
        _mk_completion(0, r=0.6, y=1, a="42"),   # V=-0.16+1=0.84 at w_A=1,r_min=0.5
        _mk_completion(1, r=0.55, y=1, a="42"),  # V=-0.2025+1=0.7975
        _mk_completion(2, r=0.99, y=1, a="42"),  # would dominate if N=4
        _mk_completion(3, r=0.95, y=1, a="42"),
    ]
    row = select_for_config(_mk_pool(completions=completions),
                            N=2, w_C=1.0, w_A=1.0, r_min=0.5, w_ratio=1.0)
    assert row["selected_index"] == 0  # idx 0 wins among the first 2
    assert row["r_selected"] == 0.6


def test_argmax_tie_broken_by_first_index():
    # Two completions with identical V (r=1.0, y=1, w_A=0 → V=0 both).
    # Tie-breaking convention: first index wins (Python max-of-enumerated).
    completions = [
        _mk_completion(0, r=1.0, y=1, a="A"),
        _mk_completion(1, r=1.0, y=1, a="B"),
    ]
    row = select_for_config(_mk_pool(completions=completions),
                            N=2, w_C=1.0, w_A=0.0, r_min=0.5, w_ratio=0)
    assert row["selected_index"] == 0
    assert row["a_selected"] == "A"


def test_a_none_included_in_argmax_and_n_parsed_counts_r_not_a():
    # CONVENTION pinned to the shipped CSV's behavior for
    # code_simple_11 seed=456 (the one a=None completion in 16,000):
    # the a=None completion IS included in argmax (selected at several
    # configs in the shipped data), and ``n_parsed`` counts completions
    # whose CONFIDENCE r is non-None, NOT whose answer a is non-None.
    # In the logprob path r is always present so n_parsed == n_candidates.
    completions = [
        _mk_completion(0, r=0.3, y=0, a=None, raw="malformed"),  # V = -0.09
        _mk_completion(1, r=0.6, y=1, a="42"),                    # V = -0.16
    ]
    row = select_for_config(_mk_pool(completions=completions),
                            N=2, w_C=1.0, w_A=0.0, r_min=0.5, w_ratio=0)
    assert row["n_parsed"] == 2  # both have r; n_parsed does NOT exclude a=None
    assert row["selected_index"] == 0  # higher V wins (-0.09 > -0.16)
    assert row["a_selected"] == ""  # a=None serialised as empty string


def test_a_none_serialised_when_selected():
    # If the a=None completion wins the argmax (e.g., it has y=0 and a very
    # low r and the other one has y=0 too but r higher → both negative V,
    # the lower-|r| one wins). a_selected must be "" not None or "None".
    completions = [
        _mk_completion(0, r=0.1, y=0, a=None, raw="malformed"),  # V=-0.01
        _mk_completion(1, r=0.9, y=0, a="bad"),                   # V=-0.81
    ]
    row = select_for_config(_mk_pool(completions=completions),
                            N=2, w_C=1.0, w_A=0.0, r_min=0.5, w_ratio=0)
    assert row["selected_index"] == 0
    assert row["a_selected"] == ""


def test_gate_cleared_uses_r_selected_not_argmax():
    # gate_cleared is reported for the SELECTED completion, not the one with
    # max r. Concretely: w_A=10 dominates, so a completion with r=0.4 (below
    # r_min=0.5) would NOT clear the gate, even if it has y=1 and competes
    # with a y=0 completion at r=0.51.
    completions = [
        _mk_completion(0, r=0.4, y=1, a="X"),   # V = -(-0.6)^2 + 0 = -0.36
        _mk_completion(1, r=0.51, y=0, a="Y"),  # V = -(0.51)^2 + 10 = 9.74
    ]
    row = select_for_config(_mk_pool(completions=completions),
                            N=2, w_C=1.0, w_A=10.0, r_min=0.5, w_ratio=10.0)
    assert row["selected_index"] == 1
    assert row["r_selected"] == 0.51
    assert row["gate_cleared"] == 1


def test_gate_not_cleared_when_below_threshold():
    completions = [_mk_completion(0, r=0.4, y=1, a="X")]
    row = select_for_config(_mk_pool(completions=completions),
                            N=1, w_C=1.0, w_A=0.0, r_min=0.5, w_ratio=0)
    assert row["gate_cleared"] == 0


def test_schema_exact_19_columns():
    pool = _mk_pool(completions=[_mk_completion(0, r=0.9, y=1, a="42")])
    row = select_for_config(pool, N=1, w_C=1.0, w_A=0.0, r_min=0.5, w_ratio=0)
    assert list(row.keys()) == SHIPPED_FIELDS, \
        f"key order drift; got {list(row.keys())!r} vs {SHIPPED_FIELDS!r}"
    assert len(SHIPPED_FIELDS) == 19


# ---------------------------------------------------------------------------
# Real-data integration: regenerate a shipped row, demand exact equality
# ---------------------------------------------------------------------------

RESULTS_DIR = ROOT / "experiment_output" / "raw_runs" / "logprob" / "results"
POOLS_DIR = ROOT / "experiment_output" / "raw_runs" / "logprob" / "pools"


def _load_pool_record(seed, task_id):
    pool_path = POOLS_DIR / f"pool_qwen2.5_7b_seed{seed}_N32.jsonl"
    for line in pool_path.open():
        d = json.loads(line)
        if d["task_id"] == task_id:
            return d
    raise KeyError(f"{task_id} not in pool seed={seed}")


def _load_shipped_row(csv_name, task_id):
    with (RESULTS_DIR / csv_name).open() as f:
        for row in csv.DictReader(f):
            if row["task_id"] == task_id:
                return row
    raise KeyError(f"{task_id} not in {csv_name}")


def _row_to_csv_str(row):
    # Convert the regenerated dict row to the exact string repr csv would
    # emit, so we can compare against the shipped string-level row.
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SHIPPED_FIELDS)
    w.writerow(row)
    return buf.getvalue().rstrip("\r\n")


@pytest.mark.parametrize("csv_name,task_id,N,w_ratio,w_A,r_min,seed", [
    ("qwen2.5_7b_N1_w0_r0.5_s0.csv",    "arith_easy_01", 1,  0,   0.0, 0.5,  0),
    ("qwen2.5_7b_N1_w0_r0.5_s0.csv",    "arith_easy_02", 1,  0,   0.0, 0.5,  0),
    ("qwen2.5_7b_N4_w1.0_r0.7_s42.csv", "arith_easy_01", 4,  1.0, 1.0, 0.7, 42),
    ("qwen2.5_7b_N4_w1.0_r0.7_s42.csv", "arith_easy_02", 4,  1.0, 1.0, 0.7, 42),
    ("qwen2.5_7b_N4_w1.0_r0.7_s42.csv", "arith_easy_03", 4,  1.0, 1.0, 0.7, 42),
])
def test_regenerated_row_byte_equal_to_shipped(csv_name, task_id, N, w_ratio, w_A, r_min, seed):
    """Load real pool record + run the selector + compare row-string to
    the shipped CSV row. Byte-equality is the load-bearing contract."""
    pool = _load_pool_record(seed, task_id)
    regen = select_for_config(pool, N=N, w_C=1.0, w_A=w_A, r_min=r_min, w_ratio=w_ratio)
    shipped = _load_shipped_row(csv_name, task_id)
    # Compare field-by-field with type coercion (CSV reads strings).
    for k in SHIPPED_FIELDS:
        sv = shipped[k]
        rv = regen[k]
        # Stringify regen the same way csv.DictWriter would, for exact match.
        if isinstance(rv, float):
            rv_s = repr(rv)
        elif rv is None or rv == "":
            rv_s = ""
        else:
            rv_s = str(rv)
        assert rv_s == sv, \
            f"{csv_name}:{task_id} field {k!r}: regen={rv_s!r} shipped={sv!r}"


# ---------------------------------------------------------------------------
# Cross-model pool-dir resolution (§11 E-B; regression for the 2026-06-13
# bug where main() loaded non-default models' pools from the qwen dir)
# ---------------------------------------------------------------------------

def test_pools_dir_for_default_model_is_legacy_path():
    from scripts.select_from_pool import pools_dir_for, POOLS_DIR
    assert pools_dir_for("qwen2.5_7b") == POOLS_DIR


def test_pools_dir_for_other_model_is_per_model_dir():
    from scripts.select_from_pool import pools_dir_for, ROOT
    got = pools_dir_for("gemma2_9b")
    expected = ROOT / "experiment_output" / "raw_runs" / "logprob-gemma2_9b" / "pools"
    assert got == expected, f"{got} != {expected}"


def test_cross_model_config_run_loads_per_model_pool(tmp_path):
    """End-to-end: a non-default --model-slug run resolves and loads the
    per-model pool (not the qwen dir). Exercises the production load path
    the 2026-06-13 regression bypassed (L79)."""
    import subprocess
    import sys as _sys
    slug = "testmodel_1b"
    pools = ROOT / "experiment_output" / "raw_runs" / f"logprob-{slug}" / "pools"
    pools.mkdir(parents=True, exist_ok=True)
    rec = _mk_pool(task_id="t1", seed=42,
                   completions=[_mk_completion(0, 0.9, 1, a="A"),
                                _mk_completion(1, 0.6, 0, a="B"),
                                _mk_completion(2, 0.8, 1, a="C"),
                                _mk_completion(3, 0.7, 0, a="D")])
    pf = pools / f"pool_{slug}_seed42_N32.jsonl"
    try:
        pf.write_text(json.dumps(rec) + "\n")
        r = subprocess.run(
            [_sys.executable, "scripts/select_from_pool.py",
             "--config", f"{slug}_N4_w1.0_r0.7_s42",
             "--model-slug", slug, "--out-dir", str(tmp_path)],
            cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        out = list(tmp_path.glob("*.csv"))
        assert len(out) == 1 and out[0].read_text().count("\n") >= 2
    finally:
        import shutil
        shutil.rmtree(pools.parent, ignore_errors=True)
