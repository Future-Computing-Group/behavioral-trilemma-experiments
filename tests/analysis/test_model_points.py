"""Tests for analysis.model_points — per-model trilemma coordinates (H, C, A).

TDD RED step: written before the implementation exists. All fixtures use
tmp_path and synthetic CSVs; the real experiment_output/ is never touched.
"""

import csv
import math
import pathlib

import pytest

from analysis.model_points import (
    classify_row,
    model_coords,
    all_model_coords,
    seed_coords,
    calibration_points,
    all_seed_coords,
    all_calibration_points,
)

HEADER = [
    "task_id", "category", "N", "w_ratio", "w_C", "w_A", "r_min",
    "seed", "r_selected", "y", "V_selected", "brier", "gate_cleared",
]


def _write_csv(path: pathlib.Path, rows: list[dict]) -> str:
    """Write a probe CSV with the canonical header. rows are partial dicts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for i, r in enumerate(rows):
            full = {
                "task_id": r.get("task_id", f"t{i:03d}"),
                "category": r.get("category", "arithmetic_easy"),
                "N": r.get("N", 1),
                "w_ratio": r.get("w_ratio", 0),
                "w_C": r.get("w_C", 1.0),
                "w_A": r.get("w_A", 0.0),
                "r_min": r.get("r_min", 0.5),
                "seed": r.get("seed", 2000),
                "r_selected": r.get("r_selected", ""),
                "y": r.get("y", ""),
                "V_selected": r.get("V_selected", ""),
                "brier": r.get("brier", ""),
                "gate_cleared": r.get("gate_cleared", ""),
            }
            w.writerow(full)
    return str(path)


def _acted(r_selected, y, brier, gate_cleared=1):
    return {"r_selected": r_selected, "y": y, "brier": brier,
            "gate_cleared": gate_cleared, "V_selected": 0.0}


def _abstained(gate_cleared=""):
    return {"r_selected": "", "y": "", "brier": "", "gate_cleared": gate_cleared,
            "V_selected": ""}


def _pad(rows: list[dict], n: int) -> list[dict]:
    """Pad to >=100 rows with abstained rows so a seed is not 'partial'."""
    out = list(rows)
    while len(out) < n:
        out.append(_abstained())
    return out


# ---- (a) classify_row blank vs non-blank ---------------------------------

def test_classify_row_acted_non_blank():
    assert classify_row({"r_selected": "0.7"}) == "acted"
    assert classify_row({"r_selected": "0.0"}) == "acted"


def test_classify_row_abstained_blank_and_sentinels():
    assert classify_row({"r_selected": ""}) == "abstained"
    assert classify_row({"r_selected": "nan"}) == "abstained"
    assert classify_row({"r_selected": "None"}) == "abstained"
    assert classify_row({"r_selected": "  "}) == "abstained"


# ---- (b) hand-computable 6-row single-seed model -------------------------

def test_hand_computable_single_seed(tmp_path):
    # 4 acted (3 correct), 2 abstained.
    # brier on acted rows: 0.0, 0.04, 0.09, 1.0  -> mean = 1.13/4 = 0.2825
    rows = [
        _acted("1.0", "1", "0.0"),    # correct
        _acted("0.8", "1", "0.04"),   # correct
        _acted("0.7", "1", "0.09"),   # correct
        _acted("1.0", "0", "1.0"),    # acted, wrong
        _abstained(),
        _abstained(),
    ]
    p = _write_csv(tmp_path / "toy_7b_N1_w0_r0.5_s2000.csv", rows)
    d = model_coords([p], random_state=0)
    assert d["n_tasks"] == 6
    assert d["n_acted"] == 4
    assert d["n_seeds"] == 1
    # A = action rate = 4/6
    assert d["A"] == pytest.approx(4 / 6)
    # H = acted AND y==1 = 3/6
    assert d["H"] == pytest.approx(3 / 6)
    # C = 1 - mean(brier over acted) = 1 - 0.2825 = 0.7175
    assert d["C"] == pytest.approx(1 - (0.0 + 0.04 + 0.09 + 1.0) / 4)
    # single seed + <100 rows => partial
    assert d["partial"] is True


# ---- (c) A decoupled from gate_cleared -----------------------------------

def test_A_follows_acted_not_gate_cleared(tmp_path):
    # gate_cleared deliberately disagrees with acted:
    #   - abstained rows carry gate_cleared=1
    #   - acted rows carry gate_cleared=0
    rows = [
        _acted("0.9", "1", "0.01", gate_cleared=0),
        _acted("0.9", "0", "0.81", gate_cleared=0),
        {"r_selected": "", "y": "", "brier": "", "gate_cleared": 1,
         "V_selected": ""},
        {"r_selected": "", "y": "", "brier": "", "gate_cleared": 1,
         "V_selected": ""},
    ]
    p = _write_csv(tmp_path / "gtoy_7b_N1_w0_r0.5_s2000.csv", rows)
    d = model_coords([p], random_state=0)
    # acted-rate = 2/4 = 0.5 ; gate_cleared-rate would be 2/4 too but
    # composed of the *opposite* rows. Make the counts asymmetric to be sure:
    rows2 = [
        _acted("0.9", "1", "0.01", gate_cleared=0),
        _acted("0.9", "0", "0.81", gate_cleared=0),
        _acted("0.6", "1", "0.16", gate_cleared=0),
        {"r_selected": "", "y": "", "brier": "", "gate_cleared": 1,
         "V_selected": ""},
    ]
    p2 = _write_csv(tmp_path / "gtoy2_7b_N1_w0_r0.5_s2000.csv", rows2)
    d2 = model_coords([p2], random_state=0)
    # acted-rate = 3/4 ; gate_cleared mean would be 1/4 — assert acted wins.
    assert d2["A"] == pytest.approx(3 / 4)
    assert d2["A"] != pytest.approx(1 / 4)
    assert d["A"] == pytest.approx(2 / 4)


# ---- (d) partial models --------------------------------------------------

def test_partial_when_few_seed_files(tmp_path):
    runs = tmp_path / "_runs"
    base = [_acted("0.9", "1", "0.01")]
    for s in (2000, 2001, 2002):  # only 3 of 5
        _write_csv(runs / f"pm_7b_N1_w0_r0.5_s{s}.csv", _pad(base, 120))
    out = all_model_coords(str(runs), random_state=0)
    assert "pm_7b" in out
    assert out["pm_7b"]["partial"] is True
    assert out["pm_7b"]["n_seeds"] == 3


def test_partial_when_short_seed_file(tmp_path):
    runs = tmp_path / "_runs"
    base = [_acted("0.9", "1", "0.01")]
    for s in (2000, 2001, 2002, 2003):
        _write_csv(runs / f"sm_7b_N1_w0_r0.5_s{s}.csv", _pad(base, 120))
    # 5th seed has only 50 rows -> short -> partial even though 5 files
    _write_csv(runs / "sm_7b_N1_w0_r0.5_s2004.csv", _pad(base, 50))
    out = all_model_coords(str(runs), random_state=0)
    assert out["sm_7b"]["n_seeds"] == 5
    assert out["sm_7b"]["partial"] is True


def test_complete_model_not_partial(tmp_path):
    runs = tmp_path / "_runs"
    base = [_acted("0.9", "1", "0.01"), _acted("0.4", "0", "0.16")]
    for s in (2000, 2001, 2002, 2003, 2004):
        _write_csv(runs / f"cm_7b_N1_w0_r0.5_s{s}.csv", _pad(base, 120))
    out = all_model_coords(str(runs), random_state=0)
    assert out["cm_7b"]["n_seeds"] == 5
    assert out["cm_7b"]["partial"] is False


# ---- (e) zero-acted model -> C is nan, no exception ----------------------

def test_zero_acted_model_C_is_nan(tmp_path):
    rows = _pad([], 120)  # all abstained
    p = _write_csv(tmp_path / "za_7b_N1_w0_r0.5_s2000.csv", rows)
    d = model_coords([p], random_state=0)
    assert d["n_acted"] == 0
    assert math.isnan(d["C"])
    assert math.isnan(d["C_ci"][0]) and math.isnan(d["C_ci"][1])
    assert d["A"] == pytest.approx(0.0)
    assert d["H"] == pytest.approx(0.0)
    assert d["partial"] is True  # single seed


# ---- (f) bootstrap CI: deterministic, lo<=point<=hi ----------------------

def test_bootstrap_ci_deterministic_and_brackets_point(tmp_path):
    runs = tmp_path / "_runs"
    # mix of acted/abstained, correct/incorrect, >=100 rows, 5 seeds
    block = (
        [_acted("0.9", "1", "0.01")] * 30
        + [_acted("0.3", "0", "0.09")] * 30
        + [_acted("0.7", "0", "0.49")] * 20
        + [_abstained()] * 40
    )
    for s in (2000, 2001, 2002, 2003, 2004):
        _write_csv(runs / f"bm_7b_N1_w0_r0.5_s{s}.csv", list(block))
    d1 = all_model_coords(str(runs), random_state=0)["bm_7b"]
    d2 = all_model_coords(str(runs), random_state=0)["bm_7b"]
    # reproducible across two calls with the same random_state
    assert d1["H_ci"] == d2["H_ci"]
    assert d1["C_ci"] == d2["C_ci"]
    assert d1["A_ci"] == d2["A_ci"]
    for axis in ("H", "C", "A"):
        lo, hi = d1[f"{axis}_ci"]
        pt = d1[axis]
        assert lo <= pt <= hi, f"{axis}: {lo} <= {pt} <= {hi}"
        assert lo <= hi
    assert d1["partial"] is False


# ---- model-id grouping incl. underscore-heavy ids ------------------------

def test_model_id_grouping_handles_underscored_names(tmp_path):
    runs = tmp_path / "_runs"
    base = _pad([_acted("0.9", "1", "0.01")], 120)
    # mistral id contains underscores; must group on the _N1_ boundary
    for s in (2000, 2001, 2002, 2003, 2004):
        _write_csv(runs / f"mistral_7b-instruct-q4_K_M_N1_w0_r0.5_s{s}.csv", base)
    for s in (2000, 2001, 2002, 2003, 2004):
        _write_csv(runs / f"qwen2.5_7b_N1_w0_r0.5_s{s}.csv", base)
    out = all_model_coords(str(runs), random_state=0)
    assert set(out) == {"mistral_7b-instruct-q4_K_M", "qwen2.5_7b"}
    assert out["mistral_7b-instruct-q4_K_M"]["n_seeds"] == 5
    assert out["qwen2.5_7b"]["n_seeds"] == 5


def test_pooling_concatenates_seed_rows(tmp_path):
    runs = tmp_path / "_runs"
    # 5 seeds, each 2 acted + 0 abstained -> pooled 10 acted, A=1.0
    for s in (2000, 2001, 2002, 2003, 2004):
        rows = _pad([_acted("1.0", "1", "0.0"), _acted("1.0", "0", "1.0")], 120)
        _write_csv(runs / f"po_7b_N1_w0_r0.5_s{s}.csv", rows)
    out = all_model_coords(str(runs), random_state=0)["po_7b"]
    assert out["n_tasks"] == 600  # 5 * 120
    assert out["n_acted"] == 10   # 5 * 2


# ===== seed_coords / calibration_points (A.1b) ============================

# ---- (a) per-seed coords, distinct & NOT the pooled value ----------------

def test_seed_coords_per_seed_not_pooled(tmp_path):
    # Seed 2000: 2 acted, both correct, briers 0.0 & 0.04 -> mean 0.02
    #   A = 2/4 = 0.5 ; H = 2/4 = 0.5 ; C = 1 - 0.02 = 0.98
    s0 = _pad([
        _acted("1.0", "1", "0.0"),
        _acted("0.8", "1", "0.04"),
    ], 4)  # 2 acted + 2 abstained -> n_tasks 4
    p0 = _write_csv(tmp_path / "tm_7b_N1_w0_r0.5_s2000.csv", s0)
    # Seed 2001: 3 acted, 1 correct, briers 0.09, 1.0, 0.25 -> mean 1.34/3
    #   A = 3/3 = 1.0 ; H = 1/3 ; C = 1 - 1.34/3
    s1 = [
        _acted("0.7", "1", "0.09"),   # correct
        _acted("1.0", "0", "1.0"),    # wrong
        _acted("0.5", "0", "0.25"),   # wrong
    ]
    p1 = _write_csv(tmp_path / "tm_7b_N1_w0_r0.5_s2001.csv", s1)

    out = seed_coords([p1, p0])  # deliberately unsorted input
    assert [e["seed"] for e in out] == [2000, 2001]  # sorted by seed

    e0, e1 = out
    assert e0["seed"] == 2000
    assert e0["n_tasks"] == 4
    assert e0["n_acted"] == 2
    assert e0["A"] == pytest.approx(2 / 4)
    assert e0["H"] == pytest.approx(2 / 4)
    assert e0["C"] == pytest.approx(1 - (0.0 + 0.04) / 2)

    assert e1["seed"] == 2001
    assert e1["n_tasks"] == 3
    assert e1["n_acted"] == 3
    assert e1["A"] == pytest.approx(3 / 3)
    assert e1["H"] == pytest.approx(1 / 3)
    assert e1["C"] == pytest.approx(1 - (0.09 + 1.0 + 0.25) / 3)

    # Per-seed values must NOT equal the pooled (model_coords) value.
    pooled = model_coords([p0, p1])
    assert e0["C"] != pytest.approx(pooled["C"])
    assert e1["C"] != pytest.approx(pooled["C"])
    assert e0["H"] != pytest.approx(pooled["H"]) or \
        e1["H"] != pytest.approx(pooled["H"])


# ---- (b) seed int parsed from _s2003.csv suffix --------------------------

def test_seed_coords_parses_seed_2003(tmp_path):
    p = _write_csv(
        tmp_path / "tm_7b-instruct_N1_w0_r0.5_s2003.csv",
        [_acted("0.9", "1", "0.01")],
    )
    out = seed_coords([p])
    assert len(out) == 1
    assert out[0]["seed"] == 2003
    assert isinstance(out[0]["seed"], int)


# ---- (c) a zero-acted seed -> C is nan, no exception, others unaffected ---

def test_seed_coords_zero_acted_seed_C_nan(tmp_path):
    good = _write_csv(
        tmp_path / "zm_7b_N1_w0_r0.5_s2000.csv",
        [_acted("0.9", "1", "0.01"), _acted("0.4", "0", "0.16")],
    )
    empty = _write_csv(
        tmp_path / "zm_7b_N1_w0_r0.5_s2001.csv",
        [_abstained(), _abstained(), _abstained()],
    )
    out = seed_coords([good, empty])
    e_good, e_empty = out
    assert e_empty["seed"] == 2001
    assert e_empty["n_acted"] == 0
    assert math.isnan(e_empty["C"])
    assert e_empty["A"] == pytest.approx(0.0)
    assert e_empty["H"] == pytest.approx(0.0)
    # other seed is unaffected and has a real C
    assert e_good["seed"] == 2000
    assert e_good["n_acted"] == 2
    assert not math.isnan(e_good["C"])
    assert e_good["C"] == pytest.approx(1 - (0.01 + 0.16) / 2)


# ---- (d) calibration_points: one (r,y) per acted row, excludes abstained -

def test_calibration_points_one_per_acted_excludes_abstained(tmp_path):
    rows_a = [
        _acted("0.9", "1", "0.01"),
        _abstained(),
        _acted("0.3", "0", "0.09"),
    ]
    rows_b = [
        _abstained(),
        _acted("0.75", "1", "0.0625"),
    ]
    pa = _write_csv(tmp_path / "cm_7b_N1_w0_r0.5_s2000.csv", rows_a)
    pb = _write_csv(tmp_path / "cm_7b_N1_w0_r0.5_s2001.csv", rows_b)

    pts = calibration_points([pa, pb])
    # 3 acted total (2 in a, 1 in b); abstained excluded
    assert len(pts) == 3
    pooled = model_coords([pa, pb])
    assert len(pts) == pooled["n_acted"]

    # deterministic order: file order, then row order
    assert pts == [
        {"r": 0.9, "y": 1},
        {"r": 0.3, "y": 0},
        {"r": 0.75, "y": 1},
    ]
    for pt in pts:
        assert isinstance(pt["r"], float)
        assert isinstance(pt["y"], int)


# ---- (e) directory API groups underscore-heavy model ids -----------------

def test_all_seed_and_calibration_group_underscored_ids(tmp_path):
    runs = tmp_path / "_runs"
    for s in (2000, 2001):
        _write_csv(
            runs / f"mistral_7b-instruct-q4_K_M_N1_w0_r0.5_s{s}.csv",
            [_acted("0.9", "1", "0.01"), _abstained()],
        )
    for s in (2000, 2001):
        _write_csv(
            runs / f"qwen2.5_7b_N1_w0_r0.5_s{s}.csv",
            [_acted("0.6", "0", "0.36")],
        )

    sc = all_seed_coords(str(runs))
    assert set(sc) == {"mistral_7b-instruct-q4_K_M", "qwen2.5_7b"}
    mistral = sc["mistral_7b-instruct-q4_K_M"]
    assert [e["seed"] for e in mistral] == [2000, 2001]
    assert mistral[0]["n_tasks"] == 2
    assert mistral[0]["n_acted"] == 1

    cp = all_calibration_points(str(runs))
    assert set(cp) == {"mistral_7b-instruct-q4_K_M", "qwen2.5_7b"}
    # mistral: 1 acted per seed * 2 seeds = 2 points
    assert cp["mistral_7b-instruct-q4_K_M"] == [
        {"r": 0.9, "y": 1},
        {"r": 0.9, "y": 1},
    ]
    assert cp["qwen2.5_7b"] == [
        {"r": 0.6, "y": 0},
        {"r": 0.6, "y": 0},
    ]
