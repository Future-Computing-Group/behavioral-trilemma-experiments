"""Tests for analysis metrics."""
import csv
import pathlib
import pytest
import pandas as pd

from analysis.metrics import (
    load_results,
    compute_per_config_metrics,
    compute_brier_decomposition,
)
from analysis.hypothesis_tests import run_all_tests


@pytest.fixture
def sample_results(tmp_path):
    """Create sample result CSVs for two configs."""
    fieldnames = [
        "task_id", "category", "N", "w_ratio", "w_C", "w_A", "r_min",
        "seed", "r_selected", "y", "V_selected", "brier", "gate_cleared",
    ]

    # Config 1: N=1, w_A=0 (baseline)
    rows1 = [
        {"task_id": "t1", "category": "arith", "N": 1, "w_ratio": 0, "w_C": 1, "w_A": 0,
         "r_min": 0.7, "seed": 42, "r_selected": 0.6, "y": 1, "V_selected": -0.16,
         "brier": 0.16, "gate_cleared": 0},
        {"task_id": "t2", "category": "arith", "N": 1, "w_ratio": 0, "w_C": 1, "w_A": 0,
         "r_min": 0.7, "seed": 42, "r_selected": 0.8, "y": 0, "V_selected": -0.64,
         "brier": 0.64, "gate_cleared": 1},
    ]
    f1 = tmp_path / "test_N1_w0_r0.7_s42.csv"
    with open(f1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows1)

    # Config 2: N=8, w_A=1 (with gating)
    rows2 = [
        {"task_id": "t1", "category": "arith", "N": 8, "w_ratio": 1.0, "w_C": 1, "w_A": 1,
         "r_min": 0.7, "seed": 42, "r_selected": 0.9, "y": 1, "V_selected": 0.99,
         "brier": 0.01, "gate_cleared": 1},
        {"task_id": "t2", "category": "arith", "N": 8, "w_ratio": 1.0, "w_C": 1, "w_A": 1,
         "r_min": 0.7, "seed": 42, "r_selected": 0.85, "y": 0, "V_selected": 0.2775,
         "brier": 0.7225, "gate_cleared": 1},
    ]
    f2 = tmp_path / "test_N8_w1.0_r0.7_s42.csv"
    with open(f2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows2)

    return tmp_path


def test_load_results(sample_results):
    df = load_results(sample_results)
    assert len(df) == 4  # 2 files × 2 rows


def test_compute_per_config_metrics(sample_results):
    df = load_results(sample_results)
    metrics = compute_per_config_metrics(df)
    # Should have 2 groups (one per config)
    assert len(metrics) == 2
    # Check fields
    row = metrics.iloc[0]
    assert "mean_brier" in metrics.columns
    assert "mean_H" in metrics.columns
    assert "mean_A" in metrics.columns


def test_brier_decomposition():
    # Simple case: 4 predictions, 2 bins
    r_values = [0.2, 0.3, 0.7, 0.8]
    y_values = [0, 0, 1, 1]
    decomp = compute_brier_decomposition(r_values, y_values, n_bins=2)
    assert "reliability" in decomp
    assert "resolution" in decomp
    assert "uncertainty" in decomp
    # BS = reliability - resolution + uncertainty
    bs = sum((r - y) ** 2 for r, y in zip(r_values, y_values)) / 4
    # Binned decomposition is approximate; tolerance reflects bin-boundary effects
    assert abs(decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"] - bs) < 0.01


def _first_existing(paths: list[pathlib.Path]) -> pathlib.Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("No expected results directory found")


def _load_phase0_simple(path: pathlib.Path) -> tuple[dict[str, float], dict[float, set[str]]]:
    df = pd.read_csv(path)
    p_hat = dict(zip(df["task_id"], df["p_hat"].astype(float)))
    binding_sets: dict[float, set[str]] = {}
    for col in df.columns:
        if col.startswith("binding_"):
            threshold = float(col.replace("binding_", ""))
            binding_sets[threshold] = set(df.loc[df[col] == 1, "task_id"])
    return p_hat, binding_sets


def test_updated_hypotheses_on_logprob_results():
    root = pathlib.Path(__file__).resolve().parent.parent
    results_dir = _first_existing(
        [
            root / "experiment_output" / "raw_runs" / "logprob" / "results",
            root / "experiment_results" / "raw_runs" / "logprob" / "results",
            root / "puhti_output" / "results",
        ]
    )

    phase0_path = results_dir / "phase0_calibration.csv"
    if not phase0_path.exists():
        pytest.skip("phase0_calibration.csv not found; cannot evaluate updated hypotheses.")

    df = load_results(results_dir)
    p_hat, binding_sets = _load_phase0_simple(phase0_path)

    results = run_all_tests(
        df,
        binding_tasks=binding_sets,
        p_hat=p_hat,
        n_boot=1,
    )


    # H1 (updated): fixed-axis degradation at N=32 under higher autonomy weight.
    assert results["H1"]["statistic"] > 0
    assert results["H1"]["p_value"] < 0.05
    assert results["H1"]["mean_diff"] > 0

    # H2 (updated): monotone inflation trend across w_A/w_C.
    assert results["H2"]["z"] > 0
    assert results["H2"]["p_value"] < 0.05
    assert results["H2"]["rho"] > 0

    # H3 (updated): tolerance-aware convexity criterion.
    assert results["H3"]["criterion_met"] is True
    assert results["H3"]["violation_rate"] < results["H3"]["tolerance"]

    # H4-H6 unchanged and should remain strongly supported.
    assert results["H4"]["p_value"] < 0.05
    assert results["H5"]["p_value"] < 0.05
    assert results["H5"]["ratio"] > 2.0
    assert results["H6"]["statistic"] < 0
    assert results["H6"]["p_value"] < 0.05
