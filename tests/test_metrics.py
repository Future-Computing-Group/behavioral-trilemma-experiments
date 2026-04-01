"""Tests for analysis metrics."""
import csv
import pathlib
import pytest

from analysis.metrics import (
    load_results,
    compute_per_config_metrics,
    compute_brier_decomposition,
)


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
