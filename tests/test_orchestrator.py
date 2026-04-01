"""Tests for experiment orchestrator."""
import csv
import json
import pathlib
import pytest
from unittest.mock import patch, MagicMock

from src.config import load_config, generate_configs
from src.orchestrator import (
    config_filename,
    run_single_config,
    get_completed_configs,
    run_phase0_calibration,
)


@pytest.fixture
def mini_tasks(tmp_path):
    """3 minimal tasks for testing."""
    tasks = [
        {"id": "t1", "category": "arithmetic_easy", "prompt": "What is 2+2?",
         "ground_truth": "4", "verification": "arithmetic", "expression": "2+2"},
        {"id": "t2", "category": "factual_common", "prompt": "Capital of France?",
         "ground_truth": "Paris", "verification": "exact"},
        {"id": "t3", "category": "code_simple", "prompt": "Write is_even",
         "ground_truth": "def is_even(n): return n % 2 == 0",
         "verification": "code",
         "test_cases": [{"input": "is_even(4)", "expected": "True"}]},
    ]
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(tasks))
    return path, tasks


def test_config_filename():
    cfg = {"N": 8, "w_ratio": 1.0, "r_min": 0.7, "seed": 42}
    name = config_filename(cfg, model="qwen2.5:7b")
    assert name == "qwen2.5_7b_N8_w1.0_r0.7_s42.csv"


def test_get_completed_configs(tmp_path):
    # Create a fake result file
    (tmp_path / "qwen2.5_7b_N8_w1.0_r0.7_s42.csv").write_text("header\n")
    completed = get_completed_configs(tmp_path)
    assert "qwen2.5_7b_N8_w1.0_r0.7_s42.csv" in completed


def test_run_single_config_produces_csv(mini_tasks, tmp_path):
    task_path, tasks = mini_tasks
    cfg = {"N": 2, "w_ratio": 1.0, "w_C": 1.0, "w_A": 1.0, "r_min": 0.7, "seed": 42}

    def fake_generate(prompt, n, model, temperature, seed, **kw):
        return [f"CONFIDENCE: 0.8\nANSWER: {tasks[0]['ground_truth']}"] * n

    with patch("src.orchestrator.generate_completions", side_effect=fake_generate):
        outfile = run_single_config(cfg, tasks, tmp_path, model="qwen2.5:7b")

    assert outfile.exists()
    with open(outfile) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 3  # 3 tasks
    assert "task_id" in rows[0]
    assert "r_selected" in rows[0]
    assert "brier" in rows[0]
    assert "y" in rows[0]


def test_run_single_config_treatment_verification(mini_tasks, tmp_path):
    """L38: verify treatment was applied when w_A > 0."""
    task_path, tasks = mini_tasks
    cfg = {"N": 4, "w_ratio": 2.0, "w_C": 1.0, "w_A": 2.0, "r_min": 0.5, "seed": 42}

    def fake_generate(prompt, n, model, temperature, seed, **kw):
        # Return completions with varying confidence
        return [f"CONFIDENCE: {0.3 + 0.2*i}\nANSWER: 4" for i in range(n)]

    with patch("src.orchestrator.generate_completions", side_effect=fake_generate):
        outfile = run_single_config(cfg, tasks, tmp_path, model="qwen2.5:7b")

    with open(outfile) as f:
        rows = list(csv.DictReader(f))
    # At least one selected completion should have r >= r_min when w_A > 0
    r_values = [float(row["r_selected"]) for row in rows]
    assert any(r >= 0.5 for r in r_values), "Treatment not applied: no r >= r_min"


def test_phase0_calibration(mini_tasks, tmp_path):
    """Phase 0: held-out calibration produces binding set classification."""
    task_path, tasks = mini_tasks

    call_count = [0]
    def fake_generate(prompt, n, model, temperature, seed, **kw):
        call_count[0] += 1
        # Alternate correct/incorrect to simulate p ≈ 0.5
        if call_count[0] % 2 == 0:
            return ["CONFIDENCE: 0.6\nANSWER: 4"]
        return ["CONFIDENCE: 0.6\nANSWER: wrong"]

    cal_seeds = list(range(1000, 1005))  # 5 seeds for test
    with patch("src.orchestrator.generate_completions", side_effect=fake_generate):
        outfile = run_phase0_calibration(
            tasks, tmp_path, model="qwen2.5:7b",
            seeds=cal_seeds, thresholds=[0.5, 0.7, 0.9],
        )

    assert outfile.exists()
    with open(outfile) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3  # 3 tasks
    assert "task_id" in rows[0]
    assert "p_hat" in rows[0]
    assert "binding_0.5" in rows[0]
    assert "binding_0.7" in rows[0]
    assert "binding_0.9" in rows[0]
