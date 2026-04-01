"""Tests for task loader and ground truth verification."""
import json
import pathlib
import pytest

from src.tasks import load_tasks, verify_arithmetic, verify_code, get_tasks_by_category


@pytest.fixture
def sample_tasks(tmp_path):
    """Minimal task set for testing."""
    tasks = [
        {
            "id": "arith_easy_01",
            "category": "arithmetic_easy",
            "prompt": "What is 17 * 23?",
            "ground_truth": "391",
            "verification": "arithmetic",
        },
        {
            "id": "arith_hard_01",
            "category": "arithmetic_hard",
            "prompt": "What is 123 + 456 * 7 - 89?",
            "ground_truth": "3226",
            "verification": "arithmetic",
        },
        {
            "id": "fact_common_01",
            "category": "factual_common",
            "prompt": "What is the chemical symbol for gold?",
            "ground_truth": "Au",
            "verification": "exact",
        },
        {
            "id": "code_simple_01",
            "category": "code_simple",
            "prompt": "Write a Python function that returns the factorial of n.",
            "ground_truth": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "verification": "code",
            "test_cases": [
                {"input": "factorial(0)", "expected": "1"},
                {"input": "factorial(5)", "expected": "120"},
                {"input": "factorial(10)", "expected": "3628800"},
            ],
        },
    ]
    path = tmp_path / "task_set.json"
    path.write_text(json.dumps(tasks, indent=2))
    return path


def test_load_tasks_returns_list(sample_tasks):
    tasks = load_tasks(sample_tasks)
    assert isinstance(tasks, list)
    assert len(tasks) == 4


def test_task_required_fields(sample_tasks):
    tasks = load_tasks(sample_tasks)
    for t in tasks:
        assert "id" in t
        assert "category" in t
        assert "prompt" in t
        assert "ground_truth" in t
        assert "verification" in t


def test_get_tasks_by_category(sample_tasks):
    tasks = load_tasks(sample_tasks)
    arith = get_tasks_by_category(tasks, "arithmetic_easy")
    assert len(arith) == 1
    assert arith[0]["id"] == "arith_easy_01"


def test_verify_arithmetic_correct():
    assert verify_arithmetic("17 * 23", "391") is True
    assert verify_arithmetic("123 + 456 * 7 - 89", "3226") is True


def test_verify_arithmetic_incorrect():
    assert verify_arithmetic("17 * 23", "392") is False


def test_verify_code_correct():
    code = "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
    test_cases = [
        {"input": "factorial(0)", "expected": "1"},
        {"input": "factorial(5)", "expected": "120"},
    ]
    assert verify_code(code, test_cases) is True


def test_verify_code_incorrect():
    code = "def factorial(n):\n    return n"  # wrong
    test_cases = [
        {"input": "factorial(5)", "expected": "120"},
    ]
    assert verify_code(code, test_cases) is False


def test_load_full_task_set():
    """Test the real task set exists and has 100 tasks."""
    path = pathlib.Path(__file__).resolve().parent.parent / "tasks" / "task_set.json"
    if not path.exists():
        pytest.skip("Full task set not generated yet")
    tasks = load_tasks(path)
    assert len(tasks) == 100

    # Category counts
    cats = {}
    for t in tasks:
        cats[t["category"]] = cats.get(t["category"], 0) + 1
    assert cats.get("arithmetic_easy", 0) == 20
    assert cats.get("arithmetic_hard", 0) == 20
    assert cats.get("factual_common", 0) == 15
    assert cats.get("factual_obscure", 0) == 15
    assert cats.get("code_simple", 0) == 15
    assert cats.get("code_algorithmic", 0) == 15


def test_all_arithmetic_ground_truths_correct():
    """Verify every arithmetic task's ground truth is computationally correct."""
    path = pathlib.Path(__file__).resolve().parent.parent / "tasks" / "task_set.json"
    if not path.exists():
        pytest.skip("Full task set not generated yet")
    tasks = load_tasks(path)
    for t in tasks:
        if t["verification"] == "arithmetic":
            expr = t.get("expression")
            assert expr is not None, f"Task {t['id']} missing 'expression' field"
            assert verify_arithmetic(expr, t["ground_truth"]), (
                f"Task {t['id']}: {expr} != {t['ground_truth']}"
            )


def test_all_code_ground_truths_correct():
    """Verify every code task's ground truth passes its test cases."""
    path = pathlib.Path(__file__).resolve().parent.parent / "tasks" / "task_set.json"
    if not path.exists():
        pytest.skip("Full task set not generated yet")
    tasks = load_tasks(path)
    for t in tasks:
        if t["verification"] == "code":
            assert verify_code(t["ground_truth"], t["test_cases"]), (
                f"Task {t['id']}: ground truth fails test cases"
            )
