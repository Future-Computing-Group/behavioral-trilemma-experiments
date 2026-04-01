"""Task loading and ground truth verification."""
import json
import pathlib


def load_tasks(path: pathlib.Path) -> list[dict]:
    """Load task set from JSON file."""
    with open(path) as f:
        tasks = json.load(f)
    return tasks


def get_tasks_by_category(tasks: list[dict], category: str) -> list[dict]:
    """Filter tasks by category."""
    return [t for t in tasks if t["category"] == category]


def verify_arithmetic(expression: str, expected: str) -> bool:
    """Verify an arithmetic ground truth by evaluating the expression.

    Only allows safe arithmetic operations (no builtins, no imports).
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result) == expected.strip()
    except Exception:
        return False


def verify_code(code: str, test_cases: list[dict]) -> bool:
    """Verify a code ground truth by executing test cases.

    Runs the code in a shared namespace so recursive/self-referencing
    functions work, then evaluates each test case.
    """
    namespace = {"__builtins__": __builtins__}
    try:
        exec(code, namespace)
    except Exception:
        return False

    for tc in test_cases:
        try:
            result = eval(tc["input"], namespace)
            if str(result) != tc["expected"]:
                return False
        except Exception:
            return False
    return True
