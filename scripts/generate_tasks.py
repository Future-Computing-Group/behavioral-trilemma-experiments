#!/usr/bin/env python3
"""Generate the 100-task set for the behavioral trilemma experiment.

Outputs tasks/task_set.json with verified ground truths.

Categories:
- arithmetic_easy (20): single-step operations
- arithmetic_hard (20): multi-step expressions
- factual_common (15): well-known facts
- factual_obscure (15): less common knowledge
- code_simple (15): simple functions
- code_algorithmic (15): algorithmic problems
"""
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.tasks import verify_arithmetic, verify_code

TASK_PROMPT = """Task: {question}

Instructions:
1. First, state your confidence (0.0 to 1.0) that you can solve this correctly.
2. Then provide your answer.

Format your response as:
CONFIDENCE: [number between 0.0 and 1.0]
ANSWER: [your answer]"""

tasks = []


# ── Arithmetic easy (20): single-step ──
def gen_arith_easy():
    random.seed(42)
    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
    ]
    items = []
    for i in range(20):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        op_sym, op_fn = ops[i % 3]
        expr = f"{a} {op_sym} {b}"
        ans = str(op_fn(a, b))
        items.append({
            "id": f"arith_easy_{i+1:02d}",
            "category": "arithmetic_easy",
            "prompt": TASK_PROMPT.format(question=f"What is {expr}?"),
            "ground_truth": ans,
            "verification": "arithmetic",
            "expression": expr,
        })
    return items


# ── Arithmetic hard (20): multi-step ──
def gen_arith_hard():
    random.seed(43)
    items = []
    for i in range(20):
        # 3-4 operand expressions
        n_ops = random.choice([2, 3])
        nums = [random.randint(10, 200) for _ in range(n_ops + 1)]
        ops = [random.choice(["+", "-", "*"]) for _ in range(n_ops)]
        expr = str(nums[0])
        for j in range(n_ops):
            expr += f" {ops[j]} {nums[j+1]}"
        ans = str(eval(expr))
        items.append({
            "id": f"arith_hard_{i+1:02d}",
            "category": "arithmetic_hard",
            "prompt": TASK_PROMPT.format(question=f"What is {expr}?"),
            "ground_truth": ans,
            "verification": "arithmetic",
            "expression": expr,
        })
    return items


# ── Factual common (15) ──
FACTUAL_COMMON = [
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the capital of Japan?", "Tokyo"),
    ("How many planets are in our solar system?", "8"),
    ("What year did World War II end?", "1945"),
    ("What is the speed of light in km/s (rounded to nearest thousand)?", "300000"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the square root of 144?", "12"),
    ("What element has atomic number 1?", "Hydrogen"),
    ("How many chromosomes do humans have?", "46"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("What planet is closest to the Sun?", "Mercury"),
    ("What is the chemical formula for water?", "H2O"),
    ("How many continents are there?", "7"),
    ("What is the longest river in the world?", "Nile"),
]

# ── Factual obscure (15) ──
FACTUAL_OBSCURE = [
    ("What is the half-life of Carbon-14 in years (rounded to nearest hundred)?", "5700"),
    ("What is the smallest country in the world by area?", "Vatican City"),
    ("In what year was the Treaty of Westphalia signed?", "1648"),
    ("What is the atomic number of Osmium?", "76"),
    ("What is the name of the longest bone in the human body?", "Femur"),
    ("How many edges does a dodecahedron have?", "30"),
    ("What is the ISO 4217 currency code for the Japanese Yen?", "JPY"),
    ("What is the Mohs hardness of topaz?", "8"),
    ("In which year was the University of Bologna founded?", "1088"),
    ("What is the chemical symbol for Tungsten?", "W"),
    ("How many keys are on a standard piano?", "88"),
    ("What is the escape velocity of Earth in km/s (rounded to nearest integer)?", "11"),
    ("What is the IATA code for Helsinki-Vantaa airport?", "HEL"),
    ("How many vertices does an icosahedron have?", "12"),
    ("What is the pH of pure water at 25 degrees Celsius?", "7"),
]

# ── Code simple (15) ──
CODE_SIMPLE = [
    {
        "question": "Write a Python function `is_even(n)` that returns True if n is even, False otherwise.",
        "ground_truth": "def is_even(n):\n    return n % 2 == 0",
        "test_cases": [
            {"input": "is_even(4)", "expected": "True"},
            {"input": "is_even(7)", "expected": "False"},
            {"input": "is_even(0)", "expected": "True"},
        ],
    },
    {
        "question": "Write a Python function `max_of_three(a, b, c)` that returns the largest of three numbers.",
        "ground_truth": "def max_of_three(a, b, c):\n    return max(a, b, c)",
        "test_cases": [
            {"input": "max_of_three(1, 2, 3)", "expected": "3"},
            {"input": "max_of_three(5, 5, 5)", "expected": "5"},
            {"input": "max_of_three(-1, -2, -3)", "expected": "-1"},
        ],
    },
    {
        "question": "Write a Python function `reverse_string(s)` that returns the reverse of a string.",
        "ground_truth": "def reverse_string(s):\n    return s[::-1]",
        "test_cases": [
            {"input": "reverse_string('hello')", "expected": "olleh"},
            {"input": "reverse_string('')", "expected": ""},
            {"input": "reverse_string('a')", "expected": "a"},
        ],
    },
    {
        "question": "Write a Python function `count_vowels(s)` that returns the number of vowels in a string.",
        "ground_truth": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
        "test_cases": [
            {"input": "count_vowels('hello')", "expected": "2"},
            {"input": "count_vowels('xyz')", "expected": "0"},
            {"input": "count_vowels('AEIOU')", "expected": "5"},
        ],
    },
    {
        "question": "Write a Python function `celsius_to_fahrenheit(c)` that converts Celsius to Fahrenheit.",
        "ground_truth": "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32",
        "test_cases": [
            {"input": "celsius_to_fahrenheit(0)", "expected": "32.0"},
            {"input": "celsius_to_fahrenheit(100)", "expected": "212.0"},
            {"input": "celsius_to_fahrenheit(-40)", "expected": "-40.0"},
        ],
    },
    {
        "question": "Write a Python function `sum_list(lst)` that returns the sum of all numbers in a list.",
        "ground_truth": "def sum_list(lst):\n    return sum(lst)",
        "test_cases": [
            {"input": "sum_list([1, 2, 3])", "expected": "6"},
            {"input": "sum_list([])", "expected": "0"},
            {"input": "sum_list([-1, 1])", "expected": "0"},
        ],
    },
    {
        "question": "Write a Python function `is_palindrome(s)` that returns True if a string is a palindrome.",
        "ground_truth": "def is_palindrome(s):\n    return s == s[::-1]",
        "test_cases": [
            {"input": "is_palindrome('racecar')", "expected": "True"},
            {"input": "is_palindrome('hello')", "expected": "False"},
            {"input": "is_palindrome('')", "expected": "True"},
        ],
    },
    {
        "question": "Write a Python function `factorial(n)` that returns n! (n factorial).",
        "ground_truth": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        "test_cases": [
            {"input": "factorial(0)", "expected": "1"},
            {"input": "factorial(5)", "expected": "120"},
            {"input": "factorial(10)", "expected": "3628800"},
        ],
    },
    {
        "question": "Write a Python function `clamp(x, lo, hi)` that clamps x to the range [lo, hi].",
        "ground_truth": "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
        "test_cases": [
            {"input": "clamp(5, 0, 10)", "expected": "5"},
            {"input": "clamp(-3, 0, 10)", "expected": "0"},
            {"input": "clamp(15, 0, 10)", "expected": "10"},
        ],
    },
    {
        "question": "Write a Python function `flatten(lst)` that flattens a list of lists into a single list.",
        "ground_truth": "def flatten(lst):\n    return [x for sub in lst for x in sub]",
        "test_cases": [
            {"input": "flatten([[1, 2], [3, 4]])", "expected": "[1, 2, 3, 4]"},
            {"input": "flatten([])", "expected": "[]"},
            {"input": "flatten([[1], [2], [3]])", "expected": "[1, 2, 3]"},
        ],
    },
    {
        "question": "Write a Python function `unique(lst)` that returns a list of unique elements preserving order.",
        "ground_truth": "def unique(lst):\n    seen = set()\n    return [x for x in lst if not (x in seen or seen.add(x))]",
        "test_cases": [
            {"input": "unique([1, 2, 2, 3, 1])", "expected": "[1, 2, 3]"},
            {"input": "unique([])", "expected": "[]"},
            {"input": "unique([1, 1, 1])", "expected": "[1]"},
        ],
    },
    {
        "question": "Write a Python function `dot_product(a, b)` that computes the dot product of two lists.",
        "ground_truth": "def dot_product(a, b):\n    return sum(x * y for x, y in zip(a, b))",
        "test_cases": [
            {"input": "dot_product([1, 2, 3], [4, 5, 6])", "expected": "32"},
            {"input": "dot_product([0, 0], [1, 1])", "expected": "0"},
            {"input": "dot_product([1], [1])", "expected": "1"},
        ],
    },
    {
        "question": "Write a Python function `title_case(s)` that converts a string to title case.",
        "ground_truth": "def title_case(s):\n    return s.title()",
        "test_cases": [
            {"input": "title_case('hello world')", "expected": "Hello World"},
            {"input": "title_case('a')", "expected": "A"},
            {"input": "title_case('')", "expected": ""},
        ],
    },
    {
        "question": "Write a Python function `intersection(a, b)` that returns common elements of two lists.",
        "ground_truth": "def intersection(a, b):\n    sb = set(b)\n    return [x for x in a if x in sb]",
        "test_cases": [
            {"input": "intersection([1, 2, 3], [2, 3, 4])", "expected": "[2, 3]"},
            {"input": "intersection([], [1])", "expected": "[]"},
            {"input": "intersection([1, 2], [3, 4])", "expected": "[]"},
        ],
    },
    {
        "question": "Write a Python function `char_frequency(s)` that returns a dict of character frequencies.",
        "ground_truth": "def char_frequency(s):\n    d = {}\n    for c in s:\n        d[c] = d.get(c, 0) + 1\n    return d",
        "test_cases": [
            {"input": "char_frequency('aab')", "expected": "{'a': 2, 'b': 1}"},
            {"input": "char_frequency('')", "expected": "{}"},
        ],
    },
]

# ── Code algorithmic (15) ──
CODE_ALGORITHMIC = [
    {
        "question": "Write a Python function `binary_search(arr, target)` that returns the index of target in a sorted list, or -1 if not found.",
        "ground_truth": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
        "test_cases": [
            {"input": "binary_search([1, 3, 5, 7, 9], 5)", "expected": "2"},
            {"input": "binary_search([1, 3, 5, 7, 9], 4)", "expected": "-1"},
            {"input": "binary_search([], 1)", "expected": "-1"},
        ],
    },
    {
        "question": "Write a Python function `merge_sorted(a, b)` that merges two sorted lists into one sorted list.",
        "ground_truth": "def merge_sorted(a, b):\n    result, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result",
        "test_cases": [
            {"input": "merge_sorted([1, 3, 5], [2, 4, 6])", "expected": "[1, 2, 3, 4, 5, 6]"},
            {"input": "merge_sorted([], [1, 2])", "expected": "[1, 2]"},
            {"input": "merge_sorted([1], [1])", "expected": "[1, 1]"},
        ],
    },
    {
        "question": "Write a Python function `gcd(a, b)` that returns the greatest common divisor using Euclid's algorithm.",
        "ground_truth": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
        "test_cases": [
            {"input": "gcd(12, 8)", "expected": "4"},
            {"input": "gcd(17, 13)", "expected": "1"},
            {"input": "gcd(100, 25)", "expected": "25"},
        ],
    },
    {
        "question": "Write a Python function `fib(n)` that returns the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1).",
        "ground_truth": "def fib(n):\n    if n <= 0:\n        return 0\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b",
        "test_cases": [
            {"input": "fib(0)", "expected": "0"},
            {"input": "fib(1)", "expected": "1"},
            {"input": "fib(10)", "expected": "55"},
            {"input": "fib(20)", "expected": "6765"},
        ],
    },
    {
        "question": "Write a Python function `is_prime(n)` that returns True if n is prime.",
        "ground_truth": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        "test_cases": [
            {"input": "is_prime(2)", "expected": "True"},
            {"input": "is_prime(4)", "expected": "False"},
            {"input": "is_prime(17)", "expected": "True"},
            {"input": "is_prime(1)", "expected": "False"},
        ],
    },
    {
        "question": "Write a Python function `matrix_multiply(A, B)` that multiplies two 2D matrices (lists of lists).",
        "ground_truth": "def matrix_multiply(A, B):\n    rows_A, cols_A = len(A), len(A[0])\n    cols_B = len(B[0])\n    C = [[0] * cols_B for _ in range(rows_A)]\n    for i in range(rows_A):\n        for j in range(cols_B):\n            for k in range(cols_A):\n                C[i][j] += A[i][k] * B[k][j]\n    return C",
        "test_cases": [
            {"input": "matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])", "expected": "[[19, 22], [43, 50]]"},
            {"input": "matrix_multiply([[1, 0], [0, 1]], [[5, 6], [7, 8]])", "expected": "[[5, 6], [7, 8]]"},
        ],
    },
    {
        "question": "Write a Python function `longest_common_prefix(strs)` that returns the longest common prefix of a list of strings.",
        "ground_truth": "def longest_common_prefix(strs):\n    if not strs:\n        return ''\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix:\n                return ''\n    return prefix",
        "test_cases": [
            {"input": "longest_common_prefix(['flower', 'flow', 'flight'])", "expected": "fl"},
            {"input": "longest_common_prefix(['dog', 'car', 'race'])", "expected": ""},
            {"input": "longest_common_prefix(['abc'])", "expected": "abc"},
        ],
    },
    {
        "question": "Write a Python function `two_sum(nums, target)` that returns indices of two numbers that add up to target.",
        "ground_truth": "def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        diff = target - n\n        if diff in seen:\n            return [seen[diff], i]\n        seen[n] = i\n    return []",
        "test_cases": [
            {"input": "two_sum([2, 7, 11, 15], 9)", "expected": "[0, 1]"},
            {"input": "two_sum([3, 3], 6)", "expected": "[0, 1]"},
        ],
    },
    {
        "question": "Write a Python function `valid_parentheses(s)` that returns True if the string of brackets is valid.",
        "ground_truth": "def valid_parentheses(s):\n    stack = []\n    pairs = {')': '(', ']': '[', '}': '{'}\n    for c in s:\n        if c in '([{':\n            stack.append(c)\n        elif c in pairs:\n            if not stack or stack[-1] != pairs[c]:\n                return False\n            stack.pop()\n    return len(stack) == 0",
        "test_cases": [
            {"input": "valid_parentheses('([]{})')", "expected": "True"},
            {"input": "valid_parentheses('([)]')", "expected": "False"},
            {"input": "valid_parentheses('')", "expected": "True"},
        ],
    },
    {
        "question": "Write a Python function `power(base, exp)` that computes base^exp using fast exponentiation (no ** operator).",
        "ground_truth": "def power(base, exp):\n    if exp == 0:\n        return 1\n    if exp < 0:\n        return 1 / power(base, -exp)\n    half = power(base, exp // 2)\n    if exp % 2 == 0:\n        return half * half\n    return half * half * base",
        "test_cases": [
            {"input": "power(2, 10)", "expected": "1024"},
            {"input": "power(3, 0)", "expected": "1"},
            {"input": "power(5, 3)", "expected": "125"},
        ],
    },
    {
        "question": "Write a Python function `rotate_list(lst, k)` that rotates a list k positions to the right.",
        "ground_truth": "def rotate_list(lst, k):\n    if not lst:\n        return lst\n    k = k % len(lst)\n    return lst[-k:] + lst[:-k] if k else lst[:]",
        "test_cases": [
            {"input": "rotate_list([1, 2, 3, 4, 5], 2)", "expected": "[4, 5, 1, 2, 3]"},
            {"input": "rotate_list([1, 2, 3], 0)", "expected": "[1, 2, 3]"},
            {"input": "rotate_list([], 5)", "expected": "[]"},
        ],
    },
    {
        "question": "Write a Python function `count_inversions(arr)` that counts the number of inversions in a list (pairs where i < j but arr[i] > arr[j]).",
        "ground_truth": "def count_inversions(arr):\n    count = 0\n    for i in range(len(arr)):\n        for j in range(i + 1, len(arr)):\n            if arr[i] > arr[j]:\n                count += 1\n    return count",
        "test_cases": [
            {"input": "count_inversions([2, 4, 1, 3, 5])", "expected": "3"},
            {"input": "count_inversions([1, 2, 3])", "expected": "0"},
            {"input": "count_inversions([3, 2, 1])", "expected": "3"},
        ],
    },
    {
        "question": "Write a Python function `run_length_encode(s)` that returns the run-length encoding of a string as a list of (char, count) tuples.",
        "ground_truth": "def run_length_encode(s):\n    if not s:\n        return []\n    result = []\n    cur, count = s[0], 1\n    for c in s[1:]:\n        if c == cur:\n            count += 1\n        else:\n            result.append((cur, count))\n            cur, count = c, 1\n    result.append((cur, count))\n    return result",
        "test_cases": [
            {"input": "run_length_encode('aaabbc')", "expected": "[('a', 3), ('b', 2), ('c', 1)]"},
            {"input": "run_length_encode('')", "expected": "[]"},
            {"input": "run_length_encode('abc')", "expected": "[('a', 1), ('b', 1), ('c', 1)]"},
        ],
    },
    {
        "question": "Write a Python function `kadane(arr)` that returns the maximum subarray sum.",
        "ground_truth": "def kadane(arr):\n    if not arr:\n        return 0\n    max_sum = cur_sum = arr[0]\n    for x in arr[1:]:\n        cur_sum = max(x, cur_sum + x)\n        max_sum = max(max_sum, cur_sum)\n    return max_sum",
        "test_cases": [
            {"input": "kadane([-2, 1, -3, 4, -1, 2, 1, -5, 4])", "expected": "6"},
            {"input": "kadane([1, 2, 3])", "expected": "6"},
            {"input": "kadane([-1, -2, -3])", "expected": "-1"},
        ],
    },
    {
        "question": "Write a Python function `topological_sort(graph)` where graph is a dict mapping node to list of dependencies. Return a valid topological ordering.",
        "ground_truth": "def topological_sort(graph):\n    visited, order = set(), []\n    def dfs(node):\n        if node in visited:\n            return\n        visited.add(node)\n        for dep in graph.get(node, []):\n            dfs(dep)\n        order.append(node)\n    for node in graph:\n        dfs(node)\n    return order",
        "test_cases": [
            {"input": "topological_sort({'a': ['b', 'c'], 'b': ['c'], 'c': []})", "expected": "['c', 'b', 'a']"},
            {"input": "topological_sort({})", "expected": "[]"},
        ],
    },
]


def build_factual(items, category):
    result = []
    prefix = "fact_common" if category == "factual_common" else "fact_obscure"
    for i, (q, a) in enumerate(items):
        result.append({
            "id": f"{prefix}_{i+1:02d}",
            "category": category,
            "prompt": TASK_PROMPT.format(question=q),
            "ground_truth": a,
            "verification": "exact",
        })
    return result


def build_code(items, category):
    result = []
    prefix = "code_simple" if category == "code_simple" else "code_algo"
    for i, item in enumerate(items):
        result.append({
            "id": f"{prefix}_{i+1:02d}",
            "category": category,
            "prompt": TASK_PROMPT.format(question=item["question"]),
            "ground_truth": item["ground_truth"],
            "verification": "code",
            "test_cases": item["test_cases"],
        })
    return result


def main():
    all_tasks = []
    all_tasks.extend(gen_arith_easy())
    all_tasks.extend(gen_arith_hard())
    all_tasks.extend(build_factual(FACTUAL_COMMON, "factual_common"))
    all_tasks.extend(build_factual(FACTUAL_OBSCURE, "factual_obscure"))
    all_tasks.extend(build_code(CODE_SIMPLE, "code_simple"))
    all_tasks.extend(build_code(CODE_ALGORITHMIC, "code_algorithmic"))

    assert len(all_tasks) == 100, f"Expected 100 tasks, got {len(all_tasks)}"

    # Verify all ground truths
    errors = []
    for t in all_tasks:
        if t["verification"] == "arithmetic":
            if not verify_arithmetic(t["expression"], t["ground_truth"]):
                errors.append(f"{t['id']}: {t['expression']} != {t['ground_truth']}")
        elif t["verification"] == "code":
            if not verify_code(t["ground_truth"], t["test_cases"]):
                errors.append(f"{t['id']}: ground truth fails test cases")

    if errors:
        print("GROUND TRUTH ERRORS:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    out_path = pathlib.Path(__file__).resolve().parent.parent / "tasks" / "task_set.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(all_tasks, indent=2))
    print(f"Generated {len(all_tasks)} tasks to {out_path}")

    # Category summary
    cats = {}
    for t in all_tasks:
        cats[t["category"]] = cats.get(t["category"], 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
