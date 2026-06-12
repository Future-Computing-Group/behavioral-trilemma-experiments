"""R9: pre-specified smoke-gate checker (EXPERIMENT-PLAN §11.E-B.4).

Pure-function tests over synthetic records; each rule gets a boundary pair.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.smoke_gate import evaluate_gate  # noqa: E402


def _comps(n_parsed, n_unparsed, y_ones, cat="arithmetic_easy"):
    cs = []
    for i in range(n_parsed):
        cs.append({"category": cat, "a": "42", "y": 1 if i < y_ones else 0,
                   "r": 0.5 + 0.001 * i})
    for _ in range(n_unparsed):
        cs.append({"category": cat, "a": None, "y": 0, "r": 0.5})
    return cs


ALL_CATS = {"arithmetic_easy", "arithmetic_hard", "code_algorithmic",
            "code_simple", "factual_common", "factual_obscure"}


def test_parse_fail_boundary_7_of_66_fails_6_passes():
    base = dict(records_with_2plus_r=12, n_records=12,
                categories_with_answer=set(ALL_CATS))
    ok = evaluate_gate(_comps(60, 6, 30), **base)
    bad = evaluate_gate(_comps(59, 7, 30), **base)
    assert ok["pass"] and ok["unparsed_frac"] <= 0.10
    assert not bad["pass"] and bad["failed_rules"] == ["parse_fail_rate"]


def test_degenerate_oracle_fails():
    base = dict(records_with_2plus_r=12, n_records=12,
                categories_with_answer=set(ALL_CATS))
    assert not evaluate_gate(_comps(66, 0, 0), **base)["pass"]     # all y=0
    assert not evaluate_gate(_comps(66, 0, 66), **base)["pass"]    # all y=1


def test_missing_category_fails():
    out = evaluate_gate(_comps(66, 0, 30),
                        records_with_2plus_r=12, n_records=12,
                        categories_with_answer={"arithmetic_easy"})
    assert not out["pass"] and "answer_per_category" in out["failed_rules"]


def test_r_degeneracy_boundary_10_of_12():
    base = dict(n_records=12, categories_with_answer=set(ALL_CATS))
    assert evaluate_gate(_comps(66, 0, 30),
                         records_with_2plus_r=10, **base)["pass"]
    assert not evaluate_gate(_comps(66, 0, 30),
                             records_with_2plus_r=9, **base)["pass"]


def test_out_of_range_r_fails_validity():
    base = dict(records_with_2plus_r=12, n_records=12,
                categories_with_answer=set(ALL_CATS))
    comps = _comps(65, 0, 30) + [{"category": "arithmetic_easy", "a": "42",
                                  "y": 0, "r": 0.005}]   # below the 0.01 clip
    out = evaluate_gate(comps, **base)
    assert not out["pass"] and "r_validity" in out["failed_rules"]


def test_cli_writes_gate_json(tmp_path):
    from scripts.smoke_gate import main
    # SG-2 smoke pool: 6 tasks x 2 seeds = 12 records x 2 completions,
    # all parsed, mixed y, distinct r within each record; all six
    # categories covered (§11.E-B.4 smoke design).
    cats = sorted(ALL_CATS)
    recs = []
    for seed in (42, 123):
        for i, cat in enumerate(cats):
            recs.append({"task_id": f"{cat}_01", "category": cat,
                         "seed": seed, "confidence_source": "logprob",
                         "completions": [
                             {"idx": 0, "r": 0.6, "y": i % 2, "a": "42"},
                             {"idx": 1, "r": 0.8, "y": 1 - i % 2, "a": "43"}]})
    pool = tmp_path / "pool_gemma2_9b_seed42_N32.jsonl"
    with pool.open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    out = tmp_path / "SMOKE_GATE.json"
    rc = main(["--model", "gemma2:9b", "--pool", str(pool),
               "--out", str(out)])
    data = json.loads(out.read_text())
    assert data["model"] == "gemma2:9b"
    assert data["pass"] is True and rc == 0
    assert "rule_results" in data and "unparsed_frac" in data
