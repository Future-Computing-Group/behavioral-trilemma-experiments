#!/usr/bin/env python3
"""Pre-specified smoke-gate checker (EXPERIMENT-PLAN §11.E-B.4).

Applies the four frozen content-validation rules to a model's SG-1/SG-2
smoke outputs and writes ``SMOKE_GATE.json``. Exit 0 iff the gate PASSES;
on FAIL the slot is swapped to the next fallback (§11.E-B.1) and the swap
is OPSLOG'd BEFORE any full run.

Rules (thresholds are module constants, frozen by §11.E-B.4):

1. ``answer_per_category`` — >= 1 parsed ANSWER in each of the six §4.1
   categories over the smoke completions.
2. ``r_validity`` / ``r_distinctness`` — every completion that carries an
   ``r`` field has r in [0.01, 1.0] (the logprob clip range; the Stage-A
   smoke records are the carriers by design), and >= 10 of the 12 SG-2
   (task, seed) records show >= 2 distinct r values (seed sensitivity,
   L65/L88).
3. ``oracle_nondegenerate`` — 0 < sum(y) < n_completions under the strict
   verifier (all-0 indicates verifier-format mismatch for the family).
4. ``parse_fail_rate`` — unparsed-ANSWER fraction (``a is None``) <= 10%.

Usage::

    python -m scripts.smoke_gate --model gemma2:9b \
        --pool experiment_output/raw_runs/logprob-gemma2_9b/pools/smoke_pool.jsonl \
        [--phase0-csv ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PARSE_FAIL_MAX = 0.10
MIN_RECORDS_WITH_DISTINCT_R = 10
R_CLIP_LO, R_CLIP_HI = 0.01, 1.0
REQUIRED_CATEGORIES = frozenset({
    "arithmetic_easy", "arithmetic_hard", "code_algorithmic",
    "code_simple", "factual_common", "factual_obscure",
})


def evaluate_gate(completions: list[dict], *, n_records: int,
                  records_with_2plus_r: int,
                  categories_with_answer: set[str]) -> dict:
    """Apply the four §11.E-B.4 rules. Pure function over aggregates.

    ``completions`` drive rules 1 (via the caller-supplied
    ``categories_with_answer``), 3 and 4; the r-distinctness aggregate of
    rule 2 is computed by the caller over the SG-2 (task, seed) records.
    """
    n = len(completions)
    n_unparsed = sum(1 for c in completions if c.get("a") is None)
    unparsed_frac = (n_unparsed / n) if n else 1.0
    y_sum = sum(c.get("y", 0) for c in completions)
    rs = [c["r"] for c in completions if "r" in c]
    r_ok = all(r is not None and R_CLIP_LO <= r <= R_CLIP_HI for r in rs)

    rule_results = {
        "answer_per_category":
            REQUIRED_CATEGORIES <= set(categories_with_answer),
        "r_validity": r_ok,
        "r_distinctness":
            records_with_2plus_r >= MIN_RECORDS_WITH_DISTINCT_R,
        "oracle_nondegenerate": 0 < y_sum < n,
        "parse_fail_rate": unparsed_frac <= PARSE_FAIL_MAX,
    }
    failed = [k for k, ok in rule_results.items() if not ok]
    return {
        "pass": not failed,
        "failed_rules": failed,
        "rule_results": rule_results,
        "unparsed_frac": unparsed_frac,
        "y_sum": y_sum,
        "n_completions": n,
        "n_records": n_records,
        "records_with_2plus_r": records_with_2plus_r,
    }


def _load_pool_records(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="smoke_gate",
        description="§11.E-B.4 smoke-gate checker (exit 0 = PASS).",
    )
    p.add_argument("--model", required=True, help="Ollama model tag.")
    p.add_argument("--pool", type=pathlib.Path, required=True, nargs="+",
                   help="SG-2 smoke pool jsonl file(s).")
    p.add_argument("--phase0-csv", type=pathlib.Path, default=None,
                   help="SG-1 Phase-0 smoke CSV (optional). Correct counts "
                        "(p_hat x --phase0-seeds) are folded into the "
                        "oracle-nondegeneracy rule; SG-1 parse fails are "
                        "visible in its STATUS.json — pass --phase0-status "
                        "to fold them into the parse-fail rule.")
    p.add_argument("--phase0-seeds", type=int, default=3,
                   help="Number of SG-1 smoke seeds per task (default 3, "
                        "per §11.E-B.4 SG-1).")
    p.add_argument("--phase0-status", type=pathlib.Path, default=None,
                   help="STATUS.json of the SG-1 smoke run; its "
                        "parse_fail_count/completions_done are folded "
                        "into the parse-fail rule as a-None completions.")
    p.add_argument("--out", type=pathlib.Path, default=None,
                   help="Output JSON (default: experiment_output/raw_runs/"
                        "logprob-<slug>/SMOKE_GATE.json).")
    args = p.parse_args(argv)

    slug = args.model.replace(":", "_").replace("/", "_")
    out = args.out or (ROOT / "experiment_output" / "raw_runs"
                       / f"logprob-{slug}" / "SMOKE_GATE.json")

    completions: list[dict] = []
    n_records = 0
    records_with_2plus_r = 0
    for pool_path in args.pool:
        for rec in _load_pool_records(pool_path):
            n_records += 1
            rs = {c.get("r") for c in rec["completions"]}
            if len(rs) >= 2:
                records_with_2plus_r += 1
            for c in rec["completions"]:
                completions.append({"category": rec["category"],
                                    "a": c.get("a"), "y": c.get("y", 0),
                                    "r": c.get("r")})

    # SG-1 contributions (no per-completion r by design — Phase-0 path).
    sg1: list[dict] = []
    if args.phase0_csv is not None:
        import csv as _csv
        with args.phase0_csv.open(newline="") as f:
            for row in _csv.DictReader(f):
                n_correct = round(float(row["p_hat"]) * args.phase0_seeds)
                for i in range(args.phase0_seeds):
                    sg1.append({"category": "phase0_smoke", "a": "parsed",
                                "y": 1 if i < n_correct else 0})
    if args.phase0_status is not None:
        st = json.loads(args.phase0_status.read_text())
        fails = int(st.get("parse_fail_count", 0))
        # Mark the y=0 SG-1 slots as unparsed, up to the recorded count
        # (parse-failed completions are y=0 under the strict verifier).
        for c in sg1:
            if fails == 0:
                break
            if c["y"] == 0 and c["a"] is not None:
                c["a"] = None
                fails -= 1
    completions.extend(sg1)

    categories_with_answer = {c["category"] for c in completions
                              if c.get("a") is not None}
    result = evaluate_gate(completions, n_records=n_records,
                           records_with_2plus_r=records_with_2plus_r,
                           categories_with_answer=categories_with_answer)
    payload = {"model": args.model, **result}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    verdict = "PASS" if result["pass"] else f"FAIL {result['failed_rules']}"
    print(f"smoke gate [{args.model}]: {verdict} "
          f"(unparsed_frac={result['unparsed_frac']:.3f}, "
          f"y_sum={result['y_sum']}/{result['n_completions']}) -> {out}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
