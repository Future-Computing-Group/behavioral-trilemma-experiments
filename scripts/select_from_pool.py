#!/usr/bin/env python3
"""Stage-B selector: regenerate the 540 logprob CSVs from the existing pools.

Background. The 540-configuration logprob sweep behind §7 Table 1 was a
TWO-STAGE pipeline. Stage A (pool generation) called Ollama
``/v1/chat/completions`` with ``logprobs: true`` at N=32 for each
(task, seed), capturing per-completion {idx, raw, r_logprob, r_verbalized,
mean_logprob, a (parsed answer), y (oracle correctness)}. Stage A wrote
``experiment_output/raw_runs/logprob/pools/pool_qwen2.5_7b_seed{SEED}_N32.jsonl``
(5 jsonl files, 100 records/file, 32 completions/record), plus a small
``pool_meta_qwen2.5_7b_N32.json`` manifest. Stage A's source code was never
committed to this repo (see ``OPSLOG.md`` 2026-06-02 + audit B-A1-1) but
its OUTPUTS survive on local disk.

Stage B (this module) is purely deterministic: for each of the 540 configs
(N, w_ratio, r_min, seed), subset the FIRST N completions from the pool
record, compute the oracle payoff
::
    V_i = -w_C (r_i - y_i)^2 + w_A * 1{r_i >= r_min}            # main.tex Eq. 11
and select i* = argmax V (first-index tie-breaking). The selected row is
written to ``experiment_output/raw_runs/logprob/results/qwen2.5_7b_N{N}_w{W}_r{R}_s{S}.csv``
in the 19-column schema the analysis pipeline (``analysis/hypothesis_tests.py``
+ ``scripts/regenerate_hypothesis_results.py``) consumes.

The selector's load-bearing invariant is BYTE-EQUALITY with the shipped CSVs
(committed in ``b453d6d`` 2026-04-20 by Nam Do): for every (config, task)
pair we regenerate, the row string must match the shipped row string
character-for-character. This means the contract is testable end-to-end
without re-running Ollama; a successful run reproduces the manuscript's
headline data exactly from the surviving pools.

Usage
-----
::

    python -m scripts.select_from_pool --config qwen2.5_7b_N1_w0_r0.5_s0   # one cell
    python -m scripts.select_from_pool --all                              # 540 configs
    python -m scripts.select_from_pool --verify-against-shipped           # diff vs shipped

See ``tests/test_select_from_pool.py`` for the contract.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from typing import Iterator

# 19-column schema, in the exact order observed in the shipped CSVs (see
# any ``experiment_output/raw_runs/logprob/results/qwen2.5_7b_N*.csv`` header).
SHIPPED_FIELDS = [
    "task_id", "category", "N", "w_ratio", "w_C", "w_A", "r_min", "seed",
    "payoff_mode", "selection_mode", "n_candidates", "n_parsed",
    "selected_index", "r_selected", "a_selected", "y", "V_selected",
    "brier", "gate_cleared",
]

ROOT = pathlib.Path(__file__).resolve().parent.parent
POOLS_DIR = ROOT / "experiment_output" / "raw_runs" / "logprob" / "pools"
RESULTS_DIR = ROOT / "experiment_output" / "raw_runs" / "logprob" / "results"

# The full 540-config grid (matches ``configs/params.yaml`` ``experiment``
# section, also pinned to the manuscript's §exp-iv).
N_VALUES = [1, 2, 4, 8, 16, 32]
W_RATIOS = [0, 0.25, 0.5, 1.0, 2.0, 4.0]
R_MINS = [0.5, 0.7, 0.9]
SEEDS = [42, 123, 456, 789, 0]

W_C = 1.0  # Fixed per ``configs/params.yaml`` ``experiment.w_C``.

MODEL_SLUG = "qwen2.5_7b"  # filesystem slug for ``qwen2.5:7b``.


# ---------------------------------------------------------------------------
# Pure selection logic (the testable contract)
# ---------------------------------------------------------------------------

def _oracle_payoff(r: float, y: int, w_C: float, w_A: float, r_min: float) -> float:
    """Per-completion oracle payoff (main.tex Eq. 11).

    Duplicates ``src.scorer.oracle_payoff`` to keep this module self-contained
    and to make the formula visible at the call site below. Tests will catch
    drift between the two.
    """
    return -w_C * (r - y) ** 2 + (w_A if r >= r_min else 0.0)


def comonotone_payoff(r: float, y: int, w_C: float, w_A: float,
                      r_min: float) -> float:
    """EXPERIMENT-PLAN §11.E-A.1: Eq. 11 with y -> 1 on {r >= r_min} only.
    Strictly increasing in r on [r_min, 1) regardless of y, so prop:bon
    hypothesis (c) holds surely. Below the gate, identical to Eq. 11."""
    if r >= r_min:
        return -w_C * (1.0 - r) ** 2 + w_A
    return -w_C * (r - y) ** 2


_SELECTORS = {"oracle": _oracle_payoff, "comonotone": comonotone_payoff}


def select_for_config(pool_record: dict, *, N: int, w_C: float, w_A: float,
                      r_min: float, w_ratio: float,
                      selector: str = "oracle") -> dict:
    """Run Stage B on one (task, seed) pool record for one config.

    Returns a 19-key dict in ``SHIPPED_FIELDS`` order, ready for
    ``csv.DictWriter.writerow``. ``n_parsed`` counts completions[:N] with
    ``a is not None`` (the parse-success criterion observed in the data).
    """
    completions = pool_record["completions"][:N]
    n_candidates = len(completions)
    # ``n_parsed`` in the shipped CSVs counts completions whose confidence ``r``
    # was extractable, NOT completions with a parsed ANSWER. In the logprob
    # path ``r`` is always present (geomean over the token logprobs) so this
    # equals ``n_candidates``. Verified against
    # ``qwen2.5_7b_N32_w*_r*_s456.csv`` row ``code_simple_11`` where the
    # pool has one completion with ``a=None`` (idx=21) and the shipped CSV
    # still reports ``n_parsed=32``. The ``a=None`` completion is INCLUDED
    # in the argmax (shipped selects it at several configs with
    # ``a_selected=''``).
    n_parsed = sum(1 for c in completions if c.get("r") is not None)

    payoff = _SELECTORS[selector]

    # argmax V with first-index tie-breaking (Python's ``max`` keeps the
    # earlier element on equality).
    best_i = 0
    best_V = payoff(
        completions[0]["r"], completions[0]["y"], w_C, w_A, r_min,
    )
    for i in range(1, n_candidates):
        v = payoff(
            completions[i]["r"], completions[i]["y"], w_C, w_A, r_min,
        )
        if v > best_V:
            best_V = v
            best_i = i

    chosen = completions[best_i]
    r_sel = chosen["r"]
    y_sel = chosen["y"]
    a_sel = chosen["a"] if chosen.get("a") is not None else ""
    brier = (r_sel - y_sel) ** 2
    gate_cleared = 1 if r_sel >= r_min else 0

    return {
        "task_id": pool_record["task_id"],
        "category": pool_record["category"],
        "N": N,
        "w_ratio": w_ratio,
        "w_C": w_C,
        "w_A": w_A,
        "r_min": r_min,
        "seed": pool_record["seed"],
        "payoff_mode": selector,
        "selection_mode": "argmax",
        "n_candidates": n_candidates,
        "n_parsed": n_parsed,
        "selected_index": best_i,
        "r_selected": r_sel,
        "a_selected": a_sel,
        "y": y_sel,
        "V_selected": best_V,
        "brier": brier,
        "gate_cleared": gate_cleared,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _format_w_ratio(w_ratio: float) -> str:
    """Filename convention: 0, 0.25, 0.5, 1.0, 2.0, 4.0 (no trailing ``.0``
    for the integer zero). Matches the shipped filenames exactly."""
    if w_ratio == 0:
        return "0"
    return str(w_ratio)


def _format_r_min(r_min: float) -> str:
    return str(r_min)


def config_filename(N: int, w_ratio: float, r_min: float, seed: int,
                    model_slug: str = MODEL_SLUG) -> str:
    return (
        f"{model_slug}_N{N}_w{_format_w_ratio(w_ratio)}_"
        f"r{_format_r_min(r_min)}_s{seed}.csv"
    )


def iter_full_grid() -> Iterator[tuple[int, float, float, int]]:
    """Yield the 540 (N, w_ratio, r_min, seed) tuples in the same order as
    the shipped filenames. Order doesn't affect content (per-cell CSV) but
    keeps the driver's stdout deterministic."""
    for N in N_VALUES:
        for w_ratio in W_RATIOS:
            for r_min in R_MINS:
                for seed in SEEDS:
                    yield N, w_ratio, r_min, seed


def _read_pool_file(path: pathlib.Path) -> list[dict]:
    """Uncached inner reader (the L97 counting seam for tests)."""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# L97: a 540-config sweep calls _load_pool once per config; without a memo
# that is 108 redundant re-parses of each ~3,200-record pool file. The memo
# is per-process and keyed on the resolved pool path (which encodes seed,
# pools_dir and model slug).
_POOL_CACHE: dict[pathlib.Path, list[dict]] = {}


def clear_pool_cache() -> None:
    """Empty the per-process pool memo (tests; long-lived processes)."""
    _POOL_CACHE.clear()


def _load_pool(seed: int, pools_dir: pathlib.Path = POOLS_DIR) -> list[dict]:
    path = (pools_dir / f"pool_{MODEL_SLUG}_seed{seed}_N32.jsonl").resolve()
    if path not in _POOL_CACHE:
        _POOL_CACHE[path] = _read_pool_file(path)
    return _POOL_CACHE[path]


def write_config_csv(N: int, w_ratio: float, r_min: float, seed: int,
                     out_dir: pathlib.Path,
                     pools_dir: pathlib.Path = POOLS_DIR,
                     selector: str = "oracle") -> pathlib.Path:
    """Regenerate one config's CSV from the pool for ``seed``."""
    w_A = W_C * w_ratio
    # Pool jsonl files are written in category-grouped order
    # (arith_easy, arith_hard, code_simple, fact_common, fact_obscure,
    # code_algo). Shipped CSVs iterate tasks in alphabetical order — verified
    # against ``qwen2.5_7b_N1_w0_r0.5_s0.csv`` whose row order is
    # arith_*, code_algo_*, code_simple_*, fact_common_*, fact_obscure_*.
    # Sort to match.
    pool_records = sorted(_load_pool(seed, pools_dir), key=lambda d: d["task_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / config_filename(N, w_ratio, r_min, seed)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHIPPED_FIELDS)
        writer.writeheader()
        for record in pool_records:
            row = select_for_config(
                record, N=N, w_C=W_C, w_A=w_A, r_min=r_min, w_ratio=w_ratio,
                selector=selector,
            )
            writer.writerow(row)
    return out_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def diff_against_shipped(out_path: pathlib.Path,
                         shipped_dir: pathlib.Path = RESULTS_DIR) -> list[str]:
    """Return a list of diff-line descriptions; empty means byte-equal."""
    shipped = shipped_dir / out_path.name
    if not shipped.exists():
        return [f"NO_SHIPPED_COUNTERPART: {shipped}"]
    a_lines = shipped.read_text().splitlines()
    b_lines = out_path.read_text().splitlines()
    diffs = []
    if len(a_lines) != len(b_lines):
        diffs.append(f"LINE_COUNT: shipped={len(a_lines)} regen={len(b_lines)}")
    for i, (al, bl) in enumerate(zip(a_lines, b_lines)):
        if al != bl:
            diffs.append(f"LINE {i}: shipped={al!r}")
            diffs.append(f"        regen ={bl!r}")
            if len(diffs) > 12:
                diffs.append("... (truncated)")
                break
    return diffs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="select_from_pool",
        description="Stage-B selector for the 540-config logprob sweep.",
    )
    p.add_argument(
        "--config",
        help="Single config descriptor "
             "'qwen2.5_7b_N<N>_w<W>_r<R>_s<S>' (filename without .csv).",
    )
    p.add_argument("--all", action="store_true",
                   help="Regenerate the full 540-config grid.")
    p.add_argument("--out-dir", type=pathlib.Path, default=RESULTS_DIR,
                   help=f"Output dir (default: {RESULTS_DIR}).")
    p.add_argument("--verify-against-shipped", action="store_true",
                   help="After writing, diff each CSV vs the shipped one. "
                        "Exits non-zero on any divergence.")
    p.add_argument("--selector", choices=sorted(_SELECTORS),
                   default="oracle",
                   help="Selection payoff: 'oracle' (Eq. 11, default) or "
                        "'comonotone' (EXPERIMENT-PLAN §11.E-A.1 V_c).")
    args = p.parse_args(argv)

    if not args.config and not args.all:
        p.error("--config NAME or --all required")
    if args.config and args.all:
        p.error("--config and --all are mutually exclusive")
    if args.selector != "oracle" and args.verify_against_shipped:
        p.error("--verify-against-shipped only applies to the oracle "
                "selector (shipped CSVs are oracle-generated; a "
                "comonotone run has a different contract)")

    # Distinguish "write to a different dir" from "overwrite shipped".
    # Default refuses to overwrite; --verify-against-shipped reads shipped.
    if args.out_dir == RESULTS_DIR:
        # Sanity guard: writing into the shipped dir would clobber the
        # ground-truth comparison target. Force the caller to be explicit.
        print(
            "ERROR: refusing to write into the shipped results dir. "
            "Pass --out-dir <somewhere-else> (e.g., /tmp/regen).",
            file=sys.stderr,
        )
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    if args.config:
        # Parse the descriptor.
        # Format: qwen2.5_7b_N<N>_w<W>_r<R>_s<S>
        import re
        m = re.match(
            r"^qwen2\.5_7b_N(\d+)_w([0-9.]+)_r([0-9.]+)_s(\d+)$",
            args.config,
        )
        if not m:
            p.error(f"could not parse config descriptor {args.config!r}")
        N = int(m.group(1))
        # Map the matched w-string back to the canonical W_RATIOS entry so
        # int-0 vs float-0.0 typing round-trips through filenames cleanly
        # (the shipped CSVs serialise the integer zero without ``.0``).
        w_str = m.group(2)
        w_ratio = next((v for v in W_RATIOS if _format_w_ratio(v) == w_str), None)
        if w_ratio is None:
            p.error(f"unknown w_ratio {w_str!r} (expected one of "
                    f"{[_format_w_ratio(v) for v in W_RATIOS]})")
        r_min = float(m.group(3))
        seed = int(m.group(4))
        written.append(write_config_csv(N, w_ratio, r_min, seed, args.out_dir,
                                        selector=args.selector))
    else:
        for N, w_ratio, r_min, seed in iter_full_grid():
            written.append(write_config_csv(N, w_ratio, r_min, seed,
                                            args.out_dir,
                                            selector=args.selector))

    print(f"wrote {len(written)} CSVs to {args.out_dir}")

    if args.verify_against_shipped:
        n_clean = 0
        n_drift = 0
        for path in written:
            diffs = diff_against_shipped(path)
            if not diffs:
                n_clean += 1
            else:
                n_drift += 1
                print(f"DRIFT: {path.name}")
                for d in diffs:
                    print(f"  {d}")
        print(f"verified: {n_clean} clean, {n_drift} drift (of {len(written)})")
        return 0 if n_drift == 0 else 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
