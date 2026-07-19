#!/usr/bin/env python3
"""Compute the §11.E-A.2 violation rates from the archived Stage-A pools.

Iterates the §5.1 grid over the pool jsonls with the ORIGINAL oracle
selection payoff (eq. (eq:selection-payoff) in the manuscript) and aggregates the prop:bon hypothesis-(c) violation
metric (``analysis.comonotone_violation.violation_counts``) at two
granularities:

* ``grid_540_view`` — protocol-faithful: every (N, w_ratio, r_min, seed)
  cell x task. Within the gate the metric is w_A-invariant (disclosed
  structural fact), so each rate is replicated across the w-levels.
* ``collapsed_90_view`` — the (N, r_min, seed) view with the w dimension
  dropped.

Selections with fewer than two gate-cleared candidates cannot witness
(c) and are excluded from the denominators; the excluded count is
reported.

Usage::

    python -m scripts.compute_comonotone_violation
    # writes experiment_output/analysis/comonotone_violation.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.comonotone_violation import violation_counts  # noqa: E402
from scripts.select_from_pool import (  # noqa: E402
    N_VALUES, POOLS_DIR, R_MINS, SEEDS, W_C, W_RATIOS, _load_pool,
)

OUT_PATH = ROOT / "experiment_output" / "analysis" / "comonotone_violation.json"


def _rates(tallies: list[dict]) -> dict:
    eligible = [t for t in tallies if t["eligible"]]
    n_events = sum(t["event"] for t in eligible)
    n_pairs = sum(t["n_comparable_pairs"] for t in eligible)
    n_viol = sum(t["n_violations"] for t in eligible)
    return {
        "n_selections": len(tallies),
        "n_eligible": len(eligible),
        "n_excluded": len(tallies) - len(eligible),
        "n_events": n_events,
        "event_rate": (n_events / len(eligible)) if eligible else None,
        "n_comparable_pairs": n_pairs,
        "n_violating_pairs": n_viol,
        "pair_rate": (n_viol / n_pairs) if n_pairs else None,
    }


def compute_violation_summary(*, pools_dir: pathlib.Path = POOLS_DIR,
                              n_values: list[int] = N_VALUES,
                              w_ratios: list[float] = W_RATIOS,
                              r_mins: list[float] = R_MINS,
                              seeds: list[int] = SEEDS,
                              w_C: float = W_C) -> dict:
    """Aggregate violation counts over the grid; returns the JSON payload."""
    full: list[dict] = []        # (N, w, r_min, seed) x task
    collapsed: list[dict] = []   # (N, r_min, seed) x task (w dropped)
    for seed in seeds:
        records = sorted(_load_pool(seed, pools_dir),
                         key=lambda d: d["task_id"])
        for N in n_values:
            for r_min in r_mins:
                for rec in records:
                    # w_A-invariant within the gate: compute once at w_A=0
                    # and replicate across the w-levels for the 540 view.
                    out = violation_counts(rec["completions"], N=N,
                                           r_min=r_min, w_C=w_C, w_A=0.0)
                    collapsed.append(out)
                    full.extend([out] * len(w_ratios))
    return {
        "metric": "EXPERIMENT-PLAN §11.E-A.2 prop:bon(c) violation rate "
                  "(original oracle selector, manuscript eq. (eq:selection-payoff))",
        "w_A_invariance_note": (
            "Within the gate-cleared set the w_A bonus is constant, so the "
            "metric is identical across the w-levels; the 540-cell view "
            "replicates the (N, r_min, seed) tallies across w (disclosed "
            "structural fact, not a finding)."
        ),
        "grid_540_view": _rates(full),
        "collapsed_90_view": _rates(collapsed),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="compute_comonotone_violation",
        description="§11.E-A.2 violation rates from the archived pools.",
    )
    p.add_argument("--pools-dir", type=pathlib.Path, default=POOLS_DIR)
    p.add_argument("--out", type=pathlib.Path, default=OUT_PATH)
    args = p.parse_args(argv)

    summary = compute_violation_summary(pools_dir=args.pools_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    for view in ("grid_540_view", "collapsed_90_view"):
        v = summary[view]
        print(f"  {view}: event_rate={v['event_rate']} "
              f"pair_rate={v['pair_rate']} "
              f"(eligible {v['n_eligible']}, excluded {v['n_excluded']})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
