"""Plot achievable-region convexity violation rate as a function of N (descriptive).

Reads H3_convexity_by_N from hypothesis_results.json and produces
a publication-quality PDF showing the violation rate vs N with exact
binomial 95% CIs and the pre-registered 15% tolerance line. This
supports the descriptive surface-geometry analysis (the achievable
region's frontier is approximately convex; finite-N runs show tolerable
slack that decreases with selection pressure).

Usage:
    python -m scripts.plot_h3_convexity_by_N
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_ROOT = pathlib.Path(__file__).resolve().parent.parent
IN_PATH = _ROOT / "experiment_output" / "analysis" / "hypothesis_results.json"
OUT_DIR = _ROOT / "experiment_output" / "analysis" / "figures"
OUT_PDF = OUT_DIR / "h3_convexity_by_N.pdf"
OUT_PNG = OUT_DIR / "h3_convexity_by_N.png"


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"Results JSON not found: {IN_PATH}. "
                         f"Run scripts.regenerate_hypothesis_results first.")

    with open(IN_PATH) as f:
        results = json.load(f)

    by_n = results.get("H3_convexity_by_N", {}).get("by_N", {})
    trend = results.get("H3_convexity_by_N", {}).get("trend", {})
    if not by_n:
        raise SystemExit("H3_convexity_by_N not in results; regenerate JSON first.")

    rows = sorted(by_n.values(), key=lambda r: r["N"])
    ns = [r["N"] for r in rows]
    rates = [r["violation_rate"] for r in rows]
    ci_lo = [r["ci_lo"] for r in rows]
    ci_hi = [r["ci_hi"] for r in rows]
    yerr = np.array([
        [r - lo for r, lo in zip(rates, ci_lo)],
        [hi - r for r, hi in zip(rates, ci_hi)],
    ])

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "text.usetex": False,
    })

    fig, ax = plt.subplots(figsize=(4.2, 3.0))

    ax.axhline(0.15, color="#999", linestyle="--", linewidth=1.0,
               label="15% tolerance")
    ax.errorbar(ns, rates, yerr=yerr, fmt="o-", color="#1f77b4",
                capsize=3, linewidth=1.3, markersize=5,
                label="Violation rate (95% CI)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Selection size $N$")
    ax.set_ylabel("Violation rate")
    ax.set_ylim(-0.02, max(0.55, max(ci_hi) + 0.05))
    ax.set_title("H3 convexity: finite-$N$ violation rate")

    rho = trend.get("spearman_rho", float("nan"))
    p_rho = trend.get("p_value_two_sided", float("nan"))
    if np.isfinite(rho):
        txt = (f"Spearman $\\rho$(N, rate) = {rho:+.2f}\n"
               f"two-sided $p$ = {p_rho:.2f}")
        ax.text(0.98, 0.96, txt, transform=ax.transAxes,
                ha="right", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.5))

    ax.legend(loc="center right", frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")

    for r in rows:
        print(f"  N={r['N']:>3}: {r['violations']:>2}/{r['total_tests']:>2} "
              f"= {r['violation_rate']:.1%} "
              f"(95% CI [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}])")
    if np.isfinite(rho):
        print(f"  Spearman(N, rate) = {rho:+.3f} (p_two-sided={p_rho:.3f})")


if __name__ == "__main__":
    main()
