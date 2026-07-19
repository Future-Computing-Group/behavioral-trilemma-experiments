"""Plot the finite-N convexity-violation rate (descriptive surface geometry).

Reads H3_convexity_by_N from hypothesis_results.json and produces a
publication-quality PDF showing the midpoint-interpolation violation
rate vs N with exact binomial 95% CIs. This supports the manuscript's
*descriptive* surface-geometry analysis (H3 was retired as a hypothesis:
the rising rate at large N is the saturation-plateau signature of H2,
not evidence about the curvature of the achievable region), so the
figure carries no hypothesis label and no tolerance criterion line
(the retired 15% falsification tolerance was dropped per the 2026-07-19
audit, C-3/LB-14; the annotated p is printed at its real value, C-10).

Usage:
    python -m scripts.plot_h3_convexity_by_N
"""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_ROOT = pathlib.Path(__file__).resolve().parent.parent
IN_PATH = _ROOT / "experiment_output" / "analysis" / "hypothesis_results.json"
OUT_DIR = _ROOT / "experiment_output" / "analysis" / "figures"
OUT_PDF = OUT_DIR / "h3_convexity_by_N.pdf"
OUT_PNG = OUT_DIR / "h3_convexity_by_N.png"


def format_p(p: float) -> str:
    """Format a p-value for in-figure mathtext, never truncating to 0.00.

    >= 1e-3: three decimals ('0.023'); below that: scientific notation
    with one significant digit ('3 \\times 10^{-4}'), matching the
    manuscript's caption style (audit C-10: the old '%.2f' printed
    'p = 0.00' for p = 3.09e-4).
    """
    if not np.isfinite(p):
        return "nan"
    if p >= 1e-3:
        return f"{p:.3f}"
    exponent = math.floor(math.log10(p))
    mantissa = p / 10 ** exponent
    if round(mantissa) >= 10:  # e.g. 9.7e-4 -> 10 x 10^-4 -> 1 x 10^-3
        mantissa /= 10.0
        exponent += 1
    return rf"{round(mantissa):.0f} \times 10^{{{exponent}}}"


def build_figure(results: dict):
    """Build the figure from a hypothesis-results dict; returns (fig, ax)."""
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

    # Single series: no legend (the redundant one-entry legend overlapped
    # the curve once the retired tolerance line was dropped); the caption
    # and y-label carry the "exact binomial 95% CI" reading.
    ax.errorbar(ns, rates, yerr=yerr, fmt="o-", color="#1f77b4",
                capsize=3, linewidth=1.3, markersize=5)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Selection size $N$")
    ax.set_ylabel("Midpoint-interpolation violation rate")
    ax.set_ylim(-0.02, max(0.55, max(ci_hi) + 0.05))
    ax.set_title("Finite-$N$ convexity-violation rate (descriptive)")

    rho = trend.get("spearman_rho", float("nan"))
    p_rho = trend.get("p_value_two_sided", float("nan"))
    if np.isfinite(rho):
        txt = (f"Spearman $\\rho$(N, rate) = {rho:+.2f}\n"
               f"two-sided $p$ = ${format_p(p_rho)}$")
        ax.text(0.98, 0.96, txt, transform=ax.transAxes,
                ha="right", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.5))

    ax.grid(True, axis="y", alpha=0.3)
    return fig, ax


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"Results JSON not found: {IN_PATH}. "
                         f"Run scripts.regenerate_hypothesis_results first.")

    with open(IN_PATH) as f:
        results = json.load(f)

    fig, ax = build_figure(results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")

    rows = sorted(results["H3_convexity_by_N"]["by_N"].values(),
                  key=lambda r: r["N"])
    for r in rows:
        print(f"  N={r['N']:>3}: {r['violations']:>2}/{r['total_tests']:>2} "
              f"= {r['violation_rate']:.1%} "
              f"(95% CI [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}])")
    trend = results["H3_convexity_by_N"]["trend"]
    rho = trend.get("spearman_rho", float("nan"))
    p_rho = trend.get("p_value_two_sided", float("nan"))
    if np.isfinite(rho):
        print(f"  Spearman(N, rate) = {rho:+.3f} "
              f"(p_two-sided = {p_rho:.2e})")


if __name__ == "__main__":
    main()
