"""Figure generation for the behavioral trilemma experiment."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Publication defaults
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

COLORS = {
    0: "#888888",      # w_A=0 (control)
    0.25: "#2997ff",   # blue
    0.5: "#bf5af2",    # purple
    1.0: "#30d158",    # green
    2.0: "#ff9f0a",    # orange
    4.0: "#ff453a",    # red
}


def plot_brier_vs_n(df: pd.DataFrame, output_dir: pathlib.Path,
                    r_min: float = 0.7):
    """Brier score vs N curves, one line per w_A/w_C ratio."""
    sub = df[df["r_min"] == r_min]
    agg = sub.groupby(["N", "w_ratio"])["brier"].agg(["mean", "sem"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for w_ratio in sorted(agg["w_ratio"].unique()):
        ws = agg[agg["w_ratio"] == w_ratio].sort_values("N")
        color = COLORS.get(w_ratio, "#666666")
        label = f"$w_A/w_C = {w_ratio}$"
        ax.errorbar(ws["N"], ws["mean"], yerr=1.96 * ws["sem"],
                     marker="o", markersize=4, color=color, label=label,
                     capsize=3, linewidth=1.5)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Selection pressure $N$")
    ax.set_ylabel("Mean Brier score")
    ax.set_title(f"Calibration degradation under Best-of-$N$ ($r_{{\\min}} = {r_min}$)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"brier_vs_n_rmin{r_min}.pdf")
    fig.savefig(output_dir / f"brier_vs_n_rmin{r_min}.png")
    plt.close(fig)


def plot_pareto_frontier(df: pd.DataFrame, output_dir: pathlib.Path,
                         r_min: float = 0.7):
    """3D scatter of (H, C, A) triples across weight vectors."""
    sub = df[(df["r_min"] == r_min) & (df["N"] == 32)]
    agg = sub.groupby("w_ratio").agg(
        H=("y", "mean"),
        BS=("brier", "mean"),
        A=("gate_cleared", "mean"),
    ).reset_index()
    agg["C"] = 1 - agg["BS"]

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    for _, row in agg.iterrows():
        color = COLORS.get(row["w_ratio"], "#666666")
        ax.scatter(row["H"], row["C"], row["A"], c=color, s=60, edgecolors="white",
                   linewidths=0.5, zorder=5)
        ax.text(row["H"], row["C"], row["A"] + 0.03,
                f'{row["w_ratio"]}', fontsize=7, ha="center")

    ax.set_xlabel("Helpfulness $H$")
    ax.set_ylabel("Calibration $C$")
    ax.set_zlabel("Autonomy $A$")
    ax.set_title(f"Pareto frontier ($N=32$, $r_{{\\min}} = {r_min}$)")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"pareto_rmin{r_min}.pdf")
    fig.savefig(output_dir / f"pareto_rmin{r_min}.png")
    plt.close(fig)


def plot_confidence_histograms(df: pd.DataFrame, output_dir: pathlib.Path,
                                r_min: float = 0.7, w_ratio: float = 1.0):
    """Confidence distribution: base (N=1) vs selected (N=32) on binding tasks."""
    sub = df[(df["r_min"] == r_min) & (df["w_ratio"] == w_ratio)]
    r_base = pd.to_numeric(sub[sub["N"] == 1]["r_selected"], errors="coerce").dropna()
    r_high = pd.to_numeric(sub[sub["N"] == 32]["r_selected"], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(r_base, bins=bins, alpha=0.5, label="$N=1$ (base)", color="#888888",
            density=True, edgecolor="white")
    ax.hist(r_high, bins=bins, alpha=0.6, label="$N=32$ (selected)", color="#ff453a",
            density=True, edgecolor="white")
    ax.axvline(r_min, color="#2997ff", linestyle="--", linewidth=1.5,
               label=f"$r_{{\\min}} = {r_min}$")
    ax.set_xlabel("Reported confidence $r$")
    ax.set_ylabel("Density")
    ax.set_title(f"Confidence distribution ($w_A/w_C = {w_ratio}$, $r_{{\\min}} = {r_min}$)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"conf_hist_w{w_ratio}_r{r_min}.pdf")
    fig.savefig(output_dir / f"conf_hist_w{w_ratio}_r{r_min}.png")
    plt.close(fig)


def generate_all_figures(df: pd.DataFrame, output_dir: pathlib.Path):
    """Generate all standard figures."""
    for r_min in [0.5, 0.7, 0.9]:
        plot_brier_vs_n(df, output_dir, r_min=r_min)
        plot_pareto_frontier(df, output_dir, r_min=r_min)
        for w_ratio in [0.5, 1.0, 4.0]:
            plot_confidence_histograms(df, output_dir, r_min=r_min, w_ratio=w_ratio)
