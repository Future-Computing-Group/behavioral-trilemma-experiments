"""Figure generation for the behavioral trilemma experiment.

Generates publication-quality figures for the A1 manuscript:
  - Brier score vs N curves
  - Pareto frontier 3D scatter
  - Confidence distribution histograms
  - Inflation heatmap (N × w_ratio)
  - Brier decomposition stacked bar chart
  - Effect size forest plot
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Publication defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "text.usetex": False,  # safe default; set True if LaTeX available
})

COLORS = {
    0: "#888888",      # w_A=0 (control)
    0.25: "#2997ff",   # blue
    0.5: "#bf5af2",    # purple
    1.0: "#30d158",    # green
    2.0: "#ff9f0a",    # orange
    4.0: "#ff453a",    # red
}

N_MARKERS = {1: "o", 2: "s", 4: "D", 8: "^", 16: "v", 32: "P"}


def _ensure_dir(output_dir: pathlib.Path):
    output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1: Brier score vs N
# ---------------------------------------------------------------------------

def plot_brier_vs_n(
    df: pd.DataFrame,
    output_dir: pathlib.Path,
    r_min: float = 0.7,
):
    """Brier score vs N curves, one line per w_A/w_C ratio."""
    sub = df[df["r_min"] == r_min].copy()
    sub["brier"] = pd.to_numeric(sub["brier"], errors="coerce")
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
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("Selection pressure $N$")
    ax.set_ylabel("Mean Brier score")
    ax.set_title(f"Calibration degradation under Best-of-$N$ ($r_{{\\min}} = {r_min}$)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    _ensure_dir(output_dir)
    fig.savefig(output_dir / f"brier_vs_n_rmin{r_min}.pdf")
    fig.savefig(output_dir / f"brier_vs_n_rmin{r_min}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Pareto frontier
# ---------------------------------------------------------------------------

def plot_pareto_frontier(
    df: pd.DataFrame,
    output_dir: pathlib.Path,
    r_min: float = 0.7,
):
    """3D scatter of (H, C, A) triples across weight vectors."""
    sub = df[(df["r_min"] == r_min) & (df["N"] == 32)].copy()
    for col in ["y", "brier", "gate_cleared"]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")

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
        ax.scatter(row["H"], row["C"], row["A"], c=color, s=60,
                   edgecolors="white", linewidths=0.5, zorder=5)
        ax.text(row["H"], row["C"], row["A"] + 0.03,
                f'{row["w_ratio"]}', fontsize=7, ha="center")

    ax.set_xlabel("Helpfulness $H$")
    ax.set_ylabel("Calibration $C$")
    ax.set_zlabel("Autonomy $A$")
    ax.set_title(f"Pareto frontier ($N=32$, $r_{{\\min}} = {r_min}$)")
    fig.tight_layout()

    _ensure_dir(output_dir)
    fig.savefig(output_dir / f"pareto_rmin{r_min}.pdf")
    fig.savefig(output_dir / f"pareto_rmin{r_min}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Confidence histograms
# ---------------------------------------------------------------------------

def plot_confidence_histograms(
    df: pd.DataFrame,
    output_dir: pathlib.Path,
    r_min: float = 0.7,
    w_ratio: float = 1.0,
):
    """Confidence distribution: base (N=1) vs selected (N=32) on binding tasks."""
    sub = df[(df["r_min"] == r_min) & (df["w_ratio"] == w_ratio)].copy()
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

    _ensure_dir(output_dir)
    fig.savefig(output_dir / f"conf_hist_w{w_ratio}_r{r_min}.pdf")
    fig.savefig(output_dir / f"conf_hist_w{w_ratio}_r{r_min}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Inflation heatmap
# ---------------------------------------------------------------------------

def plot_inflation_heatmap(
    inflation_df: pd.DataFrame,
    output_dir: pathlib.Path,
    r_min: float = 0.7,
):
    """Heatmap of mean inflation (Δ) by N × w_ratio.

    inflation_df should come from compute_inflation_metrics().
    """
    sub = inflation_df[inflation_df["r_min"] == r_min]
    pivot = sub.pivot_table(
        values="mean_inflation_binding",
        index="N",
        columns="w_ratio",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, cmap="RdYlBu_r", aspect="auto",
                   vmin=pivot.values[np.isfinite(pivot.values)].min() if np.any(np.isfinite(pivot.values)) else 0,
                   vmax=pivot.values[np.isfinite(pivot.values)].max() if np.any(np.isfinite(pivot.values)) else 1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"N={int(v)}" for v in pivot.index])
    ax.set_xlabel("$w_A / w_C$")
    ax.set_ylabel("Selection pressure $N$")
    ax.set_title(f"Mean confidence inflation on binding tasks ($r_{{\\min}} = {r_min}$)")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                text_color = "white" if abs(val) > 0.5 * (pivot.values[np.isfinite(pivot.values)].max() - pivot.values[np.isfinite(pivot.values)].min()) else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=text_color)

    plt.colorbar(im, ax=ax, label="Mean $\\Delta$", shrink=0.8)
    fig.tight_layout()

    _ensure_dir(output_dir)
    fig.savefig(output_dir / f"inflation_heatmap_r{r_min}.pdf")
    fig.savefig(output_dir / f"inflation_heatmap_r{r_min}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Brier decomposition
# ---------------------------------------------------------------------------

def plot_brier_decomposition(
    decomp_df: pd.DataFrame,
    output_dir: pathlib.Path,
    r_min: float = 0.7,
):
    """Stacked bar chart of Brier decomposition by w_ratio for N=32."""
    sub = decomp_df[(decomp_df["r_min"] == r_min) & (decomp_df["N"] == 32)]
    sub = sub.sort_values("w_ratio")

    if len(sub) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(sub))
    width = 0.6

    ax.bar(x, sub["reliability"], width, label="Reliability", color="#ff453a", alpha=0.8)
    ax.bar(x, -sub["resolution"], width, bottom=sub["reliability"],
           label="−Resolution", color="#30d158", alpha=0.8)
    ax.bar(x, sub["uncertainty"], width,
           bottom=sub["reliability"] - sub["resolution"],
           label="Uncertainty", color="#2997ff", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}" for v in sub["w_ratio"]])
    ax.set_xlabel("$w_A / w_C$")
    ax.set_ylabel("Score component")
    ax.set_title(f"Brier decomposition ($N=32$, $r_{{\\min}} = {r_min}$)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()

    _ensure_dir(output_dir)
    fig.savefig(output_dir / f"brier_decomp_r{r_min}.pdf")
    fig.savefig(output_dir / f"brier_decomp_r{r_min}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Effect size forest plot
# ---------------------------------------------------------------------------

def plot_effect_size_forest(
    effect_df: pd.DataFrame,
    output_dir: pathlib.Path,
    w_ratio: float = 1.0,
):
    """Forest plot of Cohen's d for BS(N) − BS(1) at a given w_ratio."""
    sub = effect_df[effect_df["w_ratio"] == w_ratio].sort_values("N")

    if len(sub) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 3.5))
    y_pos = np.arange(len(sub))

    for i, (_, row) in enumerate(sub.iterrows()):
        color = "#ff453a" if row["cohens_d"] > 0 else "#30d158"
        ci_lo = row.get("ci_lo", row["mean_diff"] - 0.01)
        ci_hi = row.get("ci_hi", row["mean_diff"] + 0.01)
        ax.errorbar(row["mean_diff"], i, xerr=[[row["mean_diff"] - ci_lo],
                     [ci_hi - row["mean_diff"]]],
                     fmt="o", color=color, capsize=4, markersize=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"N={int(row['N'])}" for _, row in sub.iterrows()])
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Mean Brier difference (BS$_N$ − BS$_1$)")
    ax.set_title(f"Effect sizes: Brier degradation ($w_A/w_C = {w_ratio}$)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()

    _ensure_dir(output_dir)
    fig.savefig(output_dir / f"forest_w{w_ratio}.pdf")
    fig.savefig(output_dir / f"forest_w{w_ratio}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Master generator
# ---------------------------------------------------------------------------

def generate_all_figures(
    df: pd.DataFrame,
    output_dir: pathlib.Path,
    inflation_df: pd.DataFrame | None = None,
    decomp_df: pd.DataFrame | None = None,
    effect_df: pd.DataFrame | None = None,
):
    """Generate all standard figures."""
    _ensure_dir(output_dir)

    for r_min in [0.5, 0.7, 0.9]:
        plot_brier_vs_n(df, output_dir, r_min=r_min)
        plot_pareto_frontier(df, output_dir, r_min=r_min)
        for w_ratio in [0.5, 1.0, 4.0]:
            plot_confidence_histograms(df, output_dir, r_min=r_min, w_ratio=w_ratio)

    # Additional figures if data available
    if inflation_df is not None:
        for r_min in [0.5, 0.7, 0.9]:
            plot_inflation_heatmap(inflation_df, output_dir, r_min=r_min)

    if decomp_df is not None:
        for r_min in [0.5, 0.7, 0.9]:
            plot_brier_decomposition(decomp_df, output_dir, r_min=r_min)

    if effect_df is not None:
        for w_ratio in [0.0, 0.5, 1.0, 2.0, 4.0]:
            plot_effect_size_forest(effect_df, output_dir, w_ratio=w_ratio)
