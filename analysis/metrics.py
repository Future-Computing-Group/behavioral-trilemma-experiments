"""Aggregate metrics computation for the behavioral trilemma experiment."""
import pathlib
import warnings

import numpy as np
import pandas as pd


def load_results(results_dir: pathlib.Path) -> pd.DataFrame:
    """Load all result CSVs from a directory into a single DataFrame."""
    frames = []
    for csv_file in sorted(results_dir.glob("*.csv")):
        if csv_file.name == "phase0_calibration.csv":
            continue
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                frames.append(df)
        except Exception as e:
            warnings.warn(f"Failed to load {csv_file.name}: {e}")
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_phase0(phase0_path: pathlib.Path) -> tuple[dict[str, float], dict[float, set[str]]]:
    phase0_path = pathlib.Path(phase0_path)
    df = pd.read_csv(phase0_path)

    p_hat = dict(zip(df["task_id"], df["p_hat"].astype(float)))

    binding_sets: dict[float, set[str]] = {}
    for col in df.columns:
        if col.startswith("binding_"):
            r_min = float(col.replace("binding_", ""))
            binding_sets[r_min] = set(df.loc[df[col] == 1, "task_id"])

    return p_hat, binding_sets


def compute_per_config_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate metrics per configuration (N, w_ratio, r_min, seed).

    Returns a DataFrame with one row per config.
    """
    group_cols = ["N", "w_ratio", "r_min", "seed"]

    # Ensure numeric types
    for col in ["r_selected", "y", "brier", "gate_cleared"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    agg = df.groupby(group_cols, as_index=False).agg(
        mean_brier=("brier", "mean"),
        mean_H=("y", "mean"),
        mean_A=("gate_cleared", "mean"),
        mean_r=("r_selected", "mean"),
        n_tasks=("task_id", "count"),
    )
    return agg


def compute_brier_decomposition(
    r_values: list[float],
    y_values: list[int],
    n_bins: int = 10,
) -> dict[str, float]:
    """Brier decomposition into reliability, resolution, uncertainty.

    BS = reliability - resolution + uncertainty
    """
    r = np.array(r_values, dtype=float)
    y = np.array(y_values, dtype=float)
    K = len(r)

    if K == 0:
        return {"reliability": 0.0, "resolution": 0.0, "uncertainty": 0.0}

    o_bar = y.mean()
    uncertainty = o_bar * (1.0 - o_bar)

    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(r, bin_edges[1:-1])  # 0-based bin indices

    reliability = 0.0
    resolution = 0.0

    for b in range(n_bins):
        mask = bin_indices == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        r_bar_b = r[mask].mean()
        o_bar_b = y[mask].mean()
        reliability += n_b * (r_bar_b - o_bar_b) ** 2
        resolution += n_b * (o_bar_b - o_bar) ** 2

    reliability /= K
    resolution /= K

    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
    }


def compute_brier_decomposition_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """Brier decomposition for each (N, w_ratio, r_min) group, seeds pooled.

    Ported verbatim from the frozen snapshot
    ``experiment_output/analysis/metrics.py`` (audit LB-19) so the archived
    ``brier_decomposition.csv`` is regenerable from the maintained module:

        df = load_results(results_dir)
        compute_brier_decomposition_by_config(df).to_csv(..., index=False)

    Returns one row per group with reliability / resolution / uncertainty,
    the recomposed Brier (reliability - resolution + uncertainty), the
    direct mean Brier, and the observation count.
    """
    df = df.copy()
    df["r_selected"] = pd.to_numeric(df["r_selected"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    group_cols = ["N", "w_ratio", "r_min"]
    rows = []

    for keys, grp in df.groupby(group_cols):
        grp_clean = grp.dropna(subset=["r_selected", "y"])
        if len(grp_clean) == 0:
            continue
        decomp = compute_brier_decomposition(
            grp_clean["r_selected"].tolist(),
            grp_clean["y"].astype(int).tolist(),
        )
        rows.append({
            "N": keys[0],
            "w_ratio": keys[1],
            "r_min": keys[2],
            "reliability": decomp["reliability"],
            "resolution": decomp["resolution"],
            "uncertainty": decomp["uncertainty"],
            "brier_from_decomp": decomp["reliability"] - decomp["resolution"]
            + decomp["uncertainty"],
            "brier_direct": pd.to_numeric(
                grp_clean["brier"], errors="coerce").mean(),
            "n_obs": len(grp_clean),
        })

    return pd.DataFrame(rows)
