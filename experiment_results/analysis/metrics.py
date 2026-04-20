"""Aggregate metrics computation for the behavioral trilemma experiment."""
import pathlib
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: pathlib.Path) -> pd.DataFrame:
    """Load all result CSVs from a directory into a single DataFrame.

    Handles multiline answer fields (e.g. code blocks with embedded newlines)
    by using the Python CSV engine with proper quoting.
    """
    results_dir = pathlib.Path(results_dir)
    frames = []
    skipped = []
    for csv_file in sorted(results_dir.glob("*.csv")):
        if csv_file.name == "phase0_calibration.csv":
            continue
        if csv_file.suffix != ".csv":
            continue
        try:
            df = pd.read_csv(
                csv_file,
                engine="python",
                on_bad_lines="warn",
                quoting=1,  # csv.QUOTE_ALL — tolerant of embedded newlines
            )
            if len(df) > 0:
                frames.append(df)
        except Exception as e:
            skipped.append(csv_file.name)
            warnings.warn(f"Failed to load {csv_file.name}: {e}")
            continue

    if skipped:
        warnings.warn(f"Skipped {len(skipped)} files: {skipped[:5]}...")
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)

    # Coerce numeric columns that may have been read as strings
    numeric_cols = ["N", "w_ratio", "w_C", "w_A", "r_min", "seed",
                    "r_selected", "y", "V_selected", "brier", "gate_cleared",
                    "n_candidates", "n_parsed", "selected_index"]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    return combined


def load_phase0(phase0_path: pathlib.Path) -> tuple[dict[str, float], dict[float, set[str]]]:
    """Load Phase 0 calibration data.

    Returns
    -------
    p_hat : dict[str, float]
        Mapping task_id → estimated base accuracy.
    binding_sets : dict[float, set[str]]
        Mapping r_min threshold → set of binding task_ids.
    """
    phase0_path = pathlib.Path(phase0_path)
    df = pd.read_csv(phase0_path)

    p_hat = dict(zip(df["task_id"], df["p_hat"].astype(float)))

    binding_sets = {}
    for col in df.columns:
        if col.startswith("binding_"):
            r_min = float(col.replace("binding_", ""))
            binding_ids = set(df.loc[df[col] == 1, "task_id"])
            binding_sets[r_min] = binding_ids

    return p_hat, binding_sets


# ---------------------------------------------------------------------------
# Per-config metrics
# ---------------------------------------------------------------------------

def compute_per_config_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate metrics per configuration (N, w_ratio, r_min, seed).

    Returns a DataFrame with one row per config.
    """
    group_cols = ["N", "w_ratio", "r_min", "seed"]

    # Ensure numeric types
    for col in ["r_selected", "y", "brier", "gate_cleared"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    agg = df.groupby(group_cols, as_index=False).agg(
        mean_brier=("brier", "mean"),
        mean_H=("y", "mean"),
        mean_A=("gate_cleared", "mean"),
        mean_r=("r_selected", "mean"),
        n_tasks=("task_id", "count"),
    )
    return agg


def compute_inflation_metrics(
    df: pd.DataFrame,
    p_hat: dict[str, float],
    binding_sets: dict[float, set[str]],
) -> pd.DataFrame:
    """Compute inflation (Δ = r - p_hat) per config, on binding tasks only.

    Returns a DataFrame with columns:
        N, w_ratio, r_min, seed, mean_inflation, mean_inflation_binding,
        mean_inflation_nonbinding, n_binding, n_nonbinding
    """
    df = df.copy()
    df["r_selected"] = pd.to_numeric(df["r_selected"], errors="coerce")
    df["p_hat"] = df["task_id"].map(p_hat)
    df["inflation"] = df["r_selected"] - df["p_hat"]
    df = df.dropna(subset=["inflation"])

    group_cols = ["N", "w_ratio", "r_min", "seed"]
    rows = []

    for keys, grp in df.groupby(group_cols):
        r_min_val = keys[2]
        binding = binding_sets.get(r_min_val, set())

        mask_bind = grp["task_id"].isin(binding)
        bind_grp = grp[mask_bind]
        nonbind_grp = grp[~mask_bind]

        rows.append({
            "N": keys[0],
            "w_ratio": keys[1],
            "r_min": r_min_val,
            "seed": keys[3],
            "mean_inflation": grp["inflation"].mean(),
            "mean_inflation_binding": bind_grp["inflation"].mean() if len(bind_grp) > 0 else np.nan,
            "mean_inflation_nonbinding": nonbind_grp["inflation"].mean() if len(nonbind_grp) > 0 else np.nan,
            "n_binding": len(bind_grp),
            "n_nonbinding": len(nonbind_grp),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Brier decomposition
# ---------------------------------------------------------------------------

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


def compute_brier_decomposition_by_config(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Brier decomposition for each (N, w_ratio, r_min) group, averaged over seeds."""
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
            "brier_from_decomp": decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"],
            "brier_direct": grp_clean["brier"].mean(),
            "n_obs": len(grp_clean),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def compute_summary_table(
    df: pd.DataFrame,
    metric: str = "mean_brier",
    r_min: float = 0.7,
) -> pd.DataFrame:
    """Cross-tabulated summary: N (rows) × w_ratio (columns), averaged over seeds.

    Parameters
    ----------
    df : pd.DataFrame
        The per-config metrics DataFrame (from compute_per_config_metrics).
    metric : str
        Column to aggregate (e.g. "mean_brier", "mean_H", "mean_A").
    r_min : float
        Threshold to filter on.
    """
    sub = df[df["r_min"] == r_min]
    pivot = sub.pivot_table(
        values=metric,
        index="N",
        columns="w_ratio",
        aggfunc="mean",
    )
    return pivot
