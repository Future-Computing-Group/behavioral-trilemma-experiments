from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def bonferroni_holm(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Apply Bonferroni-Holm step-down correction to a list of p-values."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [None] * m

    max_adj_p = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_p = min(p * (m - rank), 1.0)
        adjusted_p = max(adjusted_p, max_adj_p)
        max_adj_p = adjusted_p
        results[orig_idx] = {
            "rank": rank + 1,
            "original_p": float(p),
            "adjusted_p": float(adjusted_p),
            "rejected": bool(adjusted_p <= alpha),
        }
    return results

def _normalize_binding_map(
    binding_tasks: dict[float, set[str]] | set[str] | None,
) -> dict[float, set[str]]:
    """Normalize binding tasks input to dict[r_min -> set[task_id]]."""
    if binding_tasks is None:
        return {}
    if isinstance(binding_tasks, dict):
        out: dict[float, set[str]] = {}
        for key, val in binding_tasks.items():
            try:
                out[float(key)] = set(val)
            except Exception:
                continue
        return out
    return {0.7: set(binding_tasks)}


def _binding_for_rmin(binding_map: dict[float, set[str]], r_min: float) -> set[str]:
    """Return binding set for r_min, with numeric-tolerance matching."""
    if r_min in binding_map:
        return binding_map[r_min]
    for key in binding_map:
        if abs(float(key) - float(r_min)) < 1e-9:
            return binding_map[key]
    return set()


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce common numeric columns used by hypothesis tests."""
    out = df.copy()
    for col in ["N", "w_ratio", "r_min", "seed", "brier", "r_selected", "y", "gate_cleared"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size (group1 - group2) / pooled_sd."""
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def cohens_d_paired(diffs: np.ndarray) -> float:
    """Cohen's d for paired differences: mean(diff) / sd(diff)."""
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) < 2:
        return 0.0
    sd = diffs.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diffs.mean() / sd)


def bootstrap_ci(
    data: np.ndarray,
    statistic=np.mean,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a statistic."""
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return (np.nan, np.nan)
    rng = np.random.RandomState(seed)
    boot_stats = np.array([
        statistic(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_stats, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_stats, (1 + ci) / 2 * 100))
    return (lo, hi)


def bootstrap_ci_diff(
    group1: np.ndarray,
    group2: np.ndarray,
    statistic=np.mean,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for statistic(group1) - statistic(group2)."""
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)
    if len(group1) == 0 or len(group2) == 0:
        return (np.nan, np.nan)
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        s1 = rng.choice(group1, size=len(group1), replace=True)
        s2 = rng.choice(group2, size=len(group2), replace=True)
        diffs.append(statistic(s1) - statistic(s2))
    diffs = np.asarray(diffs, dtype=float)
    lo = float(np.percentile(diffs, (1 - ci) / 2 * 100))
    hi = float(np.percentile(diffs, (1 + ci) / 2 * 100))
    return (lo, hi)


# H1 (updated): fixed-axis gating degradation at N=32

def test_h1_fkg_degradation(
    df: pd.DataFrame,
    fixed_n: int = 32,
    w_ratio_control: float = 0.0,
    w_ratio_treated: float = 4.0,
    n_boot: int = 10000,
) -> dict:
    work = _coerce_numeric_columns(df)
    sub = work[work["N"] == fixed_n].copy()

    base = sub[sub["w_ratio"] == w_ratio_control].groupby("task_id")["brier"].mean()
    treated = sub[sub["w_ratio"] == w_ratio_treated].groupby("task_id")["brier"].mean()
    common = base.index.intersection(treated.index)

    if len(common) < 2:
        return {
            "statistic": np.nan,
            "p_value": 1.0,
            "effect_size": 0.0,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "mean_diff": np.nan,
            "percent_degradation": np.nan,
            "n": 0,
            "fixed_n": int(fixed_n),
            "w_ratio_control": float(w_ratio_control),
            "w_ratio_treated": float(w_ratio_treated),
        }

    base_vals = base.loc[common].values
    treated_vals = treated.loc[common].values
    diffs = treated_vals - base_vals

    t_stat, p_two = stats.ttest_rel(treated_vals, base_vals)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    d = cohens_d_paired(diffs)
    ci_lo, ci_hi = bootstrap_ci(diffs, statistic=np.mean, n_boot=n_boot)

    base_mean = float(base_vals.mean())
    percent = float((treated_vals.mean() - base_mean) / base_mean) if base_mean != 0 else np.nan

    return {
        "statistic": float(t_stat),
        "p_value": float(p_one),
        "effect_size": float(d),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "mean_diff": float(diffs.mean()),
        "percent_degradation": percent,
        "n": int(len(common)),
        "fixed_n": int(fixed_n),
        "w_ratio_control": float(w_ratio_control),
        "w_ratio_treated": float(w_ratio_treated),
    }


# H2 (updated): monotone inflation trend in w_A/w_C

def _jonckheere_terpstra_increasing(groups: list[np.ndarray]) -> dict:
    """Approximate Jonckheere-Terpstra trend test for ordered groups."""
    cleaned = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    if len(cleaned) < 2:
        return {"J": np.nan, "z": np.nan, "p_value": 1.0, "n_obs": 0, "n_groups": len(cleaned)}

    ns = [len(g) for g in cleaned]
    n_total = int(sum(ns))
    j_stat = 0.0
    all_vals = np.concatenate(cleaned)

    for i in range(len(cleaned) - 1):
        x = cleaned[i]
        for j in range(i + 1, len(cleaned)):
            y = cleaned[j]
            comp = y[:, None] - x[None, :]
            j_stat += np.sum(comp > 0) + 0.5 * np.sum(comp == 0)

    mu = (n_total**2 - sum(n_i**2 for n_i in ns)) / 4.0

    _, counts = np.unique(all_vals, return_counts=True)
    tie_sum = np.sum(counts**3 - counts)
    term1 = n_total * (n_total - 1) * (2 * n_total + 5)
    term2 = sum(n_i * (n_i - 1) * (2 * n_i + 5) for n_i in ns)
    var = (term1 - term2 - tie_sum) / 72.0

    if var <= 0:
        z = np.nan
        p_value = 1.0
    else:
        z = (j_stat - mu) / np.sqrt(var)
        p_value = float(1 - stats.norm.cdf(z))

    return {
        "J": float(j_stat),
        "z": float(z) if np.isfinite(z) else np.nan,
        "p_value": float(p_value),
        "n_obs": n_total,
        "n_groups": len(cleaned),
    }


def test_h2_inflation_scaling(
    df: pd.DataFrame,
    binding_tasks: dict[float, set[str]] | set[str] | None = None,
    p_hat: dict[str, float] | None = None,
    fixed_n: int = 32,
    r_min: float = 0.7,
    include_control: bool = False,
    n_boot: int = 10000,  # kept for API compatibility
) -> dict:
    """H2 (updated): monotone non-decreasing inflation with increasing w_A/w_C."""
    if not p_hat:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r_squared": 0.0,
            "jt_statistic": np.nan,
            "z": np.nan,
            "p_value": 1.0,
            "rho": np.nan,
            "rho_p_value": 1.0,
            "means_by_weight": {},
            "n_obs": 0,
            "n_groups": 0,
            "fixed_n": int(fixed_n),
            "r_min": float(r_min),
            "include_control": bool(include_control),
        }

    binding_map = _normalize_binding_map(binding_tasks)
    bind_set = _binding_for_rmin(binding_map, r_min)
    if not bind_set:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r_squared": 0.0,
            "jt_statistic": np.nan,
            "z": np.nan,
            "p_value": 1.0,
            "rho": np.nan,
            "rho_p_value": 1.0,
            "means_by_weight": {},
            "n_obs": 0,
            "n_groups": 0,
            "fixed_n": int(fixed_n),
            "r_min": float(r_min),
            "include_control": bool(include_control),
        }

    work = _coerce_numeric_columns(df)
    sub = work[(work["N"] == fixed_n) & (work["r_min"] == r_min)].copy()
    sub = sub[sub["task_id"].isin(bind_set)]
    sub["r_selected"] = pd.to_numeric(sub["r_selected"], errors="coerce")
    sub["p_hat"] = sub["task_id"].map(p_hat)
    sub["inflation"] = sub["r_selected"] - sub["p_hat"]
    sub = sub.dropna(subset=["inflation"])

    agg = (
        sub.groupby(["w_ratio", "seed"], as_index=False)["inflation"]
        .mean()
        .sort_values(["w_ratio", "seed"])
    )
    if not include_control:
        agg = agg[agg["w_ratio"] > 0]

    if len(agg) == 0:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r_squared": 0.0,
            "jt_statistic": np.nan,
            "z": np.nan,
            "p_value": 1.0,
            "rho": np.nan,
            "rho_p_value": 1.0,
            "means_by_weight": {},
            "n_obs": 0,
            "n_groups": 0,
            "fixed_n": int(fixed_n),
            "r_min": float(r_min),
            "include_control": bool(include_control),
        }

    grouped = [g["inflation"].values for _, g in agg.groupby("w_ratio")]
    jt = _jonckheere_terpstra_increasing(grouped)

    means = (
        agg.groupby("w_ratio", as_index=False)["inflation"]
        .mean()
        .sort_values("w_ratio")
    )

    if len(means) < 2:
        slope, intercept, r_value, rho, rho_p = np.nan, np.nan, np.nan, np.nan, 1.0
    else:
        slope, intercept, r_value, _, _ = stats.linregress(
            means["w_ratio"], means["inflation"]
        )
        rho, rho_p = stats.spearmanr(means["w_ratio"], means["inflation"])

    return {
        "slope": float(slope) if np.isfinite(slope) else np.nan,
        "intercept": float(intercept) if np.isfinite(intercept) else np.nan,
        "r_squared": float(r_value**2) if np.isfinite(r_value) else 0.0,
        "jt_statistic": jt["J"],
        "z": jt["z"],
        "p_value": jt["p_value"],
        "rho": float(rho) if np.isfinite(rho) else np.nan,
        "rho_p_value": float(rho_p),
        "means_by_weight": {
            str(float(row.w_ratio)): float(row.inflation)
            for row in means.itertuples(index=False)
        },
        "n_obs": int(jt["n_obs"]),
        "n_groups": int(jt["n_groups"]),
        "fixed_n": int(fixed_n),
        "r_min": float(r_min),
        "include_control": bool(include_control),
    }

# H3 (updated): tolerance-aware convexity

def _compute_h3_convexity_raw(df: pd.DataFrame, axis_slack: float = 0.05) -> dict:
    """Compute midpoint-convexity violations with per-axis slack."""
    work = _coerce_numeric_columns(df)

    agg = work.groupby(["w_ratio", "r_min"]).agg(
        H=("y", "mean"),
        C=("brier", lambda x: 1 - x.mean()),
        A=("gate_cleared", "mean"),
    ).reset_index()

    violations = 0
    total_tests = 0

    for rmin_val in agg["r_min"].unique():
        sub = agg[agg["r_min"] == rmin_val].sort_values("w_ratio")
        n_points = len(sub)
        for i in range(n_points - 2):
            for k in range(i + 2, n_points):
                for j in range(i + 1, k):
                    total_tests += 1
                    p1 = sub.iloc[i][["H", "C", "A"]].values.astype(float)
                    p2 = sub.iloc[j][["H", "C", "A"]].values.astype(float)
                    p3 = sub.iloc[k][["H", "C", "A"]].values.astype(float)

                    w_denom = sub.iloc[k]["w_ratio"] - sub.iloc[i]["w_ratio"]
                    if w_denom == 0:
                        continue
                    w = (sub.iloc[j]["w_ratio"] - sub.iloc[i]["w_ratio"]) / w_denom
                    interpolated = (1 - w) * p1 + w * p3

                    if np.any(p2 < interpolated - axis_slack):
                        violations += 1

    rate = violations / total_tests if total_tests > 0 else np.nan
    return {
        "violations": int(violations),
        "total_tests": int(total_tests),
        "violation_rate": float(rate) if np.isfinite(rate) else np.nan,
    }


def test_h3_pareto_convexity(
    df: pd.DataFrame,
    tolerance: float = 0.15,
    axis_slack: float = 0.05,
) -> dict:
    """H3 (updated): violation rate below tolerance under finite-sample slack."""
    raw = _compute_h3_convexity_raw(df, axis_slack=axis_slack)
    violations = raw["violations"]
    total_tests = raw["total_tests"]

    if total_tests <= 0:
        return {
            "violations": violations,
            "total_tests": total_tests,
            "violation_rate": np.nan,
            "tolerance": float(tolerance),
            "axis_slack": float(axis_slack),
            "criterion_met": False,
            "convex": False,
            "p_value": 1.0,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "statistically_supported": False,
        }

    rate = violations / total_tests
    binom = stats.binomtest(violations, total_tests, p=tolerance, alternative="less")
    ci = binom.proportion_ci(confidence_level=0.95, method="exact")
    criterion_met = bool(rate < tolerance)

    return {
        "violations": int(violations),
        "total_tests": int(total_tests),
        "violation_rate": float(rate),
        "tolerance": float(tolerance),
        "axis_slack": float(axis_slack),
        "criterion_met": criterion_met,
        "convex": criterion_met,
        "p_value": float(binom.pvalue),
        "ci_lo": float(ci.low),
        "ci_hi": float(ci.high),
        "statistically_supported": bool(ci.high < tolerance),
    }


# H4 : threshold clustering

def test_h4_threshold_clustering(
    df: pd.DataFrame,
    binding_tasks: dict[float, set[str]] | set[str] | None = None,
    n_boot: int = 10000,
) -> dict:
    """H4: excess mass in [r_min, r_min + 0.1] under gating."""
    work = _coerce_numeric_columns(df)
    binding_map = _normalize_binding_map(binding_tasks)
    results: dict = {}

    rng = np.random.RandomState(42)

    for r_min in sorted(work["r_min"].unique()):
        sub = work[work["r_min"] == r_min]

        bind_set = _binding_for_rmin(binding_map, r_min)
        if bind_set:
            sub = sub[sub["task_id"].isin(bind_set)]

        base = sub[sub["N"] == 1]
        treated = sub[(sub["N"] == 32) & (sub["w_ratio"] > 0)]

        if len(base) == 0 or len(treated) == 0:
            continue

        window = (float(r_min), float(r_min) + 0.1)
        base_r = pd.to_numeric(base["r_selected"], errors="coerce").dropna()
        treat_r = pd.to_numeric(treated["r_selected"], errors="coerce").dropna()

        p_base = float(((base_r >= window[0]) & (base_r <= window[1])).mean())
        p_treat = float(((treat_r >= window[0]) & (treat_r <= window[1])).mean())

        n_base, n_treat = len(base_r), len(treat_r)
        if n_base > 0 and n_treat > 0:
            p_pool = (p_base * n_base + p_treat * n_treat) / (n_base + n_treat)
            if 0 < p_pool < 1:
                se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_base + 1 / n_treat))
                z = (p_treat - p_base) / se
                p_value = float(1 - stats.norm.cdf(z))
            else:
                z, p_value = 0.0, 1.0
        else:
            z, p_value = 0.0, 1.0

        groups = {tid: g for tid, g in sub.groupby("task_id")}
        task_ids = list(groups.keys())
        boot_diffs = []
        if task_ids:
            for _ in range(n_boot):
                boot_tids = rng.choice(task_ids, size=len(task_ids), replace=True)
                win_parts = []
                treat_parts = []
                base_parts = []
                for tid in boot_tids:
                    g = groups[tid]
                    win_parts.append((g["r_selected"] >= window[0]) & (g["r_selected"] <= window[1]))
                    treat_parts.append((g["N"] == 32) & (g["w_ratio"] > 0))
                    base_parts.append(g["N"] == 1)

                win_arr = np.concatenate(win_parts)
                treat_arr = np.concatenate(treat_parts)
                base_arr = np.concatenate(base_parts)

                p_b = win_arr[base_arr].mean() if base_arr.any() else np.nan
                p_t = win_arr[treat_arr].mean() if treat_arr.any() else np.nan
                if not np.isnan(p_b) and not np.isnan(p_t):
                    boot_diffs.append(float(p_t - p_b))

        if boot_diffs:
            ci_lo = float(np.percentile(boot_diffs, 2.5))
            ci_hi = float(np.percentile(boot_diffs, 97.5))
        else:
            ci_lo, ci_hi = np.nan, np.nan

        diff = float(p_treat - p_base)
        results[r_min] = {
            "p_base": p_base,
            "p_treat": p_treat,
            "diff": diff,
            "z": float(z),
            "p_value": float(p_value),
            "n_base": int(n_base),
            "n_treat": int(n_treat),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "effect_size": diff,
        }

    if results:
        best = results.get(0.7, max(results.values(), key=lambda x: x.get("z", -np.inf)))
        results["z"] = float(best["z"])
        results["p_value"] = float(best["p_value"])
        results["ci_lo"] = float(best["ci_lo"])
        results["ci_hi"] = float(best["ci_hi"])
        results["effect_size"] = float(best["effect_size"])

    return results


# H5: binding-state specificity

def test_h5_binding_specificity(
    df: pd.DataFrame,
    binding_tasks: set[str],
    p_hat: dict[str, float],
    n_boot: int = 10000,
) -> dict:
    """H5: inflation is much larger on binding vs non-binding tasks."""
    if not p_hat:
        return {
            "mean_bind": np.nan,
            "mean_nonbind": np.nan,
            "ratio": np.nan,
            "statistic": np.nan,
            "p_value": 1.0,
            "effect_size": 0.0,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "n_bind": 0,
            "n_nonbind": 0,
        }

    work = _coerce_numeric_columns(df)
    sub = work[(work["N"] == 32) & (work["w_ratio"] > 0)].copy()
    sub["p_hat"] = sub["task_id"].map(p_hat)
    sub["inflation"] = sub["r_selected"] - sub["p_hat"]
    sub["binding"] = sub["task_id"].isin(binding_tasks)

    bind = sub[sub["binding"]]["inflation"].dropna().values
    non_bind = sub[~sub["binding"]]["inflation"].dropna().values

    if len(bind) < 2 or len(non_bind) < 2:
        return {
            "mean_bind": np.nan,
            "mean_nonbind": np.nan,
            "ratio": np.nan,
            "statistic": np.nan,
            "p_value": 1.0,
            "effect_size": 0.0,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "n_bind": len(bind),
            "n_nonbind": len(non_bind),
        }

    t_stat, p_val = stats.ttest_ind(bind, non_bind, alternative="greater")
    d = cohens_d(bind, non_bind)
    ci_lo, ci_hi = bootstrap_ci_diff(bind, non_bind, n_boot=n_boot)

    ratio = float(bind.mean() / non_bind.mean()) if non_bind.mean() != 0 else float("inf")

    return {
        "mean_bind": float(bind.mean()),
        "mean_nonbind": float(non_bind.mean()),
        "ratio": ratio,
        "statistic": float(t_stat),
        "p_value": float(p_val),
        "effect_size": float(d),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_bind": int(len(bind)),
        "n_nonbind": int(len(non_bind)),
    }


# H6: control improves with N

def test_h6_control_improves(
    df: pd.DataFrame,
    n_boot: int = 10000,
) -> dict:
    """H6: at w_A=0, calibration improves (BS[N=32] < BS[N=1])."""
    work = _coerce_numeric_columns(df)
    sub = work[work["w_ratio"] == 0].copy()

    base = sub[sub["N"] == 1].groupby("task_id")["brier"].mean()
    high = sub[sub["N"] == 32].groupby("task_id")["brier"].mean()
    common = base.index.intersection(high.index)

    if len(common) < 2:
        return {
            "statistic": np.nan,
            "p_value": 1.0,
            "effect_size": 0.0,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "mean_diff": np.nan,
            "n": 0,
        }

    base_vals = base.loc[common].values
    high_vals = high.loc[common].values
    diffs = high_vals - base_vals

    t_stat, p_two = stats.ttest_rel(high_vals, base_vals)
    p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
    d = cohens_d_paired(-diffs)
    ci_lo, ci_hi = bootstrap_ci(diffs, statistic=np.mean, n_boot=n_boot)

    return {
        "statistic": float(t_stat),
        "p_value": float(p_one),
        "effect_size": float(d),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "mean_diff": float(diffs.mean()),
        "n": int(len(common)),
    }


# Effect size table helper

def compute_pairwise_effect_sizes(
    df: pd.DataFrame,
    n_boot: int = 10000,
) -> pd.DataFrame:
    """Compute Cohen's d and CI for BS(N)-BS(N=1) at each w_ratio."""
    work = _coerce_numeric_columns(df)
    n_values = sorted(work["N"].unique())
    w_ratios = sorted(work["w_ratio"].unique())

    rows = []
    for w in w_ratios:
        w_df = work[work["w_ratio"] == w]
        base = w_df[w_df["N"] == 1].groupby("task_id")["brier"].mean()

        for n_val in n_values:
            if n_val == 1:
                continue
            high = w_df[w_df["N"] == n_val].groupby("task_id")["brier"].mean()
            common = base.index.intersection(high.index)
            if len(common) < 2:
                continue

            base_vals = base.loc[common].values
            high_vals = high.loc[common].values
            diffs = high_vals - base_vals
            d = cohens_d_paired(diffs)
            ci_lo, ci_hi = bootstrap_ci(diffs, statistic=np.mean, n_boot=n_boot)

            rows.append(
                {
                    "N": n_val,
                    "w_ratio": float(w),
                    "cohens_d": float(d),
                    "mean_diff": float(diffs.mean()),
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "n_tasks": int(len(common)),
                }
            )

    return pd.DataFrame(rows)


def run_all_tests(
    df: pd.DataFrame,
    binding_tasks: dict[float, set[str]] | set[str] | None = None,
    p_hat: dict[str, float] | None = None,
    alpha: float = 0.05,
    n_boot: int = 10000,
) -> dict:
    work = _coerce_numeric_columns(df)
    binding_map = _normalize_binding_map(binding_tasks)
    binding_07 = _binding_for_rmin(binding_map, 0.7)

    results: dict = {}

    results["H1"] = test_h1_fkg_degradation(
        work,
        fixed_n=32,
        w_ratio_control=0.0,
        w_ratio_treated=4.0,
        n_boot=n_boot,
    )
    results["H2"] = test_h2_inflation_scaling(
        work,
        binding_tasks=binding_map,
        p_hat=p_hat,
        fixed_n=32,
        r_min=0.7,
        include_control=False,
    )
    results["H3"] = test_h3_pareto_convexity(work, tolerance=0.15, axis_slack=0.05)
    results["H4"] = test_h4_threshold_clustering(work, binding_tasks=binding_map, n_boot=n_boot)
    results["H5"] = test_h5_binding_specificity(
        work,
        binding_tasks=binding_07,
        p_hat=p_hat or {},
        n_boot=n_boot,
    )
    results["H6"] = test_h6_control_improves(work, n_boot=n_boot)

    h4_p = min(
        (v["p_value"] for k, v in results["H4"].items() if isinstance(k, (int, float))),
        default=1.0,
    )

    p_values = [
        results["H1"]["p_value"],
        results["H2"]["p_value"],
        0.0 if results["H3"].get("criterion_met", False) else 1.0,
        h4_p,
        results["H5"]["p_value"],
        results["H6"]["p_value"],
    ]
    correction = bonferroni_holm(p_values, alpha=alpha)
    results["correction"] = {f"H{i+1}": c for i, c in enumerate(correction)}

    return results
