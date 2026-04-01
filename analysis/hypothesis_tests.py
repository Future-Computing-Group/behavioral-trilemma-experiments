"""Hypothesis tests H1-H6 for the behavioral trilemma experiment."""
import numpy as np
import pandas as pd
from scipy import stats


def bonferroni_holm(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Apply Bonferroni-Holm step-down correction to a list of p-values.

    Returns list of dicts with original_p, adjusted_p, rejected, rank.
    Adjusted p-values are monotonized (non-decreasing in rank order).
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [None] * m

    max_adj_p = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (m - rank)
        adjusted_p = min(p * (m - rank), 1.0)
        # Enforce monotonicity: adjusted p must not decrease with rank
        adjusted_p = max(adjusted_p, max_adj_p)
        max_adj_p = adjusted_p
        results[orig_idx] = {
            "rank": rank + 1,
            "original_p": p,
            "adjusted_p": adjusted_p,
            "rejected": adjusted_p <= alpha,
        }
    return results


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def bootstrap_ci(data: np.ndarray, statistic=np.mean, n_boot: int = 10000,
                 ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Bootstrap confidence interval for a statistic."""
    rng = np.random.RandomState(seed)
    boot_stats = [statistic(rng.choice(data, size=len(data), replace=True))
                  for _ in range(n_boot)]
    lo = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def test_h1_fkg_degradation(df: pd.DataFrame, w_ratio_min: float = 0.25) -> dict:
    """H1: BS(N=32) > BS(N=1) for w_A > 0.

    One-sided paired t-test across tasks, averaging over seeds.
    """
    sub = df[df["w_ratio"] >= w_ratio_min]
    base = sub[sub["N"] == 1].groupby("task_id")["brier"].mean()
    high = sub[sub["N"] == 32].groupby("task_id")["brier"].mean()
    common = base.index.intersection(high.index)
    if len(common) < 2:
        return {"statistic": np.nan, "p_value": 1.0, "effect_size": 0.0, "n": 0}

    base_vals = base.loc[common].values
    high_vals = high.loc[common].values
    t_stat, p_two = stats.ttest_rel(high_vals, base_vals)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    d = cohens_d(high_vals, base_vals)
    return {"statistic": float(t_stat), "p_value": float(p_one),
            "effect_size": float(d), "n": len(common)}


def test_h2_inflation_scaling(df: pd.DataFrame, binding_tasks: set[str] | None = None,
                              p_hat: dict[str, float] | None = None) -> dict:
    """H2: Inflation Δ increases with w_A/w_C.

    Linear regression of mean inflation (r - p_hat) on w_ratio for N=32.
    """
    sub = df[(df["N"] == 32) & (df["w_ratio"] > 0)].copy()
    if binding_tasks:
        sub = sub[sub["task_id"].isin(binding_tasks)]
    if len(sub) < 2 or not p_hat:
        return {"slope": np.nan, "p_value": 1.0, "r_squared": 0.0}

    sub["r_selected"] = pd.to_numeric(sub["r_selected"], errors="coerce")
    sub["p_hat"] = sub["task_id"].map(p_hat)
    sub["inflation"] = sub["r_selected"] - sub["p_hat"]
    sub = sub.dropna(subset=["inflation"])

    agg = sub.groupby("w_ratio")["inflation"].mean()
    if len(agg) < 2:
        return {"slope": np.nan, "p_value": 1.0, "r_squared": 0.0}

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        agg.index.values, agg.values
    )
    return {"slope": float(slope), "p_value": float(p_value),
            "r_squared": float(r_value ** 2)}


def test_h3_pareto_convexity(df: pd.DataFrame) -> dict:
    """H3: (H, C, A) triples form a convex Pareto surface.

    For each triple of weight vectors where w2 is a convex combination of w1
    and w3, check whether the achieved triple dominates the convex combination.
    """
    agg = df.groupby(["w_ratio", "r_min"]).agg(
        H=("y", "mean"),
        C=("brier", lambda x: 1 - x.mean()),  # 1 - BS for "higher is better"
        A=("gate_cleared", "mean"),
    ).reset_index()

    violations = 0
    total_tests = 0

    for r_min in agg["r_min"].unique():
        sub = agg[agg["r_min"] == r_min].sort_values("w_ratio")
        for i in range(len(sub) - 2):
            for k in range(i + 2, len(sub)):
                for j in range(i + 1, k):
                    total_tests += 1
                    p1 = sub.iloc[i][["H", "C", "A"]].values.astype(float)
                    p2 = sub.iloc[j][["H", "C", "A"]].values.astype(float)
                    p3 = sub.iloc[k][["H", "C", "A"]].values.astype(float)
                    # p2 should be above or on the line from p1 to p3
                    w = (sub.iloc[j]["w_ratio"] - sub.iloc[i]["w_ratio"]) / \
                        (sub.iloc[k]["w_ratio"] - sub.iloc[i]["w_ratio"])
                    interpolated = (1 - w) * p1 + w * p3
                    # Violation: any dimension of p2 is below interpolated by > 5%
                    if any(p2 < interpolated - 0.05):
                        violations += 1

    violation_rate = violations / total_tests if total_tests > 0 else 0.0
    return {"violations": violations, "total_tests": total_tests,
            "violation_rate": float(violation_rate),
            "convex": violation_rate <= 0.05}


def test_h4_threshold_clustering(df: pd.DataFrame, binding_tasks: set[str] | None = None) -> dict:
    """H4: Confidence clusters at r_min under gating pressure.

    One-sided proportion test for excess mass in [r_min, r_min + 0.1].
    """
    results = {}
    for r_min in df["r_min"].unique():
        sub = df[df["r_min"] == r_min]
        if binding_tasks:
            sub = sub[sub["task_id"].isin(binding_tasks)]

        base = sub[(sub["N"] == 1)]
        treated = sub[(sub["N"] == 32) & (sub["w_ratio"] > 0)]

        if len(base) == 0 or len(treated) == 0:
            continue

        window = (r_min, r_min + 0.1)
        base_r = pd.to_numeric(base["r_selected"], errors="coerce").dropna()
        treat_r = pd.to_numeric(treated["r_selected"], errors="coerce").dropna()

        p_base = ((base_r >= window[0]) & (base_r <= window[1])).mean()
        p_treat = ((treat_r >= window[0]) & (treat_r <= window[1])).mean()

        n_base, n_treat = len(base_r), len(treat_r)
        if n_base > 0 and n_treat > 0:
            p_pool = (p_base * n_base + p_treat * n_treat) / (n_base + n_treat)
            if p_pool > 0 and p_pool < 1:
                se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_base + 1 / n_treat))
                z = (p_treat - p_base) / se
                p_value = 1 - stats.norm.cdf(z)
            else:
                z, p_value = 0.0, 1.0
        else:
            z, p_value = 0.0, 1.0

        results[r_min] = {
            "p_base": float(p_base), "p_treat": float(p_treat),
            "z": float(z), "p_value": float(p_value),
        }
    return results


def test_h5_binding_specificity(df: pd.DataFrame, binding_tasks: set[str],
                                 p_hat: dict[str, float]) -> dict:
    """H5: Inflation concentrates on binding states.

    Δ|bind > 2 × Δ|¬bind. One-sided t-test.
    """
    sub = df[(df["N"] == 32) & (df["w_ratio"] > 0)].copy()
    sub["r_selected"] = pd.to_numeric(sub["r_selected"], errors="coerce")
    sub["p_hat"] = sub["task_id"].map(p_hat)
    sub["inflation"] = sub["r_selected"] - sub["p_hat"]
    sub["binding"] = sub["task_id"].isin(binding_tasks)

    bind = sub[sub["binding"]]["inflation"].dropna()
    non_bind = sub[~sub["binding"]]["inflation"].dropna()

    if len(bind) < 2 or len(non_bind) < 2:
        return {"mean_bind": np.nan, "mean_nonbind": np.nan, "ratio": np.nan,
                "p_value": 1.0}

    t_stat, p_two = stats.ttest_ind(bind, non_bind, alternative="greater")
    return {
        "mean_bind": float(bind.mean()),
        "mean_nonbind": float(non_bind.mean()),
        "ratio": float(bind.mean() / non_bind.mean()) if non_bind.mean() != 0 else float("inf"),
        "p_value": float(p_two),
    }


def test_h6_control_improves(df: pd.DataFrame) -> dict:
    """H6: When w_A=0, Best-of-N improves calibration (BS decreases with N).

    One-sided paired t-test: BS(N=32) < BS(N=1) when w_A=0.
    """
    sub = df[df["w_ratio"] == 0]
    base = sub[sub["N"] == 1].groupby("task_id")["brier"].mean()
    high = sub[sub["N"] == 32].groupby("task_id")["brier"].mean()
    common = base.index.intersection(high.index)
    if len(common) < 2:
        return {"statistic": np.nan, "p_value": 1.0, "effect_size": 0.0, "n": 0}

    base_vals = base.loc[common].values
    high_vals = high.loc[common].values
    t_stat, p_two = stats.ttest_rel(high_vals, base_vals)
    p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2  # one-sided: high < base
    d = cohens_d(base_vals, high_vals)  # reversed: improvement
    return {"statistic": float(t_stat), "p_value": float(p_one),
            "effect_size": float(d), "n": len(common)}


def run_all_tests(df: pd.DataFrame, binding_tasks: set[str] | None = None,
                  p_hat: dict[str, float] | None = None,
                  alpha: float = 0.05) -> dict:
    """Run all 6 hypothesis tests and apply Bonferroni-Holm correction."""
    results = {}
    results["H1"] = test_h1_fkg_degradation(df)
    results["H2"] = test_h2_inflation_scaling(df, binding_tasks, p_hat)
    results["H3"] = test_h3_pareto_convexity(df)
    results["H4"] = test_h4_threshold_clustering(df, binding_tasks)
    results["H5"] = test_h5_binding_specificity(df, binding_tasks or set(),
                                                  p_hat or {})
    results["H6"] = test_h6_control_improves(df)

    # Collect p-values for correction
    p_values = [
        results["H1"]["p_value"],
        results["H2"]["p_value"],
        0.0 if results["H3"].get("convex", False) else 1.0,  # binary
        min(v["p_value"] for v in results["H4"].values()) if results["H4"] else 1.0,
        results["H5"]["p_value"],
        results["H6"]["p_value"],
    ]
    correction = bonferroni_holm(p_values, alpha=alpha)
    results["correction"] = {
        f"H{i+1}": c for i, c in enumerate(correction)
    }
    return results
