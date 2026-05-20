"""Conditional helpfulness: factor deferral out of the H axis.

The unconditional H from ``analysis.model_points._logprob_point_metrics``
mixes two things: how often the model commits (deferral / action rate)
and how often a commitment is correct (competence given commitment). On
the standard cross-model scatter, a model that abstains more often shows
a lower H even if every commitment is correct — which conflates
calibration policy with competence.

This module provides the conditional H ("if you commit, are you right?")
plus a convenience triple ``(H_conditional, C, A)`` so plot code can
compose the three axes with deferral cleanly separated from helpfulness.

This is the third "competence-decoupled" view: the tau-sweep traces a
single model's competence-controlled trajectory; difficulty stratification
holds task hardness fixed; and H_conditional removes the deferral
component so the axis purely reflects "commitment-conditional competence".
"""

import csv
import math

from analysis.model_points import (
    _bootstrap_ci,
    _logprob_acted,
    _logprob_point_metrics,
    _logprob_y_is_one,
    _MIN_ROWS_PER_SEED,
    _REQUIRED_SEEDS,
)

__all__ = [
    "H_conditional",
    "decoupled_triple",
]


def _read_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _pool(csv_paths: list[str]) -> tuple[list[dict], list[int]]:
    pooled: list[dict] = []
    counts: list[int] = []
    for p in csv_paths:
        rows = _read_rows(p)
        counts.append(len(rows))
        pooled.extend(rows)
    return pooled, counts


def _h_cond_point_metrics(
    rows: list[dict],
) -> tuple[float, float, float, int]:
    """(H_cond, A, C, n_acted) shape-compatible with _logprob_point_metrics.

    H_cond replaces H here, so we can reuse the existing _bootstrap_ci
    engine via its point_fn hook. C and A definitions are unchanged.
    """
    n = len(rows)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    acted_flags = [_logprob_acted(r) for r in rows]
    n_acted = sum(acted_flags)
    a = n_acted / n
    if n_acted == 0:
        h_cond = float("nan")
        c = float("nan")
    else:
        n_correct = sum(
            1 for r, act in zip(rows, acted_flags)
            if act and _logprob_y_is_one(r)
        )
        h_cond = n_correct / n_acted
        briers = []
        for r, act in zip(rows, acted_flags):
            if not act:
                continue
            try:
                rv = float(r["r_logprob"])
                yv = float(r["y"])
            except (ValueError, KeyError, TypeError):
                continue
            briers.append((rv - yv) ** 2)
        c = 1.0 - (sum(briers) / len(briers)) if briers else float("nan")
    return h_cond, a, c, n_acted


def H_conditional(csv_paths: list[str], random_state: int = 0) -> dict:
    """Pooled H_conditional with bootstrap CI from the per-row data.

    Returns ``{H_cond, H_cond_ci, A, C, n_acted, n_tasks, partial}``. A is
    the unconditional action rate and C is the unconditional calibration
    (same definitions as ``_logprob_point_metrics``), so callers can
    compose the decoupled triple directly without re-reading the CSVs.
    """
    pooled, per_file_counts = _pool(csv_paths)
    n_seeds = len(csv_paths)
    h_cond, a, c, n_acted = _h_cond_point_metrics(pooled)
    # Reuse the existing percentile-bootstrap engine via point_fn hook
    # (the engine returns three CIs; here the "H slot" is the H_cond CI).
    h_ci, c_ci, a_ci = _bootstrap_ci(
        pooled, random_state, point_fn=_h_cond_point_metrics,
    )
    partial = (
        n_seeds < _REQUIRED_SEEDS
        or any(cnt < _MIN_ROWS_PER_SEED for cnt in per_file_counts)
    )
    return {
        "H_cond": h_cond,
        "H_cond_ci": h_ci,
        "A": a,
        "A_ci": a_ci,
        "C": c,
        "C_ci": c_ci,
        "n_acted": n_acted,
        "n_tasks": len(pooled),
        "n_seeds": n_seeds,
        "partial": partial,
    }


def decoupled_triple(csv_paths: list[str], random_state: int = 0) -> dict:
    """Convenience: the decoupled (H_conditional, C, A) triple for one model.

    Same shape as ``H_conditional`` plus the unconditional H for cross-ref.
    """
    out = H_conditional(csv_paths, random_state=random_state)
    pooled, _ = _pool(csv_paths)
    h_unc, _a, _c, _na = _logprob_point_metrics(pooled)
    out["H_unconditional"] = h_unc
    return out
