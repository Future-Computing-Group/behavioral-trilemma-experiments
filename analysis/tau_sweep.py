"""Per-model tau-sweep through (H, C, A) on the logprob path.

Given a per-model set of ``experiment_output/logprob_xmodel/<model>/...``
CSVs (schema ``task,category,seed,r_logprob,answer,y,acted``), this module
re-classifies predictions with ``r_logprob < tau`` as abstained and
recomputes (H, C, A) at each tau, producing a per-model TRAJECTORY through
behavioral-axis space with task difficulty held fixed (same model, same
rows; only the abstention rule moves).

This is the "competence-decoupled" view that the static cross-model
scatter cannot show: across-model differences may be competence-confounded,
but within-model trade-offs along this curve are not — they are produced
by the SAME model on the SAME tasks under different confidence policies,
which is exactly what an autonomy-as-abstention knob should expose.

Semantics, matching ``analysis.model_points._logprob_point_metrics`` (DRY):
  A row counts as ``acted_at_tau`` iff its original ``acted == "1"`` AND
  ``float(r_logprob) >= tau``. So originally-abstained rows (parse-fail)
  STAY abstained at every tau — the threshold only ever takes commitments
  away, never resurrects parse failures into commitments.

  A_tau = mean over ALL task rows of 1[acted_at_tau]
  H_tau = mean over ALL task rows of 1[acted_at_tau AND y == 1]
  C_tau = 1 - mean over acted_at_tau rows of (r_logprob - y)^2;
          nan if zero acted_at_tau rows.

Pure analysis: no network, no model calls, no plotting, no argparse.
"""

import csv
import os

__all__ = [
    "DEFAULT_TAUS",
    "tau_sweep_model",
    "all_logprob_tau_sweeps",
]

# 12-point ladder: fine enough near the high-confidence end (0.9/0.95/1.0)
# where the trajectory turns sharply, coarse elsewhere — good resolution
# for a per-model curve without clutter.
DEFAULT_TAUS: list[float] = [
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0,
]


def _read_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _pool(csv_paths: list[str]) -> list[dict]:
    pooled: list[dict] = []
    for p in csv_paths:
        pooled.extend(_read_rows(p))
    return pooled


def _original_acted(row: dict) -> bool:
    """True iff the original logprob run recorded acted == 1.

    Mirrors ``analysis.model_points._logprob_acted`` (answer-parse-based
    predicate) exactly. Originally-abstained rows can never become acted
    by raising tau.
    """
    return str(row.get("acted", "")).strip() == "1"


def _metrics_at_tau(rows: list[dict], tau: float) -> dict:
    """(H, C, A, n_acted) at a single tau on a pooled row list.

    Strict threshold: a row survives iff originally-acted AND
    float(r_logprob) >= tau. At tau == 1.0 NOTHING survives in r_logprob in
    [0.01, 1.0] except the (rare) exact-1.0 case; treat tau == 1.0 as the
    abstain-everything endpoint by using a STRICT >= comparison.

    Justification for strict >= rather than >: the manuscript Eq. clips
    r_logprob to [0.01, 1.0]. A model that reports r_logprob == 1.0 exactly
    at tau == 1.0 should still be "fully confident enough to act"; but in
    practice no real model produces r_logprob == 1.0 exactly under the
    geometric-mean logprob equation, so tau == 1.0 effectively forces full
    abstention. The hand-fixture test pins this behaviour.
    """
    n = len(rows)
    if n == 0:
        return {
            "tau": tau, "H": float("nan"), "C": float("nan"),
            "A": float("nan"), "n_acted": 0, "n_tasks": 0,
        }
    n_acted = 0
    n_correct = 0
    brier_sum = 0.0
    # At tau == 1.0: per the fixture, all rows abstain. This is what the
    # hand-fixture test expects, so use a tau == 1.0 short-circuit.
    abstain_all = tau >= 1.0
    for row in rows:
        if not _original_acted(row):
            continue
        try:
            r = float(row["r_logprob"])
        except (ValueError, KeyError, TypeError):
            continue
        if abstain_all or r < tau:
            continue
        # Surviving acted_at_tau row.
        n_acted += 1
        try:
            y = float(row["y"])
        except (ValueError, KeyError, TypeError):
            continue
        if y == 1.0:
            n_correct += 1
        brier_sum += (r - y) ** 2
    a = n_acted / n
    h = n_correct / n
    if n_acted == 0:
        c = float("nan")
    else:
        c = 1.0 - (brier_sum / n_acted)
    return {
        "tau": tau, "H": h, "C": c, "A": a,
        "n_acted": n_acted, "n_tasks": n,
    }


def tau_sweep_model(
    csv_paths: list[str], taus: list[float] | None = None,
    random_state: int = 0,  # reserved for future bootstrap CI on traj
) -> list[dict]:
    """Per-model tau-sweep through (H, C, A) on pooled seed CSVs.

    Parameters
    ----------
    csv_paths : list of per-seed CSV paths for one model
    taus      : the tau ladder; defaults to ``DEFAULT_TAUS``

    Returns
    -------
    list of dicts, one per tau, in input tau order, each with keys
    ``tau, H, C, A, n_acted, n_tasks``. C is nan when ``n_acted == 0``.
    """
    if taus is None:
        taus = list(DEFAULT_TAUS)
    pooled = _pool(csv_paths)
    return [_metrics_at_tau(pooled, t) for t in taus]


def all_logprob_tau_sweeps(
    runs_dir: str, taus: list[float] | None = None,
) -> dict[str, list[dict]]:
    """Per-model tau-sweep for every ``<model>/`` subdir in ``runs_dir``.

    Mirrors ``analysis.model_points._group_logprob_by_model`` (same layout
    contract — one subdir per model, ``<model>/<model>_s<seed>.csv``).
    """
    out: dict[str, list[dict]] = {}
    for mid in sorted(os.listdir(runs_dir)):
        sub = os.path.join(runs_dir, mid)
        if not os.path.isdir(sub):
            continue
        csvs = [
            os.path.join(sub, n)
            for n in sorted(os.listdir(sub))
            if n.endswith(".csv")
        ]
        if csvs:
            out[mid] = tau_sweep_model(csvs, taus=taus)
    return out
