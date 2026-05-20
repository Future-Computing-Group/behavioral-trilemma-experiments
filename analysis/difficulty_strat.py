"""Per-task difficulty stratification of the logprob cross-model data.

Per-task difficulty = 1 - (cross-model mean correctness on committed rows
of that task). Harder task -> fewer models commit AND get it right. The
metric only counts committed rows: a task that 7 of 8 models abstain on is
not automatically "hard" — what matters is, *of the models that committed*,
how many were right.

After difficulty is assigned to each task, tasks are bucketed into tertiles
("easy", "medium", "hard") by task-difficulty rank (lowest third = easy,
highest third = hard). Then (H, C, A) is computed per (model, bin) on the
SAME ``_logprob_point_metrics`` definitions used elsewhere (DRY) — so the
trilemma axes have a consistent meaning across the static scatter, the
tau-sweep trajectories, and these stratified per-bin coordinates.

This is the second of three "competence-decoupled" views: the static
scatter is competence-confounded across the full task set, but within a
fixed difficulty bin the cross-model differences are decoupled from
"which models got which easy ones for free".
"""

import csv
import math
import os
from collections import defaultdict

from analysis.model_points import (  # reuse logprob predicates + metrics
    _logprob_acted,
    _logprob_point_metrics,
)

__all__ = [
    "task_difficulty",
    "stratified_coords",
]

_BIN_LABELS = ("easy", "medium", "hard")


def _read_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _group_by_model(runs_dir: str) -> dict[str, list[str]]:
    """Same layout contract as analysis.model_points._group_logprob_by_model."""
    groups: dict[str, list[str]] = {}
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
            groups[mid] = csvs
    return groups


def task_difficulty(runs_dir: str) -> dict[str, float]:
    """Per-task difficulty = 1 - cross-model mean committed-correctness.

    For each task t, average ``1[y == 1]`` across (model, seed) rows that
    actually committed on t (acted == 1). A task that 0 models committed
    on -> nan (rare); a task that 0 models got right -> 1.0; a task
    everyone got right -> 0.0.
    """
    # task -> list of y values across all (model, seed) rows that committed
    correct_counts: dict[str, list[int]] = defaultdict(list)
    groups = _group_by_model(runs_dir)
    for mid, csvs in groups.items():
        for p in csvs:
            for row in _read_rows(p):
                task = row.get("task", "")
                if not task:
                    continue
                if not _logprob_acted(row):
                    continue
                try:
                    y = int(float(row.get("y", "")))
                except (ValueError, TypeError):
                    continue
                correct_counts[task].append(1 if y == 1 else 0)
    out: dict[str, float] = {}
    for task, ys in correct_counts.items():
        if not ys:
            out[task] = float("nan")
            continue
        mean_correct = sum(ys) / len(ys)
        out[task] = 1.0 - mean_correct
    return out


def _tertile_bins(
    difficulty: dict[str, float], n_bins: int = 3,
) -> dict[str, str]:
    """Assign each task to a bin label.

    Tasks are sorted by difficulty (ascending: easiest first), then split
    into ``n_bins`` quantile-like buckets. Ties are stable (sorted order
    is deterministic by (difficulty, task_id)). NaN difficulties are
    placed in the "hard" bin (a task no model committed on is effectively
    unanswerable — treat as the extreme).
    """
    if n_bins != 3:
        # The interface accepts n_bins for symmetry, but only 3 is wired
        # to the manuscript's "easy/medium/hard" vocabulary. For any other
        # n we still split into n quantile buckets with numeric labels.
        labels = [str(i) for i in range(n_bins)]
    else:
        labels = list(_BIN_LABELS)

    # Sort: nan goes to the end (hardest).
    def sort_key(item):
        t, d = item
        is_nan = math.isnan(d) if isinstance(d, float) else False
        return (1 if is_nan else 0, d, t)

    ordered = sorted(difficulty.items(), key=sort_key)
    n = len(ordered)
    if n == 0:
        return {}
    out: dict[str, str] = {}
    # Distribute as evenly as possible; first buckets get the extra task
    # when n % n_bins != 0.
    base = n // n_bins
    rem = n % n_bins
    sizes = [base + (1 if i < rem else 0) for i in range(n_bins)]
    idx = 0
    for bin_i, size in enumerate(sizes):
        for _ in range(size):
            t, _d = ordered[idx]
            out[t] = labels[bin_i]
            idx += 1
    return out


def stratified_coords(
    runs_dir: str, n_bins: int = 3,
) -> dict[str, dict[str, dict]]:
    """(H, C, A) per (model, difficulty_bin).

    Returns ``{model_id: {bin_label: {H, C, A, n_acted, n_tasks_in_bin,
    n_rows_in_bin, partial: False}}}``. ``partial`` is included for shape
    parity with ``model_coords`` but is always False here (stratified
    coords are only meaningful when the model has data across bins; the
    figure code can choose to fade-or-skip empty bins).
    """
    difficulty = task_difficulty(runs_dir)
    bin_of = _tertile_bins(difficulty, n_bins=n_bins)
    groups = _group_by_model(runs_dir)

    if n_bins == 3:
        labels = list(_BIN_LABELS)
    else:
        labels = [str(i) for i in range(n_bins)]

    out: dict[str, dict[str, dict]] = {}
    for mid, csvs in groups.items():
        # bin -> list of pooled rows for this model in that bin
        bin_rows: dict[str, list[dict]] = {lbl: [] for lbl in labels}
        bin_tasks: dict[str, set] = {lbl: set() for lbl in labels}
        for p in csvs:
            for row in _read_rows(p):
                task = row.get("task", "")
                lbl = bin_of.get(task)
                if lbl is None:
                    continue
                bin_rows[lbl].append(row)
                bin_tasks[lbl].add(task)
        by_bin: dict[str, dict] = {}
        for lbl in labels:
            rows = bin_rows[lbl]
            h, a, c, n_acted = _logprob_point_metrics(rows)
            by_bin[lbl] = {
                "H": h,
                "C": c,
                "A": a,
                "n_acted": n_acted,
                "n_rows_in_bin": len(rows),
                "n_tasks_in_bin": len(bin_tasks[lbl]),
                "partial": False,
            }
        out[mid] = by_bin
    return out
