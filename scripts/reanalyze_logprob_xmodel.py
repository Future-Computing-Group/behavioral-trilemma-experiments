#!/usr/bin/env python3
"""Re-analyse cross-model logprob CSVs with the charitable robust verifier.

Operational glue for L.5. For each
``experiment_output/logprob_xmodel/<model>/<model>_s<seed>.csv``:

1. If a sibling ``<csv>.orig`` does NOT already exist, copy the current
   CSV to ``<csv>.orig`` (one-time backup of the ORIGINAL exact-match ``y``).
2. Read the canonical source (``.orig``) — so re-runs are idempotent and
   we always recompute ``y`` from the original parsed answer, not from a
   previously-rewritten ``y``.
3. For each row, look up the task's gold from ``tasks/task_set.json`` by
   ``task_id`` and call ``analysis.robust_verify.robust_verify(row['answer'],
   task)`` to recompute ``y``. All other columns unchanged.
4. Write the rewritten CSV back to the original path.

Safety
------
* Skip any model whose CSV count < 5 (in-progress driver may still be
  appending to the directory; leave it alone — re-run after it finishes).
* Skip any individual CSV with empty content / zero rows defensively.
* Never touch ``experiment_output/`` files outside the per-model subdirs
  (no recursion outside the configured root).

Output
------
Per-model before -> after H so the delta is visible at a glance::

    qwen2.5_7b              H 0.503 -> 0.503  (delta +0.000)  n_acted 487/500
    granite3.1-dense_8b     H 0.221 -> 0.461  (delta +0.240)  n_acted 472/500
    ...

No network, no model calls. Run from the repo root:

    python3 scripts/reanalyze_logprob_xmodel.py
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.robust_verify import robust_verify  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_TASK_SET = _REPO_ROOT / "tasks" / "task_set.json"
_DEFAULT_RUNS_DIR = _REPO_ROOT / "experiment_output" / "logprob_xmodel"

# Mirrors scripts/eval_logprob._FIELDNAMES (must stay in lockstep).
_FIELDNAMES = [
    "task", "category", "seed", "r_logprob", "answer", "y", "acted",
]

# Minimum number of seed CSVs a model must have before we re-analyse it.
# Below this threshold we assume the driver is still in the middle of
# generating data for the model and leave the directory alone.
_MIN_SEEDS = 5


def _load_tasks(task_set_path: pathlib.Path) -> dict[str, dict]:
    """Load task_set.json and key by task['id']."""
    with open(task_set_path) as f:
        tasks = json.load(f)
    return {t["id"]: t for t in tasks}


def _read_rows(csv_path: pathlib.Path) -> list[dict]:
    """Read CSV rows; return [] for empty/header-only files."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(csv_path: pathlib.Path, rows: list[dict]) -> None:
    """Write CSV with the canonical header (column order preserved)."""
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        w.writeheader()
        for r in rows:
            # Only keep canonical fields; protect against stray columns.
            w.writerow({k: r.get(k, "") for k in _FIELDNAMES})


def _h_over_rows(rows: list[dict]) -> tuple[float, int, int]:
    """Compute (H, n_correct_acted, n_total) using the row's current y/acted.

    H = (# rows with acted==1 AND y==1) / n_total. Mirrors
    ``analysis.model_points._logprob_point_metrics``.
    """
    n_total = len(rows)
    if n_total == 0:
        return 0.0, 0, 0
    n_correct_acted = sum(
        1 for r in rows
        if str(r.get("acted", "")).strip() == "1"
        and str(r.get("y", "")).strip() == "1"
    )
    return n_correct_acted / n_total, n_correct_acted, n_total


def reanalyse_model(
    model_dir: pathlib.Path, tasks_by_id: dict[str, dict]
) -> dict:
    """Rewrite all complete CSVs in ``model_dir``. Returns a summary dict.

    Idempotent: the canonical source for ``answer`` (and original ``y``)
    is the one-time ``.orig`` backup, not the working CSV. Re-runs always
    recompute ``y`` from the original parsed answer.
    """
    csvs = sorted(p for p in model_dir.iterdir() if p.suffix == ".csv")
    summary = {
        "model": model_dir.name,
        "n_csvs": len(csvs),
        "skipped": None,
        "H_before": None,
        "H_after": None,
        "n_correct_before": 0,
        "n_correct_after": 0,
        "n_total": 0,
    }

    if len(csvs) < _MIN_SEEDS:
        summary["skipped"] = (
            f"only {len(csvs)} CSV(s) < min {_MIN_SEEDS} "
            "(driver likely still running)"
        )
        return summary

    # First pass: back up originals (idempotent), then compute H_before
    # from the original ``y`` column.
    pooled_before: list[dict] = []
    pooled_after: list[dict] = []
    for csv_path in csvs:
        orig = csv_path.with_suffix(csv_path.suffix + ".orig")
        if not orig.exists():
            shutil.copy2(csv_path, orig)

        rows_orig = _read_rows(orig)
        if not rows_orig:
            # Defensive: skip empty CSVs; do not rewrite.
            continue

        # Re-derive y from the original parsed answer + the robust verifier.
        rewritten: list[dict] = []
        for row in rows_orig:
            task = tasks_by_id.get(row.get("task", ""))
            answer = row.get("answer", "")
            if task is None or answer == "":
                new_y = 0
            else:
                new_y = robust_verify(answer, task)
            new_row = dict(row)
            new_row["y"] = new_y
            rewritten.append(new_row)

        _write_rows(csv_path, rewritten)
        pooled_before.extend(rows_orig)
        pooled_after.extend(rewritten)

    h_before, nc_before, n_total = _h_over_rows(pooled_before)
    h_after, nc_after, _ = _h_over_rows(pooled_after)
    summary.update({
        "H_before": h_before,
        "H_after": h_after,
        "n_correct_before": nc_before,
        "n_correct_after": nc_after,
        "n_total": n_total,
    })
    return summary


def reanalyse_all(
    runs_dir: pathlib.Path, task_set_path: pathlib.Path
) -> list[dict]:
    """Re-analyse every model subdirectory under ``runs_dir``."""
    tasks_by_id = _load_tasks(task_set_path)
    summaries: list[dict] = []
    if not runs_dir.is_dir():
        return summaries
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        summaries.append(reanalyse_model(sub, tasks_by_id))
    return summaries


def _print_summary(summaries: list[dict]) -> None:
    """Pretty-print the per-model before -> after H delta."""
    if not summaries:
        print("(no model directories found)")
        return
    name_w = max(len(s["model"]) for s in summaries)
    print(
        f"{'model'.ljust(name_w)}  H_before -> H_after  "
        f"(delta)             n_correct/n_total   note"
    )
    for s in summaries:
        if s["skipped"]:
            print(
                f"{s['model'].ljust(name_w)}  "
                f"   SKIPPED          "
                f"                                {s['skipped']}"
            )
            continue
        hb, ha = s["H_before"], s["H_after"]
        delta = ha - hb
        print(
            f"{s['model'].ljust(name_w)}  "
            f"H {hb:.3f} -> {ha:.3f}  (delta {delta:+.3f})    "
            f"{s['n_correct_after']}/{s['n_total']} "
            f"(was {s['n_correct_before']})"
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs-dir", default=str(_DEFAULT_RUNS_DIR),
        help="cross-model logprob runs root "
             f"(default: {_DEFAULT_RUNS_DIR})",
    )
    ap.add_argument(
        "--task-set", default=str(_DEFAULT_TASK_SET),
        help=f"100-task JSON (default: {_DEFAULT_TASK_SET})",
    )
    args = ap.parse_args(argv)

    summaries = reanalyse_all(
        pathlib.Path(args.runs_dir), pathlib.Path(args.task_set),
    )
    _print_summary(summaries)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
