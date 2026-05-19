"""Per-model trilemma coordinates (H, C, A) with bootstrap CIs.

Pure analysis of the competence-probe per-task CSVs. No network, no model
calls, no plotting, no argparse. The only I/O is reading the CSV paths
passed in.

Row semantics (confirmed against src/orchestrator.py, ~L80-120): a task the
model abstained on / failed to parse is written with blank ``r_selected``,
``y``, ``brier``, ``gate_cleared`` (empty string). An acted task has a
non-blank ``r_selected``, ``y`` in {"0","1"}, and ``brier`` = (r - y)**2.
``gate_cleared`` is an r-coupled metric that is deliberately NOT used to
derive autonomy.

Metric definitions (manuscript-consistent):
  A (autonomy)     = action rate = mean over all task rows of 1[acted].
  H (helpfulness)  = mean over all task rows of 1[acted AND y == 1].
  C (calibration)  = 1 - mean(brier over acted rows only); nan if no acted.
A model is ``partial`` if it has < 5 seed CSVs OR any seed CSV with < 100
rows; partial points are returned but flagged so callers can fade/exclude
them rather than averaging them into a "complete" point.
"""

import csv
import math
import os
import re

__all__ = [
    "classify_row",
    "model_coords",
    "all_model_coords",
    "seed_coords",
    "calibration_points",
    "all_seed_coords",
    "all_calibration_points",
]

# Sentinels that count as "no action recorded" for r_selected.
_BLANK_SENTINELS = {"", "nan", "none"}

# Thresholds for the partial-data flag.
_REQUIRED_SEEDS = 5
_MIN_ROWS_PER_SEED = 100

# Bootstrap configuration.
_B = 2000
_CI_LO_PCT = 2.5
_CI_HI_PCT = 97.5

# Filename -> model-id: strip the "_N<digits>_w..." config suffix. The model
# id may itself contain underscores (e.g. mistral_7b-instruct-q4_K_M), so we
# split on the config boundary, not on the first underscore.
_CONFIG_SUFFIX_RE = re.compile(r"_N\d+_w.*$")

# Filename -> seed: the trailing "_s<digits>" before the .csv extension.
_SEED_SUFFIX_RE = re.compile(r"_s(\d+)(?:\.csv)?$")


def classify_row(row: dict) -> str:
    """Classify a task row as "acted" or "abstained".

    A row is acted iff ``r_selected`` is non-blank: not "", "nan", "None"
    (case-insensitive) and not whitespace-only.
    """
    raw = row.get("r_selected", "")
    if raw is None:
        return "abstained"
    s = str(raw).strip()
    if s.lower() in _BLANK_SENTINELS:
        return "abstained"
    return "acted"


def _read_rows(csv_path: str) -> list[dict]:
    """Read one probe CSV into a list of dict rows."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _pool(csv_paths: list[str]) -> tuple[list[dict], list[int]]:
    """Concatenate all seed CSVs' task rows. Returns (rows, per_file_counts)."""
    pooled: list[dict] = []
    counts: list[int] = []
    for p in csv_paths:
        rows = _read_rows(p)
        counts.append(len(rows))
        pooled.extend(rows)
    return pooled, counts


def _is_acted(row: dict) -> bool:
    return classify_row(row) == "acted"


def _y_is_one(row: dict) -> bool:
    return str(row.get("y", "")).strip() == "1"


def _brier(row: dict) -> float:
    return float(row["brier"])


def _point_metrics(rows: list[dict]) -> tuple[float, float, float, int]:
    """Compute (H, A, C, n_acted) over a pooled row list."""
    n = len(rows)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    acted_flags = [_is_acted(r) for r in rows]
    n_acted = sum(acted_flags)
    a = n_acted / n
    h = sum(1 for r, act in zip(rows, acted_flags) if act and _y_is_one(r)) / n
    if n_acted == 0:
        c = float("nan")
    else:
        briers = [_brier(r) for r, act in zip(rows, acted_flags) if act]
        c = 1.0 - (sum(briers) / len(briers))
    return h, a, c, n_acted


def _bootstrap_ci(
    rows: list[dict], random_state: int
) -> tuple[tuple, tuple, tuple]:
    """Percentile bootstrap 95% CIs for (H, A, C) over per-task units.

    Resamples task rows with replacement B times. C is nan for any resample
    with zero acted rows; if every usable resample is degenerate (or there
    are no rows / no acted rows at all), the C CI is (nan, nan).
    """
    import random

    n = len(rows)
    nan_ci = (float("nan"), float("nan"))
    if n == 0:
        return nan_ci, nan_ci, nan_ci

    rng = random.Random(random_state)
    h_samples: list[float] = []
    a_samples: list[float] = []
    c_samples: list[float] = []
    for _ in range(_B):
        idx = [rng.randrange(n) for _ in range(n)]
        sample = [rows[i] for i in idx]
        h, a, c, _na = _point_metrics(sample)
        h_samples.append(h)
        a_samples.append(a)
        if not math.isnan(c):
            c_samples.append(c)

    def _pct(vals: list[float], q: float) -> float:
        # Linear-interpolation percentile (numpy "linear" / type-7).
        s = sorted(vals)
        if not s:
            return float("nan")
        if len(s) == 1:
            return s[0]
        rank = (q / 100.0) * (len(s) - 1)
        lo = math.floor(rank)
        hi = math.ceil(rank)
        if lo == hi:
            return s[lo]
        return s[lo] + (s[hi] - s[lo]) * (rank - lo)

    h_ci = (_pct(h_samples, _CI_LO_PCT), _pct(h_samples, _CI_HI_PCT))
    a_ci = (_pct(a_samples, _CI_LO_PCT), _pct(a_samples, _CI_HI_PCT))
    if c_samples:
        c_ci = (_pct(c_samples, _CI_LO_PCT), _pct(c_samples, _CI_HI_PCT))
    else:
        c_ci = nan_ci
    return h_ci, c_ci, a_ci


def model_coords(csv_paths: list[str], random_state: int = 0) -> dict:
    """Compute pooled (H, C, A) + bootstrap CIs for one model.

    csv_paths are the per-seed CSVs for a single model. Returns a dict with
    point estimates, 95% CIs, counts, and the ``partial`` flag.
    """
    pooled, per_file_counts = _pool(csv_paths)
    n_seeds = len(csv_paths)
    h, a, c, n_acted = _point_metrics(pooled)
    h_ci, c_ci, a_ci = _bootstrap_ci(pooled, random_state)

    partial = (
        n_seeds < _REQUIRED_SEEDS
        or any(cnt < _MIN_ROWS_PER_SEED for cnt in per_file_counts)
    )

    return {
        "H": h,
        "C": c,
        "A": a,
        "H_ci": h_ci,
        "C_ci": c_ci,
        "A_ci": a_ci,
        "n_acted": n_acted,
        "n_tasks": len(pooled),
        "n_seeds": n_seeds,
        "partial": partial,
    }


def _model_id_from_filename(filename: str) -> str:
    """Strip the '_N<digits>_w...' config suffix to get the model id."""
    stem = filename[:-4] if filename.endswith(".csv") else filename
    return _CONFIG_SUFFIX_RE.sub("", stem)


def _seed_from_filename(filename: str) -> int:
    """Parse the trailing ``_s<seed>`` integer from a probe CSV filename."""
    m = _SEED_SUFFIX_RE.search(filename)
    if m is None:
        raise ValueError(f"no _s<seed> suffix in filename: {filename!r}")
    return int(m.group(1))


def _group_by_model(runs_dir: str) -> dict[str, list[str]]:
    """Group ``runs_dir`` CSV paths by model-id prefix.

    Shared by ``all_model_coords`` / ``all_seed_coords`` /
    ``all_calibration_points`` so grouping stays DRY. Each model's path
    list is sorted (so seed order is deterministic).
    """
    groups: dict[str, list[str]] = {}
    for name in sorted(os.listdir(runs_dir)):
        if not name.endswith(".csv"):
            continue
        mid = _model_id_from_filename(name)
        groups.setdefault(mid, []).append(os.path.join(runs_dir, name))
    return {mid: sorted(paths) for mid, paths in groups.items()}


def all_model_coords(runs_dir: str, random_state: int = 0) -> dict[str, dict]:
    """Group CSVs in ``runs_dir`` by model-id prefix; coords per model."""
    return {
        mid: model_coords(paths, random_state=random_state)
        for mid, paths in _group_by_model(runs_dir).items()
    }


def seed_coords(csv_paths: list[str]) -> list[dict]:
    """Per-seed (H, C, A) for one model — one entry per seed CSV, unpooled.

    Uses the same ``_point_metrics`` definitions as ``model_coords`` but on
    each seed's rows in isolation (so C is nan, fail-soft, for a seed with
    zero acted rows). Entries are sorted by the seed integer parsed from
    the ``_s<seed>.csv`` filename suffix.
    """
    out: list[dict] = []
    for p in csv_paths:
        rows = _read_rows(p)
        h, a, c, n_acted = _point_metrics(rows)
        out.append({
            "seed": _seed_from_filename(os.path.basename(p)),
            "H": h,
            "C": c,
            "A": a,
            "n_acted": n_acted,
            "n_tasks": len(rows),
        })
    out.sort(key=lambda e: e["seed"])
    return out


def calibration_points(csv_paths: list[str]) -> list[dict]:
    """Pooled (r, y) for every ACTED output of one model.

    Abstained / parse-fail rows are excluded. Order is deterministic:
    input file order, then row order within each file.
    """
    pooled, _counts = _pool(csv_paths)
    return [
        {"r": float(row["r_selected"]), "y": int(row["y"])}
        for row in pooled
        if _is_acted(row)
    ]


def all_seed_coords(runs_dir: str) -> dict[str, list[dict]]:
    """Group CSVs in ``runs_dir`` by model-id; per-seed coords per model."""
    return {
        mid: seed_coords(paths)
        for mid, paths in _group_by_model(runs_dir).items()
    }


def all_calibration_points(runs_dir: str) -> dict[str, list[dict]]:
    """Group CSVs in ``runs_dir`` by model-id; calibration points per model."""
    return {
        mid: calibration_points(paths)
        for mid, paths in _group_by_model(runs_dir).items()
    }
