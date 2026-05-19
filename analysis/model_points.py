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
    "HONEST_CAPTION",
    "HONEST_CAPTION_LOGPROB",
    "classify_row",
    "model_coords",
    "all_model_coords",
    "seed_coords",
    "calibration_points",
    "all_seed_coords",
    "all_calibration_points",
    "logprob_model_coords",
    "all_logprob_model_coords",
]

# Honest framing for any figure built from these coordinates. This is the
# single source of truth for the caption and is asserted by the test suite,
# not left to authorial goodwill. It states, in order: what the figure IS
# (descriptive placement of real named models), why the autonomy axis is
# pinned (the probe ran ungated, so A is not a trade-off), what that means
# for the (H,C,A) panel (competence-confounded), where the causal claim
# actually rests (the mechanism experiment + theory, NOT this scatter),
# and what the calibration panel genuinely shows.
HONEST_CAPTION = (
    "Descriptive placement of real, named open-weight models on the "
    "behavioral axes; this is NOT a trilemma proof, an impossibility "
    "result, or a traced trade-off region. The autonomy axis A is "
    "approximately 1.0 for every model because the competence probe ran "
    "ungated (no abstention incentive / no gating reward), so A is pinned "
    "by construction and the apparent absence of an autonomy trade-off is "
    "an artifact of the design, not evidence. A is itself only a proxy: "
    "it is the answer-commitment rate, where a task counts as 'abstained' "
    "whenever no parseable answer was produced -- which conflates "
    "deliberate deferral with mere unparseable-output (parse-fail) "
    "responses, so A measures behavioral answer-commitment, not "
    "autonomy-as-chosen-deferral. Consequently the (H, C, A) "
    "panel is competence-confounded and cannot be read as a causal "
    "trade-off: model placement there reflects task competence, not a "
    "mechanism. The causal trilemma claim rests on the gated mechanism "
    "experiment (hypotheses H1/H2/H4/H5) together with the theory, not on "
    "this scatter. What the calibration panel does show is each model's "
    "per-output reliability structure (reported confidence r versus "
    "realized correctness), i.e. the direction and magnitude of "
    "per-model overconfidence."
)

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
    rows: list[dict], random_state: int, point_fn=None
) -> tuple[tuple, tuple, tuple]:
    """Percentile bootstrap 95% CIs for (H, A, C) over per-task units.

    Resamples task rows with replacement B times. C is nan for any resample
    with zero acted rows; if every usable resample is degenerate (or there
    are no rows / no acted rows at all), the C CI is (nan, nan).

    ``point_fn`` is the per-resample metric function with the
    ``_point_metrics`` signature ``rows -> (H, A, C, n_acted)``. It defaults
    to ``_point_metrics`` (the competence-probe path) so existing
    ``model_coords`` / ``seed_coords`` callers are byte-for-byte unchanged;
    the logprob loader passes ``_logprob_point_metrics`` to reuse this exact
    resampling structure + ``_pct`` (DRY).
    """
    import random

    if point_fn is None:
        point_fn = _point_metrics

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
        h, a, c, _na = point_fn(sample)
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


# --------------------------------------------------------------------------- #
# Logprob cross-model loader (L.3).
#
# Reads scripts/eval_logprob.py output: experiment_output/logprob_xmodel/
# <model>/<model>_s<seed>.csv with columns EXACTLY
#   task,category,seed,r_logprob,answer,y,acted
#
# CRITICAL divergence from the competence-probe path above (the v2-IMPORTANT
# pin; do not let this drift): the ``acted`` predicate here is
# ANSWER-PARSE-BASED. A row is acted iff a valid ANSWER was successfully
# parsed from the completion via src/parser.py (recorded by eval_logprob.py
# as ``acted == 1``). It is NOT the archived competence-probe predicate
# ``classify_row`` (``r_selected`` non-blank, i.e. confidence-parseable
# post-argmax).
#
# Why the divergence is correct, not a bug: the logprob path has N=1 (no
# argmax) and ``r_logprob`` is the logprob-confidence computed from the
# completion's token logprobs, which EVERY completion has. So an
# ``r_logprob``-non-blank predicate would mark every single row "acted" ->
# A == 1.0 for every model, which is wrong. The manuscript autonomy A is
# the answer-commitment / non-abstention rate (main.tex:968 autonomy
# definition; consistent with HONEST_CAPTION's "answer-commitment rate"
# framing). An answer-parse ``acted`` matches that; a confidence-parse one
# would not. Abstained rows (acted==0) are excluded from C, count as
# not-acted for A, and contribute H=0 for that task.
# --------------------------------------------------------------------------- #

def _logprob_acted(row: dict) -> bool:
    """True iff eval_logprob.py recorded acted==1 (a valid ANSWER parsed).

    Answer-parse based (the v2-IMPORTANT pin) — NOT r_logprob-non-blank,
    NOT ``classify_row``. ``acted`` in {"1", 1}; anything else -> abstained.
    """
    return str(row.get("acted", "")).strip() == "1"


def _logprob_y_is_one(row: dict) -> bool:
    return str(row.get("y", "")).strip() == "1"


def _logprob_brier(row: dict) -> float:
    """(r_logprob - y)^2 for an acted row. Only called on acted rows."""
    r = float(row["r_logprob"])
    y = float(row["y"])
    return (r - y) ** 2


def _logprob_point_metrics(
    rows: list[dict],
) -> tuple[float, float, float, int]:
    """Compute (H, A, C, n_acted) over logprob_xmodel rows.

    Single-source: all three axes come from the SAME ``rows`` list.
      A = mean over ALL task rows of 1[acted==1]   (answer-parse based)
      H = mean over ALL task rows of 1[acted==1 AND y==1]
      C = 1 - mean over ACTED rows of (r_logprob - y)^2 ; nan if no acted.
    """
    n = len(rows)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    acted_flags = [_logprob_acted(r) for r in rows]
    n_acted = sum(acted_flags)
    a = n_acted / n
    h = sum(
        1 for r, act in zip(rows, acted_flags)
        if act and _logprob_y_is_one(r)
    ) / n
    if n_acted == 0:
        c = float("nan")
    else:
        briers = [
            _logprob_brier(r)
            for r, act in zip(rows, acted_flags) if act
        ]
        c = 1.0 - (sum(briers) / len(briers))
    return h, a, c, n_acted


def logprob_model_coords(
    csv_paths: list[str], random_state: int = 0
) -> dict:
    """Pooled (H, C, A) + bootstrap CIs for one model from logprob CSVs.

    All three axes are derived from the SAME pooled row list (single
    source — no cross-run / cross-model mixing). ``acted`` is the
    ANSWER-parse predicate (see module-level note). Reuses the existing
    ``_bootstrap_ci`` resampling engine + ``_pct`` via the ``point_fn``
    hook (DRY). Partial-model flag semantics mirror ``model_coords``
    (< 5 seed CSVs OR any seed CSV with < 100 rows).
    """
    pooled, per_file_counts = _pool(csv_paths)
    n_seeds = len(csv_paths)
    h, a, c, n_acted = _logprob_point_metrics(pooled)
    h_ci, c_ci, a_ci = _bootstrap_ci(
        pooled, random_state, point_fn=_logprob_point_metrics
    )

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


def _group_logprob_by_model(runs_dir: str) -> dict[str, list[str]]:
    """Group ``runs_dir`` logprob CSVs by their ``<model>/`` subdirectory.

    eval_logprob.py writes one subdir per model
    (``logprob_xmodel/<model>/<model>_s<seed>.csv``), so the model id is
    the subdirectory name (it may itself contain the ``:``->``_`` config
    chars). Each model's path list is sorted (deterministic seed order).
    """
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


def all_logprob_model_coords(
    runs_dir: str, random_state: int = 0
) -> dict[str, dict]:
    """Per-model logprob (H, C, A) for every ``<model>/`` subdir.

    Models are kept strictly separate (no cross-model pooling) — each
    model's coordinates are computed only from its own subdir CSVs.
    """
    return {
        mid: logprob_model_coords(paths, random_state=random_state)
        for mid, paths in _group_logprob_by_model(runs_dir).items()
    }


# Honest framing for the logprob cross-model figure. Mirrors HONEST_CAPTION's
# enforced-by-test discipline but re-worded for the logprob path. Single
# source of truth for that figure's caption; asserted by the test suite.
HONEST_CAPTION_LOGPROB = (
    "Descriptive placement of real, named open-weight models on the "
    "behavioral axes using the logprob-confidence path; this is NOT a "
    "trilemma proof, an impossibility result, or a traced trade-off "
    "region. The autonomy axis A is the answer-commitment rate: a task "
    "counts as acted only when a valid answer was successfully parsed from "
    "the completion via the ANSWER-parser, and as 'abstained' otherwise. "
    "This answer-parse acted predicate conflates deliberate deferral with "
    "mere parse-fail (unparseable-output) responses, so A measures "
    "behavioral answer-commitment, not autonomy-as-chosen-deferral. The "
    "logprob path runs ungated at N=1 (no argmax, no abstention incentive, "
    "no gating reward), so A is design-pinned by construction, not a "
    "traced trade-off. Consequently the (H, C, A) panel is "
    "competence-confounded and cannot be read as a causal trade-off: model "
    "placement reflects task competence, not a mechanism. The causal "
    "trilemma claim rests on the gated mechanism experiment (hypotheses "
    "H1/H2/H4/H5) together with the theory, NOT on this scatter. The "
    "calibration value C uses the per-output logprob-confidence; this "
    "loader is an independent reimplementation of the manuscript "
    "logprob-confidence equation and is NOT bit-verified against the "
    "original 540-config run, so C should be read as a faithful "
    "reimplementation, not a byte-exact reproduction."
)
