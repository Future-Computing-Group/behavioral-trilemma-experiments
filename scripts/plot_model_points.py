"""Real-models-as-points figure: descriptive placement + calibration.

This builds a two-panel figure from the committed competence-probe analysis
(``analysis.model_points``). It is deliberately a DESCRIPTIVE figure, not a
trilemma proof: see ``analysis.model_points.HONEST_CAPTION``, which is
rendered onto the figure itself and is the single source of truth for the
caveats (autonomy is pinned because the probe ran ungated).

No network, no model calls. The headless Agg backend is forced before
pyplot is imported (no-UI-tools rule). All output paths are parameterised
via ``out_dir`` so tests write to tmp_path and never the real directory.

Panels:
  1. (lead) Per-model reliability: binned reliability curve (mean
     correctness vs reported-confidence bin) with the raw per-output
     points faint behind it and the y=x ideal-calibration diagonal.
  2. (secondary) (H, C) placement with A annotated. Since A approx 1.0 for
     every model in this ungated probe, plotting a ternary/region would
     fabricate structure; instead H (x) vs C (y) is shown with per-seed
     clusters, mean +/- bootstrap CI, A encoded as marker size, and an
     explicit "A approx 1.0 (ungated probe)" annotation. Partial models
     are drawn faded/hollow with a "partial: N seeds" label.

Run directly to write the real figure:
    python -m scripts.plot_model_points
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import (no-UI rule)

import matplotlib.pyplot as plt  # noqa: E402

# Make the package importable whether run as a module or a script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from analysis.model_points import (  # noqa: E402
    HONEST_CAPTION,
    HONEST_CAPTION_LOGPROB,
    all_calibration_points,
    all_logprob_model_coords,
    all_model_coords,
    all_seed_coords,
)

# Real input/output locations (only used by the __main__ driver; tests pass
# their own tmp_path).
REAL_RUNS_DIR = os.path.join(
    _REPO_ROOT, "experiment_output", "competence_probe", "_runs"
)
REAL_FIGURES_DIR = os.path.join(
    _REPO_ROOT, "experiment_output", "competence_probe", "figures"
)

_N_BINS = 10
# A stable, color-blind-friendlier qualitative palette (matplotlib "tab10").
_PALETTE = plt.get_cmap("tab10").colors


def _reliability_bins(points: list[dict], n_bins: int = _N_BINS):
    """Return (bin_centers, mean_correctness) over confidence bins.

    Only bins that contain at least one acted output are returned, so an
    empty stretch of the confidence axis is not drawn as a fake zero.
    """
    if not points:
        return [], []
    edges = [i / n_bins for i in range(n_bins + 1)]
    centers: list[float] = []
    means: list[float] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # Last bin is closed on the right so r == 1.0 is counted.
        if b == n_bins - 1:
            sel = [p for p in points if lo <= p["r"] <= hi]
        else:
            sel = [p for p in points if lo <= p["r"] < hi]
        if not sel:
            continue
        centers.append((lo + hi) / 2.0)
        means.append(sum(p["y"] for p in sel) / len(sel))
    return centers, means


def _panel_calibration(ax, calib, color_of):
    """Panel 1 (lead): per-model reliability curve + faint raw points."""
    ax.plot(
        [0, 1], [0, 1], ls="--", lw=1.2, color="0.4",
        label="ideal calibration (y = x)", zorder=1,
    )
    for mid in sorted(calib):
        pts = calib[mid]
        if not pts:
            continue
        color = color_of[mid]
        # Faint raw per-output points (slight y-jitter for visibility).
        import random

        rng = random.Random(1234)
        xs = [p["r"] for p in pts]
        ys = [p["y"] + (rng.random() - 0.5) * 0.06 for p in pts]
        ax.scatter(
            xs, ys, s=8, alpha=0.12, color=color, edgecolors="none",
            zorder=2,
        )
        # Binned reliability curve on top.
        cx, cy = _reliability_bins(pts)
        if cx:
            ax.plot(
                cx, cy, "-o", color=color, lw=1.8, ms=5, label=mid,
                zorder=4,
            )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.12, 1.12)
    ax.set_xlabel("reported confidence  r")
    ax.set_ylabel("realized correctness  (bin mean; raw faint)")
    ax.set_title("Panel 1 — per-model reliability (lead)")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.2)


def _panel_placement(ax, coords, seeds, color_of, partial_models):
    """Panel 2 (secondary): H (x) vs C (y); A annotated, NOT a region."""
    for mid in sorted(coords):
        c = coords[mid]
        color = color_of[mid]
        is_partial = mid in partial_models

        # Per-seed cluster (translucent points in the model colour).
        for e in seeds.get(mid, []):
            if e["H"] != e["H"] or e["C"] != e["C"]:  # nan guard
                continue
            ax.scatter(
                e["H"], e["C"], s=26, alpha=0.20, color=color,
                edgecolors="none", zorder=2,
            )

        if c["H"] != c["H"] or c["C"] != c["C"]:  # nan guard
            continue
        # Marker size encodes A (so the pinned axis is visible, not traced).
        a_val = c["A"] if c["A"] == c["A"] else 0.0
        size = 60 + 240 * a_val
        h_err = [[c["H"] - c["H_ci"][0]], [c["H_ci"][1] - c["H"]]]
        c_err = [[c["C"] - c["C_ci"][0]], [c["C_ci"][1] - c["C"]]]
        label = mid
        if is_partial:
            label = f"{mid} (partial: {c['n_seeds']} seeds)"
        ax.errorbar(
            c["H"], c["C"], xerr=h_err, yerr=c_err,
            fmt="o" if not is_partial else "o",
            ms=0, ecolor=color, elinewidth=1.4, capsize=3, zorder=4,
        )
        ax.scatter(
            [c["H"]], [c["C"]], s=size,
            facecolors="none" if is_partial else color,
            edgecolors=color,
            linewidths=2.0 if is_partial else 1.0,
            alpha=0.55 if is_partial else 0.95,
            label=label, zorder=5,
        )
        ax.annotate(
            f"A={a_val:.2f}", (c["H"], c["C"]),
            textcoords="offset points", xytext=(7, 7), fontsize=7,
            color=color,
        )

    ax.set_xlabel("H  (helpfulness = acted & correct rate)")
    ax.set_ylabel("C  (calibration = 1 - mean Brier | acted)")
    ax.set_title("Panel 2 — (H, C) placement; A encoded as marker size")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    # The "A pinned because ungated" caveat is NOT burned onto the image
    # any more; it lives in the analysis report / LaTeX caption.


def build_figure(runs_dir: str, out_dir: str) -> dict:
    """Build the 2-panel descriptive figure from ``runs_dir``.

    Writes ``model_points.pdf`` and ``model_points.png`` into ``out_dir``
    (created if needed) and returns a dict::

        {
          "models": {model_id: {H,C,A,n_acted,n_seeds,partial}, ...},
          "partial_models": [model_id, ...],
          "caption": HONEST_CAPTION,            # string handed to the fig
          "out_pdf": "...", "out_png": "...",
        }

    Partial models are still plotted (faded/hollow) but surfaced via
    ``partial_models`` so the caller can never silently treat them as
    complete.
    """
    os.makedirs(out_dir, exist_ok=True)

    coords = all_model_coords(runs_dir)
    seeds = all_seed_coords(runs_dir)
    calib = all_calibration_points(runs_dir)

    model_ids = sorted(coords)
    color_of = {
        mid: _PALETTE[i % len(_PALETTE)] for i, mid in enumerate(model_ids)
    }
    partial_models = [m for m in model_ids if coords[m]["partial"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.4))
    _panel_calibration(ax1, calib, color_of)
    _panel_placement(ax2, coords, seeds, color_of, partial_models)

    # The honest caption travels with the returned dict (it lives in the
    # analysis report and later the LaTeX caption) and is deliberately NOT
    # burned onto the image — caveats must not be baked into pixels.
    caption = HONEST_CAPTION
    fig.suptitle(
        "Real models as points — descriptive placement (NOT a "
        "trilemma proof)",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    out_pdf = os.path.join(out_dir, "model_points.pdf")
    out_png = os.path.join(out_dir, "model_points.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    return {
        "models": {
            mid: {
                "H": coords[mid]["H"],
                "C": coords[mid]["C"],
                "A": coords[mid]["A"],
                "H_ci": coords[mid]["H_ci"],
                "C_ci": coords[mid]["C_ci"],
                "A_ci": coords[mid]["A_ci"],
                "n_acted": coords[mid]["n_acted"],
                "n_tasks": coords[mid]["n_tasks"],
                "n_seeds": coords[mid]["n_seeds"],
                "partial": coords[mid]["partial"],
            }
            for mid in model_ids
        },
        "partial_models": partial_models,
        "caption": caption,
        "out_pdf": out_pdf,
        "out_png": out_png,
    }


# --------------------------------------------------------------------------- #
# L.5: logprob cross-model figure mode.
#
# Sources per-model (H, C, A) + bootstrap CIs from the LOGPROB loader
# (analysis.model_points.all_logprob_model_coords reading
# experiment_output/logprob_xmodel/<model>/<model>_s<seed>.csv). Additive:
# the competence-probe build_figure path above is untouched.
#
# No on-figure caption box is drawn (neither the honest-caption block nor a
# burned-in "A pinned" textbox) — those caveats live in the analysis report
# and the LaTeX caption (HONEST_CAPTION_LOGPROB travels with the result).
#
# The placement panel carries a THEORY-REFERENCE annotation: a marker at the
# infeasible joint-good corner (high H + high C + high A simultaneously — the
# trilemma's empty corner). This is a descriptive corner-only annotation
# (label + marker), NOT a traced achievable region / fake Pareto frontier.
# No directional arrow is drawn: the trilemma is a non-attainability claim at
# a point, not a directional prediction, so any slope would over-claim.
# --------------------------------------------------------------------------- #

# Where the real logprob figure reads from / writes to (driver only; tests
# always pass their own tmp_path).
REAL_LOGPROB_RUNS_DIR = os.path.join(
    _REPO_ROOT, "experiment_output", "logprob_xmodel"
)
REAL_LOGPROB_FIGURES_DIR = os.path.join(
    _REPO_ROOT, "experiment_output", "analysis_logprob", "figures"
)


def _annotate_theory_geometry(ax) -> None:
    """Mark the infeasible joint-good corner only.

    Descriptive only: a single marker at the (high-H, high-C) joint-good
    corner (A would also have to be high there) and a single concise text
    label calling it the infeasible joint-good corner of the trilemma. The
    "theory reference, not data" qualifier lives in the LaTeX caption, not
    on the figure. No directional arrow is drawn: the trilemma is a
    non-attainability claim at a point, not a directional prediction, so a
    slope label ("predicted trade-off direction") would over-claim. Artists
    carry stable gids so tests can assert presence without pixel inspection.
    """
    # Joint-good corner: top-right of the (H, C) plane. A would also have to
    # be ~1 there simultaneously — that triple is the trilemma's empty
    # corner. We mark it on the (H, C) axes the panel already uses.
    corner = (0.98, 0.98)
    star = ax.scatter(
        [corner[0]], [corner[1]], marker="*", s=360,
        facecolors="none", edgecolors="#b00020", linewidths=1.8,
        zorder=6,
    )
    star.set_gid("theory-corner")
    # ONE concise consolidated label near the corner. The redundant
    # "theory reference, not data" qualifier has been moved to the LaTeX
    # caption to prevent text-overlap mashing seen at small fontsize.
    lbl = ax.annotate(
        "infeasible joint-good corner\n(theory, no model attains)",
        xy=corner, xytext=(-14, -14), textcoords="offset points",
        fontsize=10, ha="right", va="top", color="#b00020", zorder=6,
        annotation_clip=False,
    )
    lbl.set_gid("theory-corner")


def build_logprob_figure(
    runs_dir: str, out_dir: str, return_figure: bool = False
):
    """Build the logprob cross-model placement figure from ``runs_dir``.

    Sources per-model (H, C, A) + bootstrap CIs from the LOGPROB loader
    ``analysis.model_points.all_logprob_model_coords`` over
    ``runs_dir`` = ``experiment_output/logprob_xmodel/`` layout
    (``<model>/<model>_s<seed>.csv``). Writes ``model_points_logprob.pdf``
    and ``model_points_logprob.png`` into ``out_dir`` and returns the result
    dict (same shape as :func:`build_figure`, with
    ``caption == HONEST_CAPTION_LOGPROB``). With ``return_figure=True`` the
    open Figure is returned alongside the dict so tests can inspect text /
    annotation artists (the caller must close it).

    NO on-figure caption box is drawn: caveats live in the analysis report
    and the LaTeX caption. The placement panel carries a theory-reference
    corner + trade-off annotation (descriptive, not a traced region).
    """
    os.makedirs(out_dir, exist_ok=True)

    coords = all_logprob_model_coords(runs_dir)
    model_ids = sorted(coords)
    color_of = {
        mid: _PALETTE[i % len(_PALETTE)] for i, mid in enumerate(model_ids)
    }
    partial_models = [m for m in model_ids if coords[m]["partial"]]

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.6), constrained_layout=True)

    for mid in model_ids:
        c = coords[mid]
        color = color_of[mid]
        is_partial = mid in partial_models
        if c["H"] != c["H"] or c["C"] != c["C"]:  # nan guard
            continue
        a_val = c["A"] if c["A"] == c["A"] else 0.0
        size = 60 + 240 * a_val  # A encoded as marker size (pinned-axis cue)
        h_err = [[c["H"] - c["H_ci"][0]], [c["H_ci"][1] - c["H"]]]
        c_err = [[c["C"] - c["C_ci"][0]], [c["C_ci"][1] - c["C"]]]
        label = mid
        if is_partial:
            label = f"{mid} (partial: {c['n_seeds']} seeds)"
        ax.errorbar(
            c["H"], c["C"], xerr=h_err, yerr=c_err, fmt="o",
            ms=0, ecolor=color, elinewidth=1.4, capsize=3, zorder=4,
        )
        ax.scatter(
            [c["H"]], [c["C"]], s=size,
            facecolors="none" if is_partial else color,
            edgecolors=color,
            linewidths=2.0 if is_partial else 1.0,
            alpha=0.55 if is_partial else 0.95,
            label=label, zorder=5,
        )
        ax.annotate(
            f"A={a_val:.2f}", (c["H"], c["C"]),
            textcoords="offset points", xytext=(7, 7), fontsize=9,
            color=color,
        )

    _annotate_theory_geometry(ax)

    ax.set_xlim(-0.02, 1.06)
    ax.set_ylim(-0.02, 1.06)
    ax.set_xlabel("H  (helpfulness = acted & correct rate)")
    ax.set_ylabel("C  (calibration = 1 - mean logprob-Brier | acted)")
    ax.set_title(
        "Logprob cross-model placement — (H, C); A as marker size"
    )
    ax.grid(True, alpha=0.2)
    if model_ids:
        ax.legend(fontsize=9, loc="lower left", framealpha=0.9)

    # Caption travels with the result (report / LaTeX), NOT burned in.
    caption = HONEST_CAPTION_LOGPROB
    # Short descriptive title only. The full "not a proof / impossibility /
    # region" caveats are NOT drawn on the figure — they live verbatim in
    # the analysis report and the LaTeX caption (HONEST_CAPTION_LOGPROB).
    fig.suptitle(
        "Real models as points (logprob path) — descriptive placement",
        fontsize=12,
    )
    # constrained_layout handles padding; no tight_layout call needed.

    out_pdf = os.path.join(out_dir, "model_points_logprob.pdf")
    out_png = os.path.join(out_dir, "model_points_logprob.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)

    result = {
        "models": {
            mid: {
                "H": coords[mid]["H"],
                "C": coords[mid]["C"],
                "A": coords[mid]["A"],
                "H_ci": coords[mid]["H_ci"],
                "C_ci": coords[mid]["C_ci"],
                "A_ci": coords[mid]["A_ci"],
                "n_acted": coords[mid]["n_acted"],
                "n_tasks": coords[mid]["n_tasks"],
                "n_seeds": coords[mid]["n_seeds"],
                "partial": coords[mid]["partial"],
            }
            for mid in model_ids
        },
        "partial_models": partial_models,
        "caption": caption,
        "out_pdf": out_pdf,
        "out_png": out_png,
    }
    if return_figure:
        return fig, result
    plt.close(fig)
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode", choices=("competence", "logprob"), default="competence",
        help="competence-probe figure (default) or logprob cross-model "
        "figure",
    )
    args = ap.parse_args()

    if args.mode == "logprob":
        res = build_logprob_figure(
            REAL_LOGPROB_RUNS_DIR, REAL_LOGPROB_FIGURES_DIR
        )
    else:
        res = build_figure(REAL_RUNS_DIR, REAL_FIGURES_DIR)
    print(f"wrote {res['out_pdf']}")
    print(f"wrote {res['out_png']}")
    if res["partial_models"]:
        print(f"partial models (plotted, flagged): {res['partial_models']}")
    for mid, e in res["models"].items():
        print(
            f"  {mid}: H={e['H']:.3f} C={e['C']:.3f} A={e['A']:.3f} "
            f"n_acted={e['n_acted']} n_seeds={e['n_seeds']} "
            f"partial={e['partial']}"
        )
