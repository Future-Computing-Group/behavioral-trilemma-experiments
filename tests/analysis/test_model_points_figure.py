"""Tests for scripts.plot_model_points.build_figure (A.3).

TDD RED step: written before the implementation exists. All fixtures use a
tmp_path synthetic runs-dir (a few tiny CSVs for two fake models, one of
which is partial) and the figure is written to tmp_path/out_dir — the real
experiment_output/competence_probe/figures is NEVER touched here. The real
directory is only used by the standalone driver in step 3, not by tests.

The figure is asserted at the data/contract level (files written, returned
dict shape, partial flag surfaced, caption string actually handed to the
figure), NOT by pixel inspection — the honest caption is enforced on the
string the renderer receives.
"""

import csv
import pathlib

import pytest

from analysis.model_points import HONEST_CAPTION, HONEST_CAPTION_LOGPROB

HEADER = [
    "task_id", "category", "N", "w_ratio", "w_C", "w_A", "r_min",
    "seed", "r_selected", "y", "V_selected", "brier", "gate_cleared",
]


def _write_csv(path: pathlib.Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for i, r in enumerate(rows):
            full = {
                "task_id": r.get("task_id", f"t{i:03d}"),
                "category": r.get("category", "arithmetic_easy"),
                "N": r.get("N", 1),
                "w_ratio": r.get("w_ratio", 0),
                "w_C": r.get("w_C", 1.0),
                "w_A": r.get("w_A", 0.0),
                "r_min": r.get("r_min", 0.5),
                "seed": r.get("seed", 2000),
                "r_selected": r.get("r_selected", ""),
                "y": r.get("y", ""),
                "V_selected": r.get("V_selected", ""),
                "brier": r.get("brier", ""),
                "gate_cleared": r.get("gate_cleared", ""),
            }
            w.writerow(full)
    return str(path)


def _acted(r_selected, y, brier):
    return {"r_selected": r_selected, "y": y, "brier": brier,
            "gate_cleared": 1, "V_selected": 0.0}


def _abstained():
    return {"r_selected": "", "y": "", "brier": "", "gate_cleared": "",
            "V_selected": ""}


def _pad(rows, n):
    out = list(rows)
    while len(out) < n:
        out.append(_abstained())
    return out


@pytest.fixture
def synthetic_runs(tmp_path):
    """Two fake models: ``fakeA`` complete (5 seeds, >=100 rows), ``fakeB``
    partial (only 2 seeds). Both act on almost every task (A approx 1) so the
    test exercises the A-pinned rendering path the real data hits."""
    runs = tmp_path / "_runs"
    block_complete = (
        [_acted("0.9", "1", "0.01")] * 40
        + [_acted("0.3", "0", "0.09")] * 30
        + [_acted("0.7", "1", "0.09")] * 30
        + [_acted("0.5", "0", "0.25")] * 20
    )  # 120 acted rows, A == 1.0
    for s in (2000, 2001, 2002, 2003, 2004):
        _write_csv(runs / f"fakeA_N1_w0_r0.5_s{s}.csv", list(block_complete))
    block_partial = (
        [_acted("0.8", "1", "0.04")] * 60
        + [_acted("0.6", "0", "0.36")] * 40
        + [_acted("0.4", "1", "0.36")] * 20
    )  # 120 acted rows, A == 1.0
    for s in (2000, 2001):  # only 2 seeds -> partial
        _write_csv(runs / f"fakeB_N1_w0_r0.5_s{s}.csv", list(block_partial))
    return runs


def test_build_figure_writes_pdf_and_png(synthetic_runs, tmp_path):
    from scripts.plot_model_points import build_figure

    out_dir = tmp_path / "figures"
    result = build_figure(str(synthetic_runs), str(out_dir))

    pdf = out_dir / "model_points.pdf"
    png = out_dir / "model_points.png"
    assert pdf.is_file() and pdf.stat().st_size > 0
    assert png.is_file() and png.stat().st_size > 0

    # I/O is parameterised: nothing written outside the supplied out_dir.
    assert {p.name for p in out_dir.iterdir()} == {
        "model_points.pdf", "model_points.png"
    }


def test_build_figure_returns_per_model_entries(synthetic_runs, tmp_path):
    from scripts.plot_model_points import build_figure

    result = build_figure(str(synthetic_runs), str(tmp_path / "figures"))
    assert isinstance(result, dict)
    models = result["models"]
    assert set(models) == {"fakeA", "fakeB"}
    for mid, entry in models.items():
        for key in ("H", "C", "A", "n_acted", "n_seeds", "partial"):
            assert key in entry, f"{mid} missing {key}"
    # A is pinned approx 1 in the synthetic data, mirroring real probe.
    assert models["fakeA"]["A"] == pytest.approx(1.0)
    assert models["fakeB"]["A"] == pytest.approx(1.0)


def test_build_figure_flags_partial_model(synthetic_runs, tmp_path):
    from scripts.plot_model_points import build_figure

    result = build_figure(str(synthetic_runs), str(tmp_path / "figures"))
    models = result["models"]

    # Partial model is still plotted but explicitly flagged, never silently
    # completed.
    assert models["fakeB"]["partial"] is True
    assert models["fakeB"]["n_seeds"] == 2
    assert models["fakeA"]["partial"] is False
    assert models["fakeA"]["n_seeds"] == 5
    # The result must surface the set of partial models for the caller.
    assert "fakeB" in result["partial_models"]
    assert "fakeA" not in result["partial_models"]


def test_build_figure_renders_honest_caption(synthetic_runs, tmp_path):
    from scripts.plot_model_points import build_figure

    result = build_figure(str(synthetic_runs), str(tmp_path / "figures"))
    # Assert on the caption string actually handed to the figure, not pixels.
    caption = result["caption"]
    assert isinstance(caption, str) and caption.strip()
    # It must be (or begin with) the canonical HONEST_CAPTION, verbatim.
    assert caption == HONEST_CAPTION or HONEST_CAPTION.startswith(
        caption.strip()
    ) or caption.startswith(HONEST_CAPTION[:80])

    low = caption.lower()
    for token in ("descriptive", "not", "competence", "mechanism"):
        assert token in low, f"caption missing honest token: {token!r}"
    assert ("ungated" in low) or ("no abstention" in low) or ("no gat" in low)


def test_build_figure_uses_agg_backend(synthetic_runs, tmp_path):
    """No-UI-tools rule: the script must force the headless Agg backend."""
    import matplotlib

    from scripts.plot_model_points import build_figure  # noqa: F401

    build_figure(str(synthetic_runs), str(tmp_path / "figures"))
    assert matplotlib.get_backend().lower() == "agg"


# ===== L.5: logprob figure mode =========================================== #
#
# Additive: the competence-probe build_figure path above MUST stay green.
# The new build_logprob_figure sources analysis.model_points' LOGPROB loader
# (all_logprob_model_coords) over a synthetic logprob_xmodel/ tree, removes
# BOTH burned-in on-figure caption boxes, and adds a theory-reference
# corner/trade-off annotation. RED first: build_logprob_figure does not yet
# exist.

LOGPROB_HEADER = [
    "task", "category", "seed", "r_logprob", "answer", "y", "acted",
]


def _write_logprob_csv(path: pathlib.Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOGPROB_HEADER)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({
                "task": r.get("task", f"t{i:03d}"),
                "category": r.get("category", "arithmetic_easy"),
                "seed": r.get("seed", 2000),
                "r_logprob": r.get("r_logprob", ""),
                "answer": r.get("answer", ""),
                "y": r.get("y", ""),
                "acted": r.get("acted", 0),
            })
    return str(path)


def _lp_row(task, r_logprob, answer, y, acted):
    return {"task": task, "r_logprob": r_logprob, "answer": answer,
            "y": y, "acted": acted}


@pytest.fixture
def synthetic_logprob_runs(tmp_path):
    """Synthetic logprob_xmodel/<model>/<model>_s<seed>.csv tree.

    Two models. ``lpA`` is complete (5 seeds, 120 rows each, some abstained
    so A < 1 and the trade-off geometry is visible). ``lpB`` is partial
    (2 seeds only). Schema = the canonical 7-col logprob schema.
    """
    runs = tmp_path / "logprob_xmodel"
    block_a = (
        [_lp_row("t", 0.9, "a", 1, 1)] * 50      # acted, correct
        + [_lp_row("t", 0.3, "b", 0, 1)] * 40    # acted, incorrect
        + [_lp_row("t", 0.5, "", "", 0)] * 30    # abstained (no answer)
    )  # 120 rows, A = 90/120 = 0.75
    for s in (1, 2, 3, 4, 5):
        _write_logprob_csv(runs / "lpA" / f"lpA_s{s}.csv", list(block_a))
    block_b = (
        [_lp_row("t", 0.8, "a", 1, 1)] * 70
        + [_lp_row("t", 0.6, "b", 0, 1)] * 50
    )  # 120 rows, A = 1.0
    for s in (1, 2):  # only 2 seeds -> partial
        _write_logprob_csv(runs / "lpB" / f"lpB_s{s}.csv", list(block_b))
    return runs


def test_logprob_figure_sources_logprob_loader(
    synthetic_logprob_runs, tmp_path, monkeypatch
):
    """build_logprob_figure MUST source coordinates from the LOGPROB loader
    analysis.model_points.all_logprob_model_coords, not the competence
    loader. We spy on the loader and assert it was invoked with the runs
    dir, and that the returned per-model dict carries logprob coords."""
    import scripts.plot_model_points as mod

    calls = {}
    real = mod.all_logprob_model_coords

    def spy(runs_dir, *a, **kw):
        calls["runs_dir"] = runs_dir
        return real(runs_dir, *a, **kw)

    monkeypatch.setattr(mod, "all_logprob_model_coords", spy)

    out_dir = tmp_path / "figures"
    result = mod.build_logprob_figure(
        str(synthetic_logprob_runs), str(out_dir)
    )
    assert calls.get("runs_dir") == str(synthetic_logprob_runs)

    models = result["models"]
    assert set(models) == {"lpA", "lpB"}
    # lpA: 90 acted / 120 -> A == 0.75 (proves the answer-parse loader,
    # NOT a competence path that would mark every row acted).
    assert models["lpA"]["A"] == pytest.approx(0.75)
    assert models["lpB"]["A"] == pytest.approx(1.0)
    assert models["lpB"]["partial"] is True
    assert models["lpA"]["partial"] is False

    png = out_dir / "model_points_logprob.png"
    assert png.is_file() and png.stat().st_size > 0


def test_logprob_figure_no_burned_in_caption_box(
    synthetic_logprob_runs, tmp_path
):
    """BOTH on-figure caption boxes are removed in logprob mode. No Text
    artist on the figure or any axes may contain the honest-caption
    sentinel or the 'ungated probe' textbox phrase, and no Text artist
    may be a large multi-sentence caption block (> 200 chars)."""
    import scripts.plot_model_points as mod

    fig, result = mod.build_logprob_figure(
        str(synthetic_logprob_runs), str(tmp_path / "figures"),
        return_figure=True,
    )
    try:
        texts = list(fig.findobj(match=mod.plt.Text))
        for t in texts:
            s = t.get_text() or ""
            low = s.lower()
            # The competence-probe burned-in textbox phrasing.
            assert "ungated probe" not in low, (
                f"burned-in A-pinned textbox still present: {s!r}"
            )
            # The honest-caption sentinel must NOT be drawn on the figure.
            assert "is not a trilemma proof" not in low and (
                "not a trilemma proof" not in low
            ), f"honest-caption box still burned into figure: {s!r}"
            # No giant caption block burned in.
            assert len(s) <= 200, (
                f"large caption block burned into figure ({len(s)} chars): "
                f"{s[:80]!r}"
            )
    finally:
        mod.plt.close(fig)

    # The caption text still travels with the result (for the report /
    # LaTeX caption), it is just not drawn on the image.
    assert result["caption"] == HONEST_CAPTION_LOGPROB


def test_logprob_figure_has_three_panels(
    synthetic_logprob_runs, tmp_path
):
    """The redesigned logprob figure MUST be a 3-panel pairwise scatter:
    panel 0 = H vs C, panel 1 = H vs A, panel 2 = C vs A. Asserted on the
    Figure's axes count and per-axis x/y label text (not pixels) so the
    pairwise structure cannot silently collapse back to a single panel."""
    import scripts.plot_model_points as mod

    fig, _result = mod.build_logprob_figure(
        str(synthetic_logprob_runs), str(tmp_path / "figures"),
        return_figure=True,
    )
    try:
        # Exactly three data-bearing axes (the suptitle / legend live on
        # the figure, not as extra Axes objects).
        assert len(fig.axes) == 3, (
            f"expected 3 pairwise panels, got {len(fig.axes)}"
        )
        labels = [
            (ax.get_xlabel().lower(), ax.get_ylabel().lower())
            for ax in fig.axes
        ]
        # Panel 0: H (x) vs C (y).
        assert labels[0][0].startswith("h "), labels
        assert labels[0][1].startswith("c "), labels
        # Panel 1: H (x) vs A (y).
        assert labels[1][0].startswith("h "), labels
        assert labels[1][1].startswith("a "), labels
        # Panel 2: C (x) vs A (y).
        assert labels[2][0].startswith("c "), labels
        assert labels[2][1].startswith("a "), labels
    finally:
        mod.plt.close(fig)


def test_logprob_figure_no_corner_star_or_joint_good_label(
    synthetic_logprob_runs, tmp_path
):
    """Affirmative-removal guard (closes the L60-style trap raised in the
    L.5 review). The corner-star + 'joint-good corner' / 'infeasible'
    annotations were dropped on 2026-05-20 because the scatter does not in
    fact exhibit a trilemma-shaped achievable boundary; the empty top-right
    is equally compatible with competence-confounding. No Text artist may
    contain 'infeasible', 'joint-good', or 'joint good'; and no artist may
    carry the gid 'theory-corner' or 'theory-tradeoff'."""
    import scripts.plot_model_points as mod

    fig, _result = mod.build_logprob_figure(
        str(synthetic_logprob_runs), str(tmp_path / "figures"),
        return_figure=True,
    )
    try:
        texts = [
            (t.get_text() or "").lower()
            for t in fig.findobj(match=mod.plt.Text)
        ]
        joined = " ".join(texts)
        for forbidden in ("infeasible", "joint-good", "joint good"):
            assert forbidden not in joined, (
                f"forbidden corner-star label fragment {forbidden!r} "
                f"present in figure text: {texts}"
            )
        gids = {
            getattr(o, "get_gid", lambda: None)()
            for o in fig.findobj()
        }
        for forbidden_gid in ("theory-corner", "theory-tradeoff"):
            assert forbidden_gid not in gids, (
                f"forbidden artist gid {forbidden_gid!r} present "
                f"(gids seen: {sorted(g for g in gids if g)})"
            )
    finally:
        mod.plt.close(fig)


def test_logprob_figure_no_unjustified_directional_arrow(
    synthetic_logprob_runs, tmp_path
):
    """Affirmatively forbid the resurrection of a 'predicted trade-off
    direction' label or any artist with gid ``theory-tradeoff``. The
    trilemma is a non-attainability claim at a point, not a directional
    prediction, so any directional arrow on the placement panel would
    over-claim (closes the L60 figure-claim-chain gap raised in the L.5
    review). Asserted across all Text artists (case-insensitive substring)
    and across all artist gids."""
    import scripts.plot_model_points as mod

    fig, _result = mod.build_logprob_figure(
        str(synthetic_logprob_runs), str(tmp_path / "figures"),
        return_figure=True,
    )
    try:
        objs = list(fig.findobj())
        gids = {getattr(o, "get_gid", lambda: None)() for o in objs}
        assert "theory-tradeoff" not in gids, (
            "forbidden directional-arrow gid 'theory-tradeoff' present "
            "(the slope has no theoretical basis; do not redraw it)"
        )
        texts = [
            (t.get_text() or "").lower()
            for t in fig.findobj(match=mod.plt.Text)
        ]
        joined = " ".join(texts)
        assert "predicted trade-off direction" not in joined, (
            "forbidden directional label 'predicted trade-off direction' "
            f"present in figure text: {texts}"
        )
        # Also catch the en-dash / hyphen-collapsed variants.
        assert "predicted tradeoff direction" not in joined, (
            "forbidden directional label (hyphen-collapsed variant) present"
        )
    finally:
        mod.plt.close(fig)


def test_logprob_figure_uses_agg_backend(synthetic_logprob_runs, tmp_path):
    """No-UI-tools rule still holds for the logprob path."""
    import matplotlib

    import scripts.plot_model_points as mod

    mod.build_logprob_figure(
        str(synthetic_logprob_runs), str(tmp_path / "figures")
    )
    assert matplotlib.get_backend().lower() == "agg"


def test_competence_probe_path_unaffected(synthetic_runs, tmp_path):
    """Regression guard: the original competence-probe build_figure must
    still write its PDF+PNG and return the same dict shape after the
    logprob mode is added (additive change, no regression)."""
    from scripts.plot_model_points import build_figure

    out_dir = tmp_path / "figures"
    result = build_figure(str(synthetic_runs), str(out_dir))
    assert (out_dir / "model_points.pdf").is_file()
    assert (out_dir / "model_points.png").is_file()
    assert set(result["models"]) == {"fakeA", "fakeB"}
    assert result["caption"] == HONEST_CAPTION
