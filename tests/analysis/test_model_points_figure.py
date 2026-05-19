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

from analysis.model_points import HONEST_CAPTION

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
