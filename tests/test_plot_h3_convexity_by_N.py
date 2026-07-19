"""Tests for scripts/plot_h3_convexity_by_N.py (audit C-3/LB-14/C-10).

The manuscript retired H3 as a hypothesis (descriptive surface geometry,
no inferential weight) and nowhere mentions a 15% tolerance; the figure
must not carry a live 'H3' hypothesis label nor the retired tolerance
line, and the annotated p must be the real value (scientific notation),
never a '0.00' truncation.
"""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.plot_h3_convexity_by_N import build_figure, format_p  # noqa: E402


@pytest.fixture
def results():
    path = ROOT / "experiment_output" / "analysis" / "hypothesis_results.json"
    if not path.exists():
        pytest.skip("canonical hypothesis_results.json not present")
    with open(path) as f:
        return json.load(f)


def test_format_p_never_truncates_to_zero():
    # The audit finding: annotation printed 'p = 0.00' for p = 3.09e-4.
    assert "0.00" not in format_p(0.000309)
    assert format_p(0.000309) == r"3 \times 10^{-4}"
    assert format_p(0.0234) == "0.023"
    assert format_p(0.5) == "0.500"
    assert format_p(1.7e-6) == r"2 \times 10^{-6}"


def test_figure_has_descriptive_title_no_hypothesis_label():
    fig, ax = build_figure_from_canonical()
    title = ax.get_title()
    assert "descriptive" in title.lower()
    assert "H3" not in title


def test_figure_has_no_tolerance_line():
    fig, ax = build_figure_from_canonical()
    labels = _all_labels(ax)
    assert not any("tolerance" in lab.lower() for lab in labels)
    # No horizontal dashed line at 0.15 either.
    for line in ax.get_lines():
        ys = set(line.get_ydata())
        assert ys != {0.15}, "retired 15% tolerance line present"


def test_figure_annotation_prints_real_p():
    fig, ax = build_figure_from_canonical()
    texts = [t.get_text() for t in ax.texts]
    joined = " ".join(texts)
    assert "p$ = 0.00" not in joined and "p = 0.00" not in joined
    assert "10^{-4}" in joined  # canonical trend p is 3.09e-4


def build_figure_from_canonical():
    path = ROOT / "experiment_output" / "analysis" / "hypothesis_results.json"
    if not path.exists():
        pytest.skip("canonical hypothesis_results.json not present")
    with open(path) as f:
        results = json.load(f)
    return build_figure(results)


def _all_labels(ax):
    legend = ax.get_legend()
    labels = [t.get_text() for t in legend.get_texts()] if legend else []
    labels += [line.get_label() for line in ax.get_lines()]
    return labels
