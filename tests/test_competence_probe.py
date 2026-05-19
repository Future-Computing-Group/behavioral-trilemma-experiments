"""Tests for the multi-model competence probe pure functions (Path B A.2)."""

import pytest

from scripts.competence_probe import classify_tier, compute_separation


# --- classify_tier: one case per known suffix family + unknown -------------

@pytest.mark.parametrize(
    "category,expected",
    [
        ("arithmetic_easy", "easy"),
        ("factual_common", "easy"),
        ("code_simple", "easy"),
        ("arithmetic_hard", "hard"),
        ("factual_obscure", "hard"),
        ("code_algorithmic", "hard"),
    ],
)
def test_classify_tier_known_suffixes(category, expected):
    assert classify_tier(category) == expected


def test_classify_tier_unknown_suffix_raises():
    with pytest.raises(ValueError, match="unknown category suffix"):
        classify_tier("arithmetic_medium")


# --- helpers to build per-task dicts hitting exact means -------------------

def _build(easy_vals, hard_vals):
    """4 easy + 4 hard task ids; means equal the means of the given lists."""
    per_task_acc = {}
    task_category = {}
    for i, v in enumerate(easy_vals):
        tid = f"e{i}"
        per_task_acc[tid] = v
        task_category[tid] = "arithmetic_easy"
    for i, v in enumerate(hard_vals):
        tid = f"h{i}"
        per_task_acc[tid] = v
        task_category[tid] = "arithmetic_hard"
    return per_task_acc, task_category


# --- compute_separation: one fixture isolating each clause ----------------

def test_separation_separates_when_well_separated():
    # easy avg 0.85, hard avg 0.30
    acc, cat = _build([0.80, 0.90, 0.80, 0.90], [0.25, 0.35, 0.25, 0.35])
    out = compute_separation(acc, cat)
    assert out["easy_acc"] == pytest.approx(0.85)
    assert out["binding_acc"] == pytest.approx(0.30)
    assert out["h_ceiling"] == pytest.approx(0.85)
    assert out["separates"] is True
    assert out["reason"] == "separates"


def test_separation_fails_clause1_easy_below_min():
    # easy 0.38 / hard 0.30 -> fails clause 1 only
    acc, cat = _build([0.36, 0.40, 0.36, 0.40], [0.25, 0.35, 0.25, 0.35])
    out = compute_separation(acc, cat)
    assert out["easy_acc"] == pytest.approx(0.38)
    assert out["binding_acc"] == pytest.approx(0.30)
    assert out["separates"] is False
    assert out["reason"] == "easy_below_min"


def test_separation_fails_clause2_no_binding_tasks():
    # easy 0.95 / hard 0.92 -> passes 1, fails 2
    acc, cat = _build([0.95, 0.95, 0.95, 0.95], [0.90, 0.94, 0.90, 0.94])
    out = compute_separation(acc, cat)
    assert out["easy_acc"] == pytest.approx(0.95)
    assert out["binding_acc"] == pytest.approx(0.92)
    assert out["separates"] is False
    assert out["reason"] == "no_binding_tasks"


def test_separation_fails_clause3_insufficient_spread():
    # easy 0.55 / hard 0.45 -> passes 1 & 2, fails 3 (easy_min_acc=0.50)
    acc, cat = _build([0.50, 0.60, 0.50, 0.60], [0.40, 0.50, 0.40, 0.50])
    out = compute_separation(acc, cat, easy_min_acc=0.50)
    assert out["easy_acc"] == pytest.approx(0.55)
    assert out["binding_acc"] == pytest.approx(0.45)
    assert out["separates"] is False
    assert out["reason"] == "insufficient_spread"


def test_separation_empty_easy_set_raises():
    acc = {"h0": 0.3, "h1": 0.4}
    cat = {"h0": "arithmetic_hard", "h1": "arithmetic_hard"}
    with pytest.raises(ValueError):
        compute_separation(acc, cat)


def test_separation_empty_hard_set_raises():
    acc = {"e0": 0.8, "e1": 0.9}
    cat = {"e0": "arithmetic_easy", "e1": "arithmetic_easy"}
    with pytest.raises(ValueError):
        compute_separation(acc, cat)
