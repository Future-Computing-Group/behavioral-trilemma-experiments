"""TDD tests for analysis.conditional_metrics (L.5 decoupled view #3).

H_conditional = correctness given commitment ("if you commit, are you
right?") — factors deferral out of helpfulness. The unconditional H from
analysis.model_points is action_rate * H_conditional; the conditional view
isolates competence from deferral policy.
"""

import csv
import math
import pathlib

import pytest

LOGPROB_HEADER = ["task", "category", "seed", "r_logprob", "answer", "y",
                  "acted"]


def _write_csv(path: pathlib.Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOGPROB_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({
                "task": r.get("task", "t"),
                "category": r.get("category", "arithmetic_easy"),
                "seed": r.get("seed", 1),
                "r_logprob": r.get("r_logprob", ""),
                "answer": r.get("answer", "a"),
                "y": r.get("y", ""),
                "acted": r.get("acted", 0),
            })
    return str(path)


def _r(r, y, acted, task="t"):
    return {"task": task, "r_logprob": r, "answer": "x" if acted else "",
            "y": y, "acted": acted}


def test_hand_fixture_h_cond_06(tmp_path):
    """5 acted of which 3 correct -> H_cond = 0.6, regardless of how many
    additional abstained rows are added."""
    from analysis.conditional_metrics import H_conditional

    base = [
        _r(0.9, 1, 1), _r(0.8, 1, 1), _r(0.7, 1, 1),
        _r(0.6, 0, 1), _r(0.5, 0, 1),
    ]
    path = _write_csv(tmp_path / "five.csv", base)
    out = H_conditional([path])
    assert out["H_cond"] == pytest.approx(0.6)
    assert out["n_acted"] == 5
    assert out["n_tasks"] == 5

    # Adding abstained rows must NOT move H_conditional
    with_absten = base + [_r(0.0, "", 0)] * 20
    path2 = _write_csv(tmp_path / "many.csv", with_absten)
    out2 = H_conditional([path2])
    assert out2["H_cond"] == pytest.approx(0.6)
    assert out2["n_acted"] == 5
    assert out2["n_tasks"] == 25


def test_zero_acted_returns_nan(tmp_path):
    from analysis.conditional_metrics import H_conditional
    path = _write_csv(tmp_path / "all_absten.csv", [_r(0.0, "", 0)] * 10)
    out = H_conditional([path])
    assert math.isnan(out["H_cond"])
    assert out["n_acted"] == 0
    assert out["partial"] is True or out["partial"] is False  # shape


def test_relation_to_unconditional_H(tmp_path):
    """H_unconditional == action_rate * H_conditional, exactly.

    From the definitions in analysis.model_points._logprob_point_metrics:
      H_unc = (correct ∧ acted) / n_total
      A     = acted / n_total
      H_cond = (correct ∧ acted) / acted
      ==>  H_unc = A * H_cond
    """
    from analysis.conditional_metrics import H_conditional
    from analysis.model_points import _logprob_point_metrics
    rows = [
        _r(0.9, 1, 1), _r(0.5, 0, 1), _r(0.7, 1, 1),
        _r(0.0, "", 0), _r(0.0, "", 0),
    ]
    path = _write_csv(tmp_path / "mix.csv", rows)
    out = H_conditional([path])
    import csv as _csv
    with open(path) as f:
        all_rows = list(_csv.DictReader(f))
    h_unc, a_unc, _c, _na = _logprob_point_metrics(all_rows)
    assert h_unc == pytest.approx(a_unc * out["H_cond"])


def test_returns_a_and_c_for_mix_and_match(tmp_path):
    """Module re-exports A and C so plot code can compose
    (H_conditional, C, A) without round-tripping through model_points."""
    from analysis.conditional_metrics import decoupled_triple
    rows = [
        _r(0.9, 1, 1), _r(0.5, 0, 1),
        _r(0.0, "", 0),
    ]
    path = _write_csv(tmp_path / "tri.csv", rows)
    out = decoupled_triple([path])
    # H_cond = 1/2; A = 2/3; C = 1 - mean((0.9-1)^2, (0.5-0)^2)
    #                       = 1 - mean(0.01, 0.25) = 1 - 0.13 = 0.87
    assert out["H_cond"] == pytest.approx(0.5)
    assert out["A"] == pytest.approx(2.0 / 3.0)
    assert out["C"] == pytest.approx(0.87)


def test_bootstrap_ci_is_returned(tmp_path):
    from analysis.conditional_metrics import H_conditional
    rows = [_r(0.9, 1, 1), _r(0.5, 0, 1)] * 30
    path = _write_csv(tmp_path / "boot.csv", rows)
    out = H_conditional([path], random_state=0)
    ci = out["H_cond_ci"]
    assert isinstance(ci, tuple) and len(ci) == 2
    lo, hi = ci
    # H_cond should be 0.5, CI brackets it
    assert lo <= 0.5 + 1e-9 and hi >= 0.5 - 1e-9
