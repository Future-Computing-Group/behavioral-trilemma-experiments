"""Regression tests for the H3 binomial confidence interval.

Bug context: `test_h3_convexity_by_N` and the primary-H3 reconciliation
wrapper computed the CI as
`stats.binomtest(k, n, p=tol, alternative="less").proportion_ci(method="exact")`.
With a one-sided binomtest, scipy returns a *one-sided* interval whose
lower bound is forced to 0.0 for every input. This contaminated Fig 2
(h3_convexity_by_N) and the primary H3 statistic quoted in the A1
abstract/table/conclusion ([0.000, 0.188] instead of the correct
two-sided exact [0.038, 0.205] for 6/60).

The one-sided *p-value* is legitimate (the H3 criterion is directional:
violation rate < tolerance). Only the reported 95% CI must be the
two-sided Clopper-Pearson interval.

Correct two-sided Clopper-Pearson 95% reference values (scipy
beta.ppf): 6/60 -> (0.0376, 0.2051); 0/10 -> (0.0, 0.3085);
10/10 -> (0.6915, 1.0).
"""

from __future__ import annotations

import pytest

from analysis.hypothesis_tests import clopper_pearson_ci


def test_cp_ci_six_of_sixty_is_two_sided() -> None:
    """The primary-H3 case. The bug returned (0.000, 0.188)."""
    lo, hi = clopper_pearson_ci(6, 60, confidence=0.95)
    assert lo == pytest.approx(0.0376, abs=1e-3), (
        f"lower bound must be the two-sided CP value ~0.0376, got {lo} "
        "(0.0 means the one-sided-CI bug is still present)"
    )
    assert hi == pytest.approx(0.2051, abs=1e-3), f"upper ~0.2051, got {hi}"


def test_cp_ci_lower_bound_strictly_positive_when_k_positive() -> None:
    """The exact invariant the bug violated: k>0 => lo>0."""
    for k, n in [(1, 10), (8, 60), (17, 60), (6, 60), (3, 12)]:
        lo, hi = clopper_pearson_ci(k, n)
        assert lo > 0.0, f"k={k},n={n}: two-sided CP lower bound must be >0, got {lo}"
        assert hi < 1.0, f"k={k},n={n}: upper bound must be <1, got {hi}"
        assert lo < k / n < hi, f"k={k},n={n}: CI must bracket the point estimate"


def test_cp_ci_boundary_cases() -> None:
    """k=0 and k=n: one bound legitimately saturates, the other does not."""
    lo0, hi0 = clopper_pearson_ci(0, 10)
    assert lo0 == 0.0 and hi0 == pytest.approx(0.3085, abs=1e-3)
    lon, hin = clopper_pearson_ci(10, 10)
    assert lon == pytest.approx(0.6915, abs=1e-3) and hin == 1.0


def test_cp_ci_confidence_level_widens() -> None:
    lo95, hi95 = clopper_pearson_ci(6, 60, confidence=0.95)
    lo99, hi99 = clopper_pearson_ci(6, 60, confidence=0.99)
    assert lo99 < lo95 and hi99 > hi95, "99% CI must be wider than 95%"
