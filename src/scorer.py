"""Scoring functions for the behavioral trilemma experiment."""


def oracle_payoff(r: float, y: int, w_C: float, w_A: float, r_min: float) -> float:
    """Oracle selection payoff (manuscript Eq. 11).

    V = -w_C * (r - y)^2 + w_A * 1{r >= r_min}
    """
    brier_term = -w_C * (r - y) ** 2
    gate_term = w_A * (1.0 if r >= r_min else 0.0)
    return brier_term + gate_term


def proxy_payoff(r: float, w_C: float, w_A: float, r_min: float) -> float:
    """No-oracle proxy payoff (control condition only).

    V_proxy = -w_C * r * (1 - r) + w_A * 1{r >= r_min}
    NOT a proper scoring rule.
    """
    variance_term = -w_C * r * (1.0 - r)
    gate_term = w_A * (1.0 if r >= r_min else 0.0)
    return variance_term + gate_term


def brier_score(r: float, y: int) -> float:
    """Brier score: (r - y)^2. Lower is better."""
    return (r - y) ** 2


def inflation(r: float, p_hat: float) -> float:
    """Confidence inflation: r - p_hat."""
    return r - p_hat
