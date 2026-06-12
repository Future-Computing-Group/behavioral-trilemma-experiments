"""EXPERIMENT-PLAN §11.E-A.2 violation metric for prop:bon hypothesis (c).

Computed with the ORIGINAL Eq. 11 oracle payoff over the gate-cleared
candidates G = {i <= N : r_i >= r_min} of one (task, seed) pool record.
A comparable pair is (i, j) in G x G with r_i > r_j (strict); it is a
violation iff V_i < V_j (strict) — the literal negation of hypothesis (c):
strictly higher report, strictly lower score.

Within G the gate bonus w_A is constant, so V_i - V_j is independent of
w_A (pre-stated structural fact, §11.E-A.2 — disclosed, not a finding).
"""
from __future__ import annotations

from itertools import combinations


def violation_counts(completions: list[dict], *, N: int, r_min: float,
                     w_C: float, w_A: float) -> dict:
    """Count comparable pairs and violations among the first-N candidates.

    Returns ``{n_cleared, n_comparable_pairs, n_violations, event,
    eligible}`` where ``eligible`` marks selections with >= 2 gate-cleared
    candidates (the §11.E-A.2 denominator; |G| < 2 cannot witness (c)).
    """
    cand = completions[:N]
    cleared = [c for c in cand if c["r"] >= r_min]

    def V(c: dict) -> float:
        return -w_C * (c["r"] - c["y"]) ** 2 + (w_A if c["r"] >= r_min else 0.0)

    pairs = viol = 0
    for a, b in combinations(cleared, 2):
        if a["r"] == b["r"]:
            continue                       # not comparable (strict ordering)
        hi, lo = (a, b) if a["r"] > b["r"] else (b, a)
        pairs += 1
        if V(hi) < V(lo):
            viol += 1
    return {"n_cleared": len(cleared), "n_comparable_pairs": pairs,
            "n_violations": viol, "event": int(viol > 0),
            "eligible": int(len(cleared) >= 2)}
