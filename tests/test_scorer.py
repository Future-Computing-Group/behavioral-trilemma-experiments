"""Tests for scoring functions. Hand-calculated examples."""
import pytest
from src.scorer import oracle_payoff, proxy_payoff, brier_score, inflation


def test_oracle_payoff_correct_high_confidence():
    # r=0.9, y=1 (correct), w_C=1, w_A=1, r_min=0.7
    # V = -(0.9-1)^2 + 1*1{0.9>=0.7} = -0.01 + 1 = 0.99
    v = oracle_payoff(r=0.9, y=1, w_C=1.0, w_A=1.0, r_min=0.7)
    assert abs(v - 0.99) < 1e-10


def test_oracle_payoff_incorrect_high_confidence():
    # r=0.9, y=0 (wrong), w_C=1, w_A=1, r_min=0.7
    # V = -(0.9-0)^2 + 1*1{0.9>=0.7} = -0.81 + 1 = 0.19
    v = oracle_payoff(r=0.9, y=0, w_C=1.0, w_A=1.0, r_min=0.7)
    assert abs(v - 0.19) < 1e-10


def test_oracle_payoff_below_threshold():
    # r=0.5, y=1, w_C=1, w_A=1, r_min=0.7
    # V = -(0.5-1)^2 + 1*1{0.5>=0.7} = -0.25 + 0 = -0.25
    v = oracle_payoff(r=0.5, y=1, w_C=1.0, w_A=1.0, r_min=0.7)
    assert abs(v - (-0.25)) < 1e-10


def test_oracle_payoff_no_autonomy():
    # w_A=0: pure Brier
    # r=0.8, y=1, w_C=1, w_A=0, r_min=0.7
    # V = -(0.8-1)^2 + 0 = -0.04
    v = oracle_payoff(r=0.8, y=1, w_C=1.0, w_A=0.0, r_min=0.7)
    assert abs(v - (-0.04)) < 1e-10


def test_proxy_payoff():
    # r=0.8, w_C=1, w_A=0, r_min=0.7
    # V_proxy = -1*0.8*0.2 + 0 = -0.16
    v = proxy_payoff(r=0.8, w_C=1.0, w_A=0.0, r_min=0.7)
    assert abs(v - (-0.16)) < 1e-10


def test_proxy_payoff_with_gating():
    # r=0.8, w_C=1, w_A=2, r_min=0.7
    # V_proxy = -1*0.8*0.2 + 2*1{0.8>=0.7} = -0.16 + 2 = 1.84
    v = proxy_payoff(r=0.8, w_C=1.0, w_A=2.0, r_min=0.7)
    assert abs(v - 1.84) < 1e-10


def test_brier_score():
    assert abs(brier_score(r=0.8, y=1) - 0.04) < 1e-10
    assert abs(brier_score(r=0.8, y=0) - 0.64) < 1e-10
    assert abs(brier_score(r=1.0, y=1) - 0.0) < 1e-10
    assert abs(brier_score(r=0.0, y=0) - 0.0) < 1e-10


def test_inflation():
    assert abs(inflation(r=0.85, p_hat=0.6) - 0.25) < 1e-10
    assert abs(inflation(r=0.5, p_hat=0.5) - 0.0) < 1e-10
    assert abs(inflation(r=0.3, p_hat=0.6) - (-0.3)) < 1e-10
