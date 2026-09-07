"""Tests for the realised-vol / variance-premium primitives."""
import numpy as np

from vrp.premium import forward_realised_vol, realised_vol, variance_premium


def test_realised_vol_recovers_annualised_sigma():
    rng = np.random.default_rng(0)
    sigma = 0.012                                  # daily
    r = rng.normal(0, sigma, 200_000)
    rv = realised_vol(r)                           # points
    assert abs(rv - sigma * np.sqrt(252) * 100) < 0.5


def test_forward_realised_vol_is_causal_and_shifted():
    r = np.zeros(50)
    r[30:40] = 0.02                                # a burst of vol at 30..39
    rv = forward_realised_vol(r, H=10)
    # decision at t=29 sees returns 30..39 (the burst) -> high; at t=39 sees calm
    assert rv[29] > rv[39]
    assert np.isnan(rv[-1])                        # no future at the end


def test_variance_premium_signs():
    vol_p, var_p = variance_premium(np.array([20.0]), np.array([15.0]))
    assert np.isclose(vol_p[0], 5.0)
    assert np.isclose(var_p[0], 400 - 225)
