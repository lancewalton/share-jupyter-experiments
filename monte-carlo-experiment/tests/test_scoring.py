"""Correctness tests for scoring rules — a silent bug here invalidates everything."""
import numpy as np
from statistics import NormalDist

from mc.scoring import (
    crps_from_quantiles,
    interval_hit,
    pinball_loss,
    pit_value,
)

_N = NormalDist()


def test_pinball_zero_when_exact():
    assert pinball_loss(np.array([1.0]), np.array([1.0]), 0.5) == 0.0


def test_pinball_asymmetry():
    # y above q: penalised by level; y below q: penalised by (1-level).
    over = pinball_loss(np.array([0.0]), np.array([1.0]), 0.9)  # 0.9 * 1
    under = pinball_loss(np.array([0.0]), np.array([-1.0]), 0.9)  # 0.1 * 1
    assert np.isclose(over, 0.9)
    assert np.isclose(under, 0.1)


def test_pinball_minimised_at_true_quantile():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=200_000)
    level = 0.7
    grid = np.linspace(-1.0, 1.5, 60)
    mean_loss = [pinball_loss(np.full_like(sample, q), sample, level).mean() for q in grid]
    best = grid[int(np.argmin(mean_loss))]
    assert abs(best - _N.inv_cdf(level)) < 0.05


def test_crps_of_point_mass_is_absolute_error():
    # Degenerate predictive (all quantiles equal c): CRPS -> |y - c|.
    levels = np.round(np.arange(0.005, 0.9951, 0.005), 4)
    c = 2.0
    q = np.full(len(levels), c)
    crps = crps_from_quantiles(q, np.array(5.0), levels)
    assert np.isclose(crps, 3.0, atol=1e-2)


def test_crps_matches_closed_form_normal():
    # CRPS of N(0,1) predictive at y=0 has closed form 1/sqrt(pi) - ... ; use
    # the known value CRPS(N(0,1), 0) = 2/sqrt(2 pi) - 1/sqrt(pi) ~= 0.2337.
    levels = np.round(np.arange(0.0005, 0.99951, 0.0005), 5)
    q = np.array([_N.inv_cdf(float(p)) for p in levels])
    crps = crps_from_quantiles(q, np.array(0.0), levels)
    expected = 2.0 / np.sqrt(2 * np.pi) - 1.0 / np.sqrt(np.pi)
    assert np.isclose(crps, expected, atol=2e-3)


def test_interval_hit():
    assert interval_hit(np.array(-1.0), np.array(1.0), np.array(0.0))
    assert not interval_hit(np.array(-1.0), np.array(1.0), np.array(2.0))
    # boundary inclusive
    assert interval_hit(np.array(-1.0), np.array(1.0), np.array(1.0))


def test_pit_uniform_for_calibrated_normal():
    # If predictive == data-generating dist, PIT ~ Uniform(0,1).
    rng = np.random.default_rng(1)
    levels = np.round(np.arange(0.001, 0.9991, 0.001), 4)
    q = np.array([_N.inv_cdf(float(p)) for p in levels])
    ys = rng.normal(size=20_000)
    pits = np.array([pit_value(q, float(y), levels) for y in ys])
    # Uniform mean 0.5, and roughly flat: check quantiles of PIT.
    assert abs(pits.mean() - 0.5) < 0.01
    assert abs(np.quantile(pits, 0.9) - 0.9) < 0.02
