"""Tests for the FHS vol-matched surrogate."""
import numpy as np

from matching.surrogate import (
    ewma_vol,
    fhs_surrogate,
    resid_abs_autocorr,
    sign_flip_surrogate,
)


def _autocorr1(r):
    r = r - r.mean()
    return float(np.sum(r[:-1] * r[1:]) / (np.sum(r * r) + 1e-12))


def test_surrogate_preserves_drift_and_scale():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0004, 0.012, 4000)
    logP = np.concatenate([[np.log(100)], np.log(100) + np.cumsum(r)])
    sur = fhs_surrogate(logP, np.random.default_rng(1))
    rs = np.diff(sur)
    assert abs(rs.mean() - r.mean()) < 0.0003        # drift preserved
    assert abs(rs.std() / r.std() - 1.0) < 0.1       # overall scale preserved


def test_surrogate_destroys_return_autocorrelation():
    # build strongly momentum-autocorrelated returns; surrogate must remove it
    rng = np.random.default_rng(2)
    n = 6000
    e = rng.normal(0, 0.01, n)
    r = np.empty(n); r[0] = e[0]
    for i in range(1, n):
        r[i] = 0.4 * r[i - 1] + e[i]                 # AR(1), positive memory
    logP = np.concatenate([[np.log(50)], np.log(50) + np.cumsum(r)])
    sur = fhs_surrogate(logP, np.random.default_rng(3))
    assert _autocorr1(r) > 0.25                      # real has memory
    assert abs(_autocorr1(np.diff(sur))) < 0.06      # surrogate does not


def test_surrogate_preserves_vol_clustering():
    # returns with clustered volatility: |returns| autocorrelation should survive
    rng = np.random.default_rng(4)
    n = 8000
    h = np.zeros(n)                                   # persistent log-volatility
    for i in range(1, n):
        h[i] = 0.98 * h[i - 1] + 0.15 * rng.normal()
    r = rng.normal(0, 1, n) * (0.01 * np.exp(h))
    logP = np.concatenate([[np.log(80)], np.log(80) + np.cumsum(r)])
    sur = fhs_surrogate(logP, np.random.default_rng(5))
    real_absac = _autocorr1(np.abs(np.diff(logP)))
    sur_absac = _autocorr1(np.abs(np.diff(sur)))
    assert real_absac > 0.1
    assert sur_absac > 0.5 * real_absac              # clustering largely retained


def test_sign_flip_preserves_magnitudes_exactly_kills_direction():
    rng = np.random.default_rng(6)
    n = 6000
    e = rng.normal(0, 0.01, n)
    r = np.empty(n); r[0] = e[0]
    for i in range(1, n):
        r[i] = 0.4 * r[i - 1] + e[i]                  # directional memory
    logP = np.concatenate([[np.log(50)], np.log(50) + np.cumsum(r)])
    sur = sign_flip_surrogate(logP, np.random.default_rng(7))
    rs = np.diff(sur)
    # volatility held exactly (the property that matters), direction destroyed
    assert abs(rs.std() / r.std() - 1.0) < 1e-9        # variance preserved exactly
    assert abs(_autocorr1(rs)) < 0.06                  # direction destroyed


def test_ewma_vol_tracks_level_changes():
    x = np.concatenate([np.full(200, 0.01), np.full(200, 0.05)])
    v = ewma_vol(x, halflife=20)
    assert v[190] < v[-1]                             # rises after the level jump
