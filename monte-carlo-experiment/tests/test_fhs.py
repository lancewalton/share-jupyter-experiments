"""Tests for Filtered Historical Simulation."""
import numpy as np

from mc.forecasters import (fhs_ewma, _ewma_vol_series, _gjr_innovation,
                            fit_stack_coeffs, fhs_stack, _expert_var_features)


def test_ewma_vol_series_is_causal_and_positive():
    rng = np.random.default_rng(0)
    r = rng.normal(scale=0.01, size=500)
    sig2 = _ewma_vol_series(r, lam=0.94, burn=100)
    assert (sig2 > 0).all()
    # forecast for day s must not depend on r[s] (causality): perturbing the
    # last return changes only the *next* forecast, not the current series.
    r2 = r.copy(); r2[-1] *= 5
    sig2b = _ewma_vol_series(r2, lam=0.94, burn=100)
    assert np.allclose(sig2, sig2b)  # same series; r[-1] only affects var_next


def test_fhs_standardised_residuals_have_unit_variance():
    # If the filter captures the clustering, z = r/sigma should be ~homoscedastic
    # with variance ~1. Build clustered data and check.
    rng = np.random.default_rng(1)
    n = 4000
    vol = 0.01 * np.ones(n)
    r = np.empty(n)
    r[0] = rng.normal(scale=vol[0])
    for i in range(1, n):
        vol[i] = np.sqrt(0.94 * vol[i - 1] ** 2 + 0.06 * r[i - 1] ** 2)
        r[i] = rng.normal(scale=vol[i])
    sig2 = _ewma_vol_series(r, lam=0.94, burn=100)
    z = r[100:] / np.sqrt(sig2[100:])
    assert 0.85 < z.std() < 1.15


def test_fhs_drift_zero_and_widens_with_horizon():
    rng = np.random.default_rng(2)
    r = rng.normal(scale=0.01, size=1500)
    levels = np.array([0.05, 0.5, 0.95])
    q = fhs_ewma(n_sims=20_000, seed=1)(r, T=10, levels=levels)
    assert abs(q[1, 0]) < 1e-3          # median ~ 0
    assert q[2, 9] > q[2, 0]            # widens with horizon


def test_fhs_scales_to_current_volatility_regime():
    # Same standardised shape, but a history ending calm vs ending turbulent
    # must yield a narrower vs wider one-day envelope.
    rng = np.random.default_rng(3)
    base = rng.normal(scale=0.01, size=1500)
    calm = base.copy(); calm[-30:] = rng.normal(scale=0.003, size=30)
    turb = base.copy(); turb[-30:] = rng.normal(scale=0.03, size=30)
    levels = np.array([0.05, 0.95])
    qc = fhs_ewma(n_sims=20_000, seed=1)(calm, T=1, levels=levels)
    qt = fhs_ewma(n_sims=20_000, seed=1)(turb, T=1, levels=levels)
    width_c = qc[1, 0] - qc[0, 0]
    width_t = qt[1, 0] - qt[0, 0]
    assert width_t > 2.0 * width_c


def test_gjr_innovation_asymmetry():
    up = _gjr_innovation(np.array([0.02]), gamma=1.0)[0]
    down = _gjr_innovation(np.array([-0.02]), gamma=1.0)[0]
    assert np.isclose(up, 0.02**2)
    assert np.isclose(down, 2.0 * 0.02**2)  # down move weighted (1+gamma)=2


def test_gjr_leverage_widens_after_down_move_only():
    # Two histories differing only in the SIGN of the last (large) return.
    # gamma>0 must widen the next-step envelope after the DOWN ending, but leave
    # the UP ending ~unchanged relative to gamma=0.
    rng = np.random.default_rng(4)
    base = rng.normal(scale=0.01, size=1500)
    down = base.copy(); down[-1] = -0.05
    up = base.copy(); up[-1] = +0.05
    levels = np.array([0.05, 0.95])

    def width(r, gamma):
        q = fhs_ewma(gamma=gamma, n_sims=30_000, seed=1)(r, T=1, levels=levels)
        return q[1, 0] - q[0, 0]

    # with leverage: a down ending gives a wider envelope than an equal up ending
    # (histories are identical except the last day, isolating the leverage term).
    assert width(down, 1.0) > 1.1 * width(up, 1.0)
    # without leverage the sign of the last day is irrelevant (r^2 symmetric).
    assert abs(width(down, 0.0) / width(up, 0.0) - 1.0) < 0.05


def test_expert_features_causal():
    rng = np.random.default_rng(5)
    r = rng.normal(scale=0.01, size=500)
    F = _expert_var_features(r)
    assert F.shape == (500, 6)
    assert (F[:, 0] == 1).all()                 # const column
    assert (F[60:, 1:] > 0).all()               # variances positive
    r2 = r.copy(); r2[-1] *= 5                   # perturb only the last return
    assert np.allclose(F[:-1], _expert_var_features(r2)[:-1])  # earlier rows unchanged


def test_fhs_stack_runs_and_widens():
    rng = np.random.default_rng(6)
    r = rng.normal(scale=0.01, size=1500)
    coeffs = fit_stack_coeffs(r[:800])
    q = fhs_stack(coeffs, n_sims=20_000, seed=1)(r, T=10, levels=np.array([0.05, 0.5, 0.95]))
    assert abs(q[1, 0]) < 1e-3                   # median ~ 0 (drift pinned)
    assert q[2, 9] > q[2, 0]                     # widens with horizon
