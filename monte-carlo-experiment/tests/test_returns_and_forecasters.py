"""Tests for returns indexing and the drift-zero baselines."""
import numpy as np

from mc.returns import log_returns, realised_cumulative, history_upto
from mc.forecasters import (
    gaussian_ewma,
    empirical_scaled,
    bootstrap_iid,
    bootstrap_age_weighted,
    bootstrap_stationary,
    bootstrap_vol_scaled,
    combine_average,
)


def test_log_returns_basic():
    p = np.array([100.0, 110.0, 99.0])
    r = log_returns(p)
    assert np.allclose(r, [np.log(1.1), np.log(99 / 110)])


def test_realised_cumulative_indexing():
    r = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # origin t=1 -> future r[2],r[3],r[4] = 0.3,0.4,0.5; cumsum 0.3,0.7,1.2
    out = realised_cumulative(r, t=1, T=3)
    assert np.allclose(out, [0.3, 0.7, 1.2])


def test_history_upto_excludes_future():
    r = np.arange(10.0)
    assert np.allclose(history_upto(r, 4), [0, 1, 2, 3, 4])


def test_realised_cumulative_raises_without_future():
    r = np.array([0.1, 0.2, 0.3])
    try:
        realised_cumulative(r, t=1, T=3)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_gaussian_ewma_centred_and_sqrt_scaled():
    rng = np.random.default_rng(0)
    r = rng.normal(scale=0.01, size=500)
    levels = np.array([0.1, 0.5, 0.9])
    q = gaussian_ewma()(r, T=4, levels=levels)
    # median (level 0.5) is zero at every horizon -> drift pinned to zero.
    assert np.allclose(q[1], 0.0, atol=1e-12)
    # symmetric: 0.1 and 0.9 quantiles mirror.
    assert np.allclose(q[0], -q[2])
    # sqrt-scaling: horizon-4 width is 2x horizon-1 width.
    assert np.allclose(q[2, 3] / q[2, 0], 2.0)


def test_empirical_scaled_drift_zero_and_monotone():
    rng = np.random.default_rng(2)
    r = rng.normal(loc=0.005, scale=0.01, size=300)  # nonzero mean on purpose
    levels = np.array([0.05, 0.5, 0.95])
    # use a symmetric level pair so the *mean* being pinned is testable.
    levels = np.array([0.05, 0.5, 0.95])
    q = empirical_scaled(N=250)(r, T=2, levels=levels)
    # drift (mean) pinned to zero: median only ~0 up to sample skew, not exact.
    assert abs(q[1, 0]) < 5e-3
    # quantiles increasing in level at each horizon.
    assert q[0, 0] < q[1, 0] < q[2, 0]


def test_bootstrap_iid_matches_empirical_at_horizon_1():
    # At h=1 the bootstrap is just the empirical 1-day quantile distribution,
    # so it must agree with empirical_scaled(h=1) up to Monte-Carlo noise.
    rng = np.random.default_rng(3)
    r = rng.standard_t(df=5, size=800) * 0.01  # fat-tailed daily returns
    levels = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    qb = bootstrap_iid(N=500, n_sims=40_000, seed=1)(r, T=1, levels=levels)[:, 0]
    qe = empirical_scaled(N=500)(r, T=1, levels=levels)[:, 0]
    assert np.allclose(qb, qe, atol=2e-3)


def test_bootstrap_iid_centred_and_widens_with_horizon():
    rng = np.random.default_rng(4)
    r = rng.normal(scale=0.01, size=600)
    levels = np.array([0.05, 0.5, 0.95])
    q = bootstrap_iid(N=500, n_sims=20_000, seed=2)(r, T=10, levels=levels)
    assert abs(q[1, 0]) < 1e-3          # median ~ 0 (drift pinned)
    assert q[2, 9] > q[2, 0]            # 95% envelope widens with horizon
    # IID convolution should roughly follow sqrt-T for the upper quantile.
    assert 2.5 < q[2, 9] / q[2, 0] < 3.7  # sqrt(10) ~= 3.16


def _width(q, level_lo=0, level_hi=-1, h=10):
    return q[level_hi, h - 1] - q[level_lo, h - 1]


def test_age_weighting_tracks_recent_volatility():
    # Window: calm old half, turbulent recent half. Age-weighting should give a
    # WIDER envelope than uniform because it leans on the recent turbulent data.
    rng = np.random.default_rng(5)
    old = rng.normal(scale=0.005, size=400)
    recent = rng.normal(scale=0.02, size=400)
    r = np.concatenate([old, recent])
    levels = np.array([0.05, 0.95])
    q_uni = bootstrap_iid(N=800, n_sims=20_000, seed=1)(r, T=10, levels=levels)
    q_age = bootstrap_age_weighted(N=800, half_life=40, n_sims=20_000, seed=1)(
        r, T=10, levels=levels
    )
    assert _width(q_age) > 1.3 * _width(q_uni)


def test_stationary_block_inflates_tail_on_autocorrelated_data():
    # Positively autocorrelated returns: consecutive moves reinforce, so longer
    # blocks should widen the multi-day envelope vs IID (block=1).
    rng = np.random.default_rng(6)
    eps = rng.normal(scale=0.01, size=3000)
    r = np.empty_like(eps)
    r[0] = eps[0]
    for i in range(1, len(eps)):
        r[i] = 0.5 * r[i - 1] + eps[i]  # AR(1), phi=0.5
    levels = np.array([0.025, 0.975])
    q_iid = bootstrap_stationary(N=2000, mean_block=1, n_sims=30_000, seed=2)(
        r, T=10, levels=levels
    )
    q_blk = bootstrap_stationary(N=2000, mean_block=15, n_sims=30_000, seed=2)(
        r, T=10, levels=levels
    )
    assert _width(q_blk) > 1.15 * _width(q_iid)


def test_stationary_block_neutral_on_iid_data():
    # On genuinely IID data, block length must NOT materially change the envelope.
    rng = np.random.default_rng(7)
    r = rng.normal(scale=0.01, size=3000)
    levels = np.array([0.025, 0.975])
    q_iid = bootstrap_stationary(N=2000, mean_block=1, n_sims=30_000, seed=3)(
        r, T=10, levels=levels
    )
    q_blk = bootstrap_stationary(N=2000, mean_block=15, n_sims=30_000, seed=3)(
        r, T=10, levels=levels
    )
    ratio = _width(q_blk) / _width(q_iid)
    assert 0.9 < ratio < 1.1


def test_vol_scaled_widens_when_recent_vol_exceeds_window():
    # Long calm window, turbulent recent tail: alpha=1 should scale the envelope
    # up towards the recent vol; alpha=0 should ignore it.
    rng = np.random.default_rng(8)
    calm = rng.normal(scale=0.005, size=980)
    recent = rng.normal(scale=0.02, size=20)
    r = np.concatenate([calm, recent])
    levels = np.array([0.05, 0.95])
    q0 = bootstrap_vol_scaled(N_shape=1000, B_vol=20, alpha=0.0, seed=1)(r, 10, levels)
    q1 = bootstrap_vol_scaled(N_shape=1000, B_vol=20, alpha=1.0, seed=1)(r, 10, levels)
    assert _width(q1) > 2.0 * _width(q0)  # recent ~4x window vol -> big widening


def test_combine_average_between_components():
    rng = np.random.default_rng(9)
    r = rng.normal(scale=0.01, size=1200)
    levels = np.array([0.05, 0.5, 0.95])
    fa = bootstrap_iid(N=60, n_sims=20_000, seed=1)
    fb = bootstrap_iid(N=1000, n_sims=20_000, seed=1)
    qa = fa(r, 10, levels); qb = fb(r, 10, levels)
    qc = combine_average([fa, fb])(r, 10, levels)
    # averaged upper quantile lies between the two components.
    lo, hi = sorted([qa[2, 9], qb[2, 9]])
    assert lo - 1e-9 <= qc[2, 9] <= hi + 1e-9
