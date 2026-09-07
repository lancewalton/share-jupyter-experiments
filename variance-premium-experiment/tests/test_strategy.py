"""Tests for the variance-swap P&L, stats, and causal HAR forecaster."""
import numpy as np

from vrp.forecast import har_walk_forward, trailing_rv
from vrp.strategy import max_drawdown, sharpe, varswap_pnl


def test_varswap_pnl_signs_and_tail():
    # calm month: realised below implied -> seller profits
    assert varswap_pnl(np.array([20.0]), np.array([12.0]))[0] > 0
    # vol spike: realised far above implied -> large loss (convex tail)
    calm = varswap_pnl(np.array([15.0]), np.array([10.0]))[0]
    spike = varswap_pnl(np.array([15.0]), np.array([80.0]))[0]
    assert spike < -100 and spike < -20 * calm


def test_sharpe_and_drawdown():
    rng = np.random.default_rng(0)
    p = rng.normal(1.0, 5.0, 2000)
    assert 0.2 < sharpe(p, ppy=12) < 1.2
    assert max_drawdown(np.array([1.0, -3.0, 1.0])) < 0


def test_trailing_rv_is_causal_and_reasonable():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, 500)
    rv = trailing_rv(r, windows=(22,))
    # ~ 0.01*sqrt(252)*100 ~ 15.9 points, and undefined before the window fills
    assert np.isnan(rv[10, 0])
    assert abs(np.nanmean(rv[100:, 0]) - 0.01 * np.sqrt(252) * 100) < 3


def test_har_forecast_has_no_lookahead():
    # target known only H ahead; forecaster must not use future rows
    n = 800
    feat = np.cumsum(np.random.default_rng(2).normal(0, 1, (n, 2)), axis=0)
    target = np.r_[feat[10:, 0], [np.nan] * 10]        # target leads feat by 10
    pred = har_walk_forward(feat, target, H=10, min_train=100)
    assert np.all(np.isnan(pred[:110]))                # nothing before min_train+H
    assert np.isfinite(pred[300:]).any()               # produces forecasts later
