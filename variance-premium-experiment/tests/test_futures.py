"""Tests for the tradeable short-VIX-futures return."""
import numpy as np

from vrp.futures import equity_stats, short_vix_return


def test_carry_earned_in_flat_contango():
    # vol constant, curve in contango (VIX3M > VIX) -> short earns the carry daily
    vix = np.full(50, 20.0); vix3m = np.full(50, 23.0)
    sr = short_vix_return(vix, vix3m)
    expected = (np.log(23.0) - np.log(20.0)) / 63
    assert np.allclose(sr[1:], expected)
    assert expected > 0


def test_short_loses_when_vol_spikes():
    vix = np.array([15.0, 15.0, 30.0, 30.0])   # a doubling on day 2
    vix3m = np.array([17.0, 17.0, 17.0, 17.0])
    sr = short_vix_return(vix, vix3m)
    assert sr[2] < -0.5                          # ln(30/15)=0.69 loss, minus small carry


def test_causal_and_floored():
    vix = np.array([15.0, 60.0])                 # 4x spike -> ln=1.39, floored at -0.95
    sr = short_vix_return(vix, np.array([17.0, 17.0]))
    assert np.isnan(sr[0])                        # no return on day 0
    assert sr[1] == -0.95                         # floored


def test_equity_stats_positive_drift():
    rng = np.random.default_rng(0)
    d = rng.normal(0.001, 0.01, 3000)
    s = equity_stats(d)
    assert s["cagr"] > 0 and s["sharpe"] > 0 and s["mdd"] < 0
