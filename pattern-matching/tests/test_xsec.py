"""Tests for cross-sectional portfolio primitives."""
import numpy as np

from matching.xsec import beta, max_drawdown, sharpe, tail_weights


def test_tail_weights_long_losers_short_winners_neutral():
    sig = np.arange(10.0)                     # 9 = biggest loser signal (long), 0 = winner (short)
    w = tail_weights(sig, np.ones(10), frac=0.2)
    assert w[9] > 0 and w[8] > 0              # top signal -> long
    assert w[0] < 0 and w[1] < 0              # bottom signal -> short
    assert np.isclose(w[w > 0].sum(), 1.0)    # long leg +1
    assert np.isclose(w[w < 0].sum(), -1.0)   # short leg -1
    assert np.isclose(w.sum(), 0.0)           # market neutral


def test_tail_weights_inverse_vol_sizing():
    sig = np.arange(8.0)
    vol = np.ones(8); vol[7] = 2.0            # idx6,7 are the longs; 7 is high-vol
    w = tail_weights(sig, vol, frac=0.25)     # k=2 per side
    assert w[6] > 0 and w[7] > 0
    assert w[6] > w[7]                         # higher vol -> smaller weight
    assert np.isclose(w[w > 0].sum(), 1.0)


def test_tail_weights_ignores_bad_entries():
    sig = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0])
    vol = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    w = tail_weights(sig, vol, frac=0.34)
    assert w[0] == 0 and w[1] == 0            # nan signal / bad vol excluded


def test_sharpe_and_drawdown():
    r = np.full(252, 0.001)
    assert sharpe(r) > 100                     # zero-vol positive -> huge (constant); use noisy below
    rng = np.random.default_rng(0)
    r2 = rng.normal(0.0005, 0.01, 5000)
    assert 0.5 < sharpe(r2) < 1.2              # ~0.0005/0.01*sqrt(252)=0.79
    dd = max_drawdown(np.r_[0.1, -0.2, 0.05])
    assert dd < 0


def test_beta_of_scaled_market_is_scale():
    rng = np.random.default_rng(1)
    mkt = rng.normal(0, 0.01, 1000)
    assert abs(beta(0.5 * mkt, mkt) - 0.5) < 1e-9
