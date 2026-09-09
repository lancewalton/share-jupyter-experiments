"""Tests for the volatility-regime gate applied to the MACD long signal."""
import numpy as np
import pandas as pd

from macd.indicator import macd_cross_state, long_state, vol_gate, gated_long_state


def _series(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def _frame(mat):
    idx = pd.date_range("2020-01-01", periods=len(mat), freq="B")
    return pd.DataFrame(np.asarray(mat, dtype=float).reshape(len(mat), -1), index=idx,
                        columns=["A"])


def test_cross_state_is_the_unshifted_source_of_long_state():
    s = _series(np.r_[np.linspace(200, 100, 40), np.linspace(100, 300, 40)])
    raw = macd_cross_state(s)
    assert long_state(s).equals(raw.shift(1, fill_value=False))
    assert raw.iloc[-1]              # rising at the end -> MACD above signal


def test_vol_gate_opens_in_the_high_volatility_stretch():
    # a long calm history then a sharp volatile burst (25x): the gate is shut deep
    # in the calm region and open in the burst
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.002, 300)
    wild = rng.normal(0, 0.05, 30)
    r = _frame(np.concatenate([calm, wild]))
    gate = vol_gate(r, window=20, q=0.8, q_window=200)
    assert not bool(gate["A"].iloc[150])     # calm: below its trailing 80th pct
    assert bool(gate["A"].iloc[-1])          # burst: far above


def test_zero_percentile_gate_recovers_the_ungated_baseline():
    rng = np.random.default_rng(1)
    prices = _frame(100 + np.cumsum(rng.normal(0, 1, 300)))
    returns = prices.pct_change()
    gated = gated_long_state(prices, returns, vol_window=20, vol_q=0.0, q_window=100)
    base = long_state(prices)
    # q=0 -> threshold is the trailing minimum, vol always clears it, so once the
    # gate is defined the gated positions match the ungated baseline exactly
    tail = base.index[130:]
    assert gated.loc[tail, "A"].equals(base.loc[tail, "A"])


def test_gated_long_state_is_causal():
    rng = np.random.default_rng(2)
    base = 100 + np.cumsum(rng.normal(0, 1, 250))
    prices = _frame(base)
    returns = prices.pct_change()
    g = gated_long_state(prices, returns, vol_window=20, vol_q=0.5, q_window=100)

    bumped = base.copy()
    bumped[180:] += 40.0
    p2 = _frame(bumped)
    g2 = gated_long_state(p2, p2.pct_change(), vol_window=20, vol_q=0.5, q_window=100)
    assert g.iloc[:181].equals(g2.iloc[:181])
