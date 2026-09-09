"""Tests for the standard MACD indicator and its causal long-state signal."""
import numpy as np
import pandas as pd

from macd.indicator import macd, long_state


def _series(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def test_macd_is_zero_for_a_constant_series():
    s = _series([100.0] * 60)
    line, signal, hist = macd(s, fast=12, slow=26, signal=9)
    # once both EMAs have warmed up they coincide, so every component is ~0
    assert abs(line.iloc[-1]) < 1e-9
    assert abs(signal.iloc[-1]) < 1e-9
    assert abs(hist.iloc[-1]) < 1e-9


def test_macd_line_is_positive_on_a_rising_ramp():
    # a strictly rising series: the fast EMA sits above the slow EMA
    s = _series(np.arange(100.0, 200.0))
    line, signal, hist = macd(s, fast=12, slow=26, signal=9)
    assert line.iloc[-1] > 0
    # histogram is macd line minus its signal line, by construction
    assert np.allclose((line - signal).to_numpy(), hist.to_numpy(), equal_nan=True)


def test_long_state_follows_the_signal_line_crossover():
    # fall for 40 bars then rise for 40: long only after the fast crosses up
    down = np.linspace(200.0, 100.0, 40)
    up = np.linspace(100.0, 300.0, 40)
    s = _series(np.concatenate([down, up]))
    state = long_state(s, fast=12, slow=26, signal=9)
    assert not state.iloc[35]      # still falling
    assert state.iloc[-1]          # firmly rising by the end
    assert state.dtype == bool


def test_long_state_is_causal_no_lookahead():
    # perturbing the tail of the series must not change any earlier position
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.normal(0, 1, 200))
    s = _series(base)
    state = long_state(s, fast=12, slow=26, signal=9)

    bumped = base.copy()
    bumped[150:] += 50.0           # change only from bar 150 onward
    state2 = long_state(_series(bumped), fast=12, slow=26, signal=9)

    # positions up to and including bar 150 depend only on closes before them,
    # so they must be identical despite the future perturbation
    assert state.iloc[:151].equals(state2.iloc[:151])
