"""Tests for the walk-forward parameter selector used in the adaptive-MACD test."""
import numpy as np
import pandas as pd

from macd.adaptive import walk_forward


def _series(vals):
    idx = pd.date_range("2020-01-01", periods=len(vals), freq="B")
    return pd.Series(np.asarray(vals, dtype=float), index=idx)


def test_walk_forward_trades_the_dominant_param_out_of_sample():
    a = _series([0.001] * 100)          # always better
    b = _series([0.000] * 100)
    oos, picks = walk_forward({"A": a, "B": b}, train=50, step=10)
    # OOS span is everything after the first training window, all from A
    assert oos.index.equals(a.index[50:])
    assert np.allclose(oos.to_numpy(), 0.001)
    assert all(k == "A" for _, k in picks)


def test_walk_forward_switches_when_leadership_changes_and_is_causal():
    n = 120
    a = _series([0.001] * 60 + [-0.001] * (n - 60))
    b = _series([-0.001] * 60 + [0.001] * (n - 60))
    oos, picks = walk_forward({"A": a, "B": b}, train=30, step=10)
    keys = [k for _, k in picks]
    assert keys[0] == "A"               # A led in the first trailing window
    assert keys[-1] == "B"              # B leads by the end
    # causal: the choice for a block uses only the trailing window, so the switch
    # lags the true regime change (stale adaptation) rather than pre-empting it
    assert keys.index("B") > 3


def test_walk_forward_oos_series_is_contiguous_and_named_by_dates():
    a = _series(np.linspace(0.001, 0.002, 80))
    b = _series(np.linspace(0.002, 0.001, 80))
    oos, picks = walk_forward({"A": a, "B": b}, train=40, step=20)
    assert oos.notna().all()
    assert oos.index[0] == a.index[40]
    assert oos.index[-1] == a.index[-1]
