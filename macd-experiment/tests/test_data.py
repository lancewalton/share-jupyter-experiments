"""Tests for winsorising bad adjusted-close ticks out of the daily log returns."""
import numpy as np
import pandas as pd

from macd.data import winsorise_log_returns


def _frame(mat):
    idx = pd.date_range("2020-01-01", periods=len(mat), freq="B")
    return pd.DataFrame(np.asarray(mat, dtype=float), index=idx, columns=["A", "B"])


def test_extreme_single_day_moves_are_capped():
    # a +2.0 / -2.0 log-return spike pair (a bad tick and its reversal) is clipped
    lr = _frame([[0.01, 0.02], [2.0, -0.01], [-2.0, 0.00]])
    out = winsorise_log_returns(lr, cap=0.6)
    assert out.loc[out.index[1], "A"] == 0.6
    assert out.loc[out.index[2], "A"] == -0.6


def test_ordinary_returns_are_untouched():
    lr = _frame([[0.01, -0.02], [0.03, 0.05], [-0.04, 0.02]])
    out = winsorise_log_returns(lr, cap=0.6)
    assert np.allclose(out.to_numpy(), lr.to_numpy())


def test_nans_are_preserved():
    lr = _frame([[np.nan, 0.02], [0.03, np.nan]])
    out = winsorise_log_returns(lr, cap=0.6)
    assert np.isnan(out.iloc[0, 0])
    assert np.isnan(out.iloc[1, 1])
