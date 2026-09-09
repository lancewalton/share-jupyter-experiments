"""Tests for the point-in-time top-N liquidity universe mask."""
import numpy as np
import pandas as pd

from macd.data import top_n_mask


def _frame(mat):
    idx = pd.date_range("2020-01-01", periods=len(mat), freq="B")
    cols = list("ABCD")[: len(mat[0])]
    return pd.DataFrame(np.asarray(mat, dtype=float), index=idx, columns=cols)


def test_keeps_the_n_most_liquid_names_each_day():
    liq = _frame([[10, 5, 8, 1], [1, 9, 2, 7]])
    mask = top_n_mask(liq, n=2)
    # day 0: A(10), C(8) win; day 1: B(9), D(7) win
    assert list(mask.iloc[0][mask.iloc[0]].index) == ["A", "C"]
    assert list(mask.iloc[1][mask.iloc[1]].index) == ["B", "D"]


def test_names_without_liquidity_are_never_eligible():
    liq = _frame([[np.nan, 5.0, 3.0]])
    mask = top_n_mask(liq, n=3)
    assert not mask.iloc[0]["A"]         # NaN liquidity -> excluded even under quota
    assert mask.iloc[0]["B"] and mask.iloc[0]["C"]


def test_returns_boolean_frame_aligned_to_input():
    liq = _frame([[1, 2, 3, 4]])
    mask = top_n_mask(liq, n=2)
    assert mask.shape == liq.shape
    assert mask.dtypes.eq(bool).all()
