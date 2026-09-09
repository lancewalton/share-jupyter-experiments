"""Tests for the serial-structure-destroying surrogate used in the permutation null."""
import numpy as np
import pandas as pd

from macd.surrogate import permute_within_columns


def _frame(mat):
    idx = pd.date_range("2020-01-01", periods=len(mat), freq="B")
    return pd.DataFrame(np.asarray(mat, dtype=float), index=idx, columns=["A", "B"])


def test_preserves_each_columns_values_and_nan_positions():
    df = _frame([[0.01, np.nan], [np.nan, 0.02], [0.03, 0.04], [0.05, 0.06]])
    out = permute_within_columns(df, np.random.default_rng(0))
    for col in df.columns:
        assert np.array_equal(df[col].isna().to_numpy(), out[col].isna().to_numpy())
        assert sorted(df[col].dropna()) == sorted(out[col].dropna())


def test_shuffles_the_time_order():
    # a strictly ordered column should almost surely be reordered
    df = _frame([[float(i), float(i)] for i in range(50)])
    out = permute_within_columns(df, np.random.default_rng(1))
    assert not np.array_equal(df["A"].to_numpy(), out["A"].to_numpy())


def test_columns_are_permuted_independently():
    df = _frame([[float(i), float(i)] for i in range(50)])
    out = permute_within_columns(df, np.random.default_rng(2))
    assert not np.array_equal(out["A"].to_numpy(), out["B"].to_numpy())
