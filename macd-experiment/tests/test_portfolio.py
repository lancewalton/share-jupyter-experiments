"""Tests for equal-weight portfolio aggregation of MACD timing vs buy-and-hold."""
import numpy as np
import pandas as pd

from macd.backtest import portfolio


def _frame(mat):
    idx = pd.date_range("2020-01-01", periods=len(mat), freq="B")
    return pd.DataFrame(np.asarray(mat, dtype=float), index=idx, columns=["A", "B"])


def test_buy_and_hold_is_the_equal_weight_mean_of_eligible_returns():
    returns = _frame([[0.02, 0.04], [0.00, 0.10], [-0.02, 0.06]])
    positions = pd.DataFrame(True, index=returns.index, columns=returns.columns)
    out = portfolio(positions, returns, cost=0.0)
    # always invested -> strategy equals buy-and-hold equals the cross-name mean
    assert np.allclose(out["bnh"].to_numpy(), [0.03, 0.05, 0.02])
    assert np.allclose(out["strat"].to_numpy(), out["bnh"].to_numpy())


def test_flat_names_sit_in_cash_and_do_not_earn():
    returns = _frame([[0.02, 0.04], [0.10, 0.10], [0.10, 0.10]])
    # A stays flat throughout, B is fully invested
    positions = pd.DataFrame(
        [[False, True], [False, True], [False, True]],
        index=returns.index, columns=returns.columns,
    )
    out = portfolio(positions, returns, cost=0.0)
    # equal-weight over two eligible names: only B's leg contributes
    assert np.allclose(out["strat"].to_numpy(), [0.02, 0.05, 0.05])


def test_eligibility_ignores_names_with_no_data_that_day():
    # A is not yet listed on day 0 (NaN return) -> day 0 weights only B
    returns = _frame([[np.nan, 0.10], [0.02, 0.10], [0.02, 0.10]])
    positions = pd.DataFrame(True, index=returns.index, columns=returns.columns)
    out = portfolio(positions, returns, cost=0.0)
    assert abs(out["bnh"].iloc[0] - 0.10) < 1e-12          # only B counts on day 0
    assert abs(out["bnh"].iloc[1] - 0.06) < 1e-12          # both count later


def test_concentrated_mode_puts_all_capital_in_the_signalling_names():
    # A flat, B long -> concentrated book is 100% B (no idle cash), not 50/50
    returns = _frame([[0.02, 0.04], [0.10, 0.10], [0.10, 0.10]])
    positions = pd.DataFrame(
        [[False, True], [False, True], [True, True]],
        index=returns.index, columns=returns.columns,
    )
    out = portfolio(positions, returns, cost=0.0, concentrate=True)
    # day0: only B long -> weight 1.0 on B -> 0.04
    # day2: both long -> equal-weight -> mean(0.10,0.10)=0.10
    assert abs(out["strat"].iloc[0] - 0.04) < 1e-12
    assert abs(out["strat"].iloc[2] - 0.10) < 1e-12


def test_concentrated_mode_sits_in_cash_when_nothing_signals():
    returns = _frame([[0.05, 0.05], [0.05, 0.05]])
    positions = pd.DataFrame(False, index=returns.index, columns=returns.columns)
    out = portfolio(positions, returns, cost=0.0, concentrate=True)
    assert out["strat"].iloc[0] == 0.0 and out["strat"].iloc[1] == 0.0


def test_per_hold_day_counts_invested_name_days():
    returns = _frame([[0.05, 0.05], [0.05, 0.05]])
    positions = pd.DataFrame(
        [[True, False], [True, True]],
        index=returns.index, columns=returns.columns,
    )
    out = portfolio(positions, returns, cost=0.0)
    # invested name-days: day0 has 1 (A), day1 has 2 (A,B) -> 3 hold-days
    assert out["hold_days"] == 3
