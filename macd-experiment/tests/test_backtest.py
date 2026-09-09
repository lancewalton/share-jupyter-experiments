"""Tests for per-name trade extraction, cost accounting, and daily metrics."""
import numpy as np
import pandas as pd

from macd.backtest import extract_trades, strategy_returns
from macd.metrics import sharpe, cagr, max_drawdown, hit_rate, mean_payoff


def _ret(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def test_strategy_returns_are_position_times_asset_return_minus_costs():
    # hold for two days, flat otherwise; a round-trip costs 2*cost (in and out)
    asset = _ret([0.01, 0.02, 0.03, 0.04, -0.05])
    pos = pd.Series([False, True, True, False, False], index=asset.index)
    r = strategy_returns(pos, asset, cost=0.001)
    # days 1,2 earn the asset return; entry charged on day 1, exit on day 3
    assert r.iloc[0] == 0.0
    assert abs(r.iloc[1] - (0.02 - 0.001)) < 1e-12
    assert abs(r.iloc[2] - 0.03) < 1e-12
    assert abs(r.iloc[3] - (0.0 - 0.001)) < 1e-12


def test_extract_trades_gives_one_return_per_holding_spell():
    asset = _ret([0.0, 0.10, 0.10, 0.0, -0.20, 0.0])
    pos = pd.Series([False, True, True, False, True, False], index=asset.index)
    trades = extract_trades(pos, asset)
    # two spells: (+0.10,+0.10) compounds to +0.21; (-0.20) is -0.20
    assert len(trades) == 2
    assert abs(trades[0] - (1.10 * 1.10 - 1)) < 1e-9
    assert abs(trades[1] - (-0.20)) < 1e-9


def test_hit_rate_and_mean_payoff_decompose_a_trade_list():
    trades = np.array([0.10, -0.05, 0.20, -0.05])
    assert abs(hit_rate(trades) - 0.5) < 1e-12
    assert abs(mean_payoff(trades) - 0.05) < 1e-12   # (0.10-0.05+0.20-0.05)/4


def test_daily_metrics_annualise_on_252():
    r = _ret([0.001] * 252)
    assert abs(cagr(r) - ((1.001 ** 252) - 1)) < 1e-6
    assert max_drawdown(_ret([0.1, -0.5, 0.1])) < 0
    # positive drift, tiny noise -> a high but finite Sharpe
    rng = np.random.default_rng(0)
    noisy = _ret(0.0005 + rng.normal(0, 0.01, 500))
    assert 0.0 < sharpe(noisy) < 5.0
