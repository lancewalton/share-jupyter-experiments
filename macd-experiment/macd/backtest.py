"""Turn a per-name position series into daily returns and discrete trade P&Ls.

Positions are already causal (see ``indicator.long_state``): the value on day *t* is
the position held that day. A daily return is ``position * asset_return`` less a
turnover cost charged whenever the position changes. A trade is one maximal holding
spell; its P&L is the compounded gross asset return over the spell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio(
    positions: pd.DataFrame, returns: pd.DataFrame, cost: float = 0.0
) -> dict:
    """Equal-weight the eligible cross-section each day; MACD timing vs buy-and-hold.

    A name is eligible on a day when it has a return (it is listed and liquid).
    Buy-and-hold is always invested; the strategy is invested only where its
    position is True, sitting in cash otherwise. Both weight eligible names equally,
    so they share one benchmark. A turnover cost is charged when a name's invested
    state changes.
    """
    eligible = returns.notna()
    n_elig = eligible.sum(axis=1).replace(0, np.nan)
    weight = 1.0 / n_elig

    invested = (positions.astype(bool) & eligible)
    inv = invested.astype(float)
    turnover = inv.diff()
    turnover.iloc[0] = inv.iloc[0]

    name_pnl = inv * returns.fillna(0.0) - turnover.abs() * cost
    strat = name_pnl.sum(axis=1) * weight
    bnh = returns.mean(axis=1)

    return {
        "strat": strat.astype(float),
        "bnh": bnh.astype(float),
        "hold_days": int(inv.to_numpy().sum()),
    }


def strategy_returns(pos: pd.Series, asset: pd.Series, cost: float = 0.0) -> pd.Series:
    pos = pos.astype(float)
    turnover = pos.diff()
    turnover.iloc[0] = pos.iloc[0]
    return pos * asset - turnover.abs() * cost


def extract_trades(pos: pd.Series, asset: pd.Series) -> list[float]:
    pos = pos.astype(bool).to_numpy()
    a = asset.to_numpy()
    trades: list[float] = []
    growth = 1.0
    holding = False
    for held, ret in zip(pos, a):
        if held:
            growth *= 1.0 + ret
            holding = True
        elif holding:
            trades.append(growth - 1.0)
            growth = 1.0
            holding = False
    if holding:
        trades.append(growth - 1.0)
    return trades
