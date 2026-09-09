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
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    cost: float = 0.0,
    concentrate: bool = False,
) -> dict:
    """Portfolio return of the MACD book each day, vs equal-weight buy-and-hold.

    A name is eligible on a day when it has a return (listed and liquid). The book
    holds only names whose position is True. With ``concentrate`` the capital is
    equal-weighted across just the currently-long names (fully invested; cash only
    when none signal) — the portfolio approach, so idle names never dilute the book.
    Without it, weights are 1/(eligible count), leaving cash in the non-signalling
    names. Buy-and-hold weights the whole eligible universe equally. Turnover cost is
    charged on the change in each name's actual weight.
    """
    eligible = returns.notna()
    invested = positions.astype(bool) & eligible
    inv = invested.astype(float)

    denom = inv.sum(axis=1) if concentrate else eligible.sum(axis=1).astype(float)
    denom = denom.replace(0, np.nan)
    weights = inv.div(denom, axis=0).fillna(0.0)

    turnover = weights.diff()
    turnover.iloc[0] = weights.iloc[0]
    strat = (weights * returns.fillna(0.0)).sum(axis=1) - turnover.abs().sum(axis=1) * cost
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
