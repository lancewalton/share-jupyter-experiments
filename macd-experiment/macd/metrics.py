"""Daily-frequency performance metrics and trade-level decomposition.

Daily returns annualise on 252 trading days. The trade-level split of an edge into
hit-rate (how often trades win) versus mean payoff (average trade return) is the
programme's key diagnostic: direction filters tend to move the former, not the latter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PPY = 252


def sharpe(r: pd.Series, ppy: int = PPY) -> float:
    r = r.dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0


def cagr(r: pd.Series, ppy: int = PPY) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    return float((1 + r).prod() ** (ppy / len(r)) - 1)


def max_drawdown(r: pd.Series) -> float:
    r = r.dropna()
    cum = (1 + r).cumprod()
    return float((cum / cum.cummax() - 1).min())


def hit_rate(trades) -> float:
    t = np.asarray(trades, dtype=float)
    return float((t > 0).mean()) if len(t) else float("nan")


def mean_payoff(trades) -> float:
    t = np.asarray(trades, dtype=float)
    return float(t.mean()) if len(t) else float("nan")
