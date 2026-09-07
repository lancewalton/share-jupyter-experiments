"""Variance-swap seller P&L and performance stats.

A 1-month variance-swap seller receives the strike variance and pays realised
variance. Per unit of vega notional the seller's P&L is ``(K^2 - RV^2)/(2K)``
with ``K = VIX`` -- in vol points, and carrying the brutal convex left tail
(when realised >> implied the loss explodes). This is the honest instrument for
the variance risk premium.
"""
from __future__ import annotations

import numpy as np

ANN_M = 12.0  # monthly (21-trading-day) periods per year


def varswap_pnl(vix: np.ndarray, rv: np.ndarray) -> np.ndarray:
    """Short-variance seller P&L per unit vega (vol points): (VIX^2 - RV^2)/(2 VIX)."""
    vix, rv = np.asarray(vix, float), np.asarray(rv, float)
    return (vix ** 2 - rv ** 2) / (2.0 * vix)


def sharpe(pnl: np.ndarray, ppy: float = ANN_M) -> float:
    p = np.asarray(pnl, float); p = p[np.isfinite(p)]
    sd = p.std(ddof=1)
    return 0.0 if sd == 0 or len(p) < 2 else float(p.mean() / sd * np.sqrt(ppy))


def max_drawdown(pnl: np.ndarray) -> float:
    """Max drawdown of the cumulative-P&L curve (additive P&L, in the pnl units)."""
    cum = np.cumsum(np.asarray(pnl, float))
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def stats(pnl: np.ndarray) -> dict:
    p = np.asarray(pnl, float)[np.isfinite(pnl)]
    return dict(sharpe=sharpe(p), mean=float(p.mean()), worst=float(p.min()),
                mdd=max_drawdown(p), n=len(p))
