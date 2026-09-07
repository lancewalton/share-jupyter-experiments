"""Tradeable short-VIX-futures return from the VIX term structure.

The variance-swap proxy ignored the roll. A real short-vol position is a short
constant-maturity ~1-month VIX future, whose daily return has two pieces:

  * carry  -- the future rolls DOWN the curve toward spot as it ages; in contango
              (VIX3M > VIX) the short EARNS this each day, ~ (lnVIX3M - lnVIX)/63;
  * MTM    -- the short LOSES when spot vol rises, ~ -Δln(VIX).

So ``short_return[t] = carry[t-1] - Δln(VIX)[t]`` -- carry known at the start of
the day (causal), mark-to-market realised over the day. This replicates the VXX /
SPVXSTR short-vol return (the real return source, roll included) from two free
FRED series, and reproduces the known facts: strong positive carry punctuated by
violent drawdowns (2008, Feb-2018, 2020).
"""
from __future__ import annotations

import numpy as np

ROLL_DAYS = 63  # trading days between the 1-month and 3-month points (~3 months)


def short_vix_return(vix: np.ndarray, vix3m: np.ndarray, roll_days: int = ROLL_DAYS,
                     floor: float = -0.95) -> np.ndarray:
    """Daily return of a short constant-1-month VIX-futures position (causal)."""
    lnv = np.log(np.asarray(vix, float))
    lnv3 = np.log(np.asarray(vix3m, float))
    carry = (lnv3 - lnv) / roll_days                 # per-day, known at day start
    dln = np.diff(lnv)                                # MTM over the day
    sr = np.full(len(vix), np.nan)
    sr[1:] = carry[:-1] - dln                         # short: earn carry, lose on vol rise
    return np.clip(sr, floor, None)                   # can't lose > ~100% unlevered


def equity_stats(daily: np.ndarray, ppy: int = 252) -> dict:
    """Sharpe / CAGR / max-drawdown / worst-day of a daily-return series."""
    d = np.asarray(daily, float); d = d[np.isfinite(d)]
    if len(d) < 2:
        return dict(sharpe=0.0, cagr=0.0, mdd=0.0, worst=0.0, n=len(d))
    eq = np.cumprod(1.0 + d)
    dd = eq / np.maximum.accumulate(eq) - 1.0
    cagr = eq[-1] ** (ppy / len(d)) - 1.0
    sd = d.std(ddof=1)
    return dict(sharpe=(d.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0,
                cagr=float(cagr), mdd=float(dd.min()), worst=float(d.min()), n=len(d))
