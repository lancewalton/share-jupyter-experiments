"""Cross-sectional portfolio primitives for the reversion-after-large-moves test.

Pure, tested helpers: market-neutral tail weights (long the biggest losers, short
the biggest winners, inverse-vol sized) and the usual performance statistics.
"""
from __future__ import annotations

import numpy as np


def tail_weights(signal: np.ndarray, vol: np.ndarray, frac: float = 0.2) -> np.ndarray:
    """Market- and dollar-neutral weights from a cross-sectional signal.

    Long the top ``frac`` of ``signal``, short the bottom ``frac``; within each
    side size inversely to ``vol`` (so risk, not notional, is balanced). Long leg
    sums to +1, short leg to -1 (gross exposure 2). Stocks with non-finite signal
    or non-positive vol get zero weight.
    """
    signal = np.asarray(signal, float); vol = np.asarray(vol, float)
    w = np.zeros(len(signal))
    ok = np.isfinite(signal) & np.isfinite(vol) & (vol > 0)
    idx = np.where(ok)[0]
    if len(idx) < 5:
        return w
    k = max(1, int(frac * len(idx)))
    order = idx[np.argsort(signal[idx])]
    lo, hi = order[:k], order[-k:]               # bottom = winners (short), top = losers (long)
    iv = 1.0 / vol
    w[hi] = iv[hi] / iv[hi].sum()
    w[lo] = -iv[lo] / iv[lo].sum()
    return w


def sharpe(r: np.ndarray, ppy: int = 252) -> float:
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    return 0.0 if sd == 0 or len(r) < 2 else float(r.mean() / sd * np.sqrt(ppy))


def max_drawdown(daily: np.ndarray) -> float:
    """Max drawdown of the equity curve implied by a series of daily log returns."""
    cum = np.exp(np.cumsum(np.asarray(daily, float)))
    peak = np.maximum.accumulate(cum)
    return float((cum / peak - 1.0).min())


def ann_return(daily: np.ndarray, ppy: int = 252) -> float:
    return float(np.nanmean(daily) * ppy)


def beta(r: np.ndarray, mkt: np.ndarray) -> float:
    r, mkt = np.asarray(r, float), np.asarray(mkt, float)
    m = np.isfinite(r) & np.isfinite(mkt)
    if m.sum() < 3:
        return np.nan
    c = np.cov(r[m], mkt[m])
    return float(c[0, 1] / c[1, 1]) if c[1, 1] > 0 else np.nan
