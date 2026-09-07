"""Realised volatility and variance-risk-premium primitives.

All volatilities are annualised and expressed in VIX-comparable *points* (a value
of 20.0 means 20 % annualised), so implied (VIX) and realised sit on one scale.
"""
from __future__ import annotations

import numpy as np

ANN = 252.0


def realised_vol(rets: np.ndarray) -> float:
    """Annualised realised volatility (points) of a block of daily log returns."""
    rets = np.asarray(rets, float)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:
        return np.nan
    return float(np.sqrt(ANN * np.mean(rets ** 2)) * 100.0)


def forward_realised_vol(rets: np.ndarray, H: int) -> np.ndarray:
    """``rv[t]`` = annualised realised vol over the H returns *after* index t.

    ``rets[i]`` is the return into day i, so the forward window for a decision at
    day t is ``rets[t+1 : t+1+H]`` -- strictly future, no look-ahead. NaN where
    the future is unavailable.
    """
    rets = np.asarray(rets, float)
    n = len(rets)
    out = np.full(n, np.nan)
    for t in range(n):
        if t + 1 + H <= n:
            out[t] = realised_vol(rets[t + 1: t + 1 + H])
    return out


def variance_premium(vix: np.ndarray, fwd_rv: np.ndarray):
    """Return (vol-point premium, variance-point premium).

    ``vol_premium = vix - fwd_rv`` (positive = implied richer than realised);
    ``var_premium = vix^2 - fwd_rv^2`` -- the quantity a variance-swap seller
    receives (they are paid the strike variance, pay realised variance).
    """
    vix, fwd = np.asarray(vix, float), np.asarray(fwd_rv, float)
    return vix - fwd, vix ** 2 - fwd ** 2
