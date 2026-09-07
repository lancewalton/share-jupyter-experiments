"""Causal HAR-style forward-volatility forecaster (the regime signal).

Predict the next-H-day realised volatility from trailing realised vols at several
horizons (1 / 5 / 22 days -- the HAR structure) plus the VIX level, fit on an
expanding window with a strict causal cut: at decision day t only training pairs
whose forward window has fully closed (s + H <= t) are used, so nothing leaks.
"""
from __future__ import annotations

import numpy as np

ANN = 252.0


def trailing_rv(rets: np.ndarray, windows=(1, 5, 22)) -> np.ndarray:
    """(n, len(windows)) trailing realised vol (annualised points) at each day."""
    rets = np.asarray(rets, float)
    r2 = rets ** 2
    n = len(rets)
    out = np.full((n, len(windows)), np.nan)
    for j, w in enumerate(windows):
        c = np.cumsum(np.where(np.isfinite(r2), r2, 0.0))
        for t in range(w - 1, n):
            seg = r2[t - w + 1: t + 1]
            if np.isfinite(seg).all():
                out[t, j] = np.sqrt(ANN * seg.mean()) * 100.0
    return out


def har_walk_forward(feat: np.ndarray, target: np.ndarray, H: int,
                     min_train: int = 252) -> np.ndarray:
    """Expanding-window OLS forecast of ``target`` from ``feat``, strictly causal.

    ``target[s]`` is the forward realised vol observed only by day ``s + H``; at
    decision day t we train on ``s <= t - H`` (forward fully closed) and predict
    ``feat[t]``.
    """
    feat = np.asarray(feat, float); target = np.asarray(target, float)
    n = len(target)
    X = np.c_[np.ones(n), feat]
    pred = np.full(n, np.nan)
    for t in range(min_train + H, n):
        s = np.arange(0, t - H + 1)
        m = np.isfinite(target[s]) & np.isfinite(feat[s]).all(1)
        if m.sum() < min_train or not np.isfinite(feat[t]).all():
            continue
        b, *_ = np.linalg.lstsq(X[s][m], target[s][m], rcond=None)
        pred[t] = X[t] @ b
    return pred
