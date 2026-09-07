"""Honest evaluation: fair walk-forward, skill vs a baseline, shuffled control."""
import numpy as np
import pandas as pd

from .models import (FixedFIR, AdaptiveFIR, MetaAdaptiveFIR, IDBD, MetaIDBD,
                     run_online, run_idbd)


def rolling_mean_baseline(v, window):
    return pd.Series(v).shift(1).rolling(window, min_periods=1).mean().to_numpy()


def skill(actual, pred, baseline):
    """Correlation, RMSE, R^2 vs the target mean, and skill vs a baseline predictor."""
    m = np.isfinite(actual) & np.isfinite(pred) & np.isfinite(baseline)
    a, p, b = actual[m], pred[m], baseline[m]
    ss_res = np.sum((a - p) ** 2)
    return pd.Series({
        "Corr": np.corrcoef(a, p)[0, 1] if np.std(p) > 0 else np.nan,
        "RMSE": np.sqrt(ss_res / len(a)),
        "R2_vs_mean": 1 - ss_res / np.sum((a - a.mean()) ** 2),
        "Skill_vs_base": 1 - ss_res / np.sum((a - b) ** 2),
        "N": len(a),
    })


def compare(v, n_lags=20, roll=20, ad_lr=0.1, meta_rate=0.05,
            idbd_theta=0.1, metaidbd_mm=0.1):
    """Fit Fixed on the first half; score every model on the identical second half."""
    v = np.asarray(v, float)
    n = len(v)
    split = n // 2
    idx = slice(split, n)
    base = rolling_mean_baseline(v, roll)

    fx = FixedFIR(n_lags); fx.fit(v[:split])
    fpred = np.full(n, np.nan)
    for t in range(split, n):
        fpred[t] = fx.predict(v[t - n_lags:t][::-1])

    apred, _ = run_online(v, AdaptiveFIR(n_lags, lr=ad_lr))
    mpred, _ = run_online(v, MetaAdaptiveFIR(n_lags, lr=ad_lr, meta_rate=meta_rate))
    ipred, _, _ = run_idbd(v, lambda k: IDBD(k, theta=idbd_theta), n_lags, split)
    m3pred, _, _ = run_idbd(v, lambda k: MetaIDBD(k, meta_meta=metaidbd_mm), n_lags, split)

    return pd.DataFrame({
        "RollingMean":     skill(v[idx], base[idx], base[idx]),
        "Fixed_L0":        skill(v[idx], fpred[idx], base[idx]),
        "Adaptive_L1":     skill(v[idx], apred[idx], base[idx]),
        "MetaTrend_L2":    skill(v[idx], mpred[idx], base[idx]),
        "IDBD_L2":         skill(v[idx], ipred[idx], base[idx]),
        "MetaIDBD_L3":     skill(v[idx], m3pred[idx], base[idx]),
    }).T


def shuffled_returns(r, seed=0):
    rng = np.random.default_rng(seed)
    rs = np.asarray(r, float).copy()
    rng.shuffle(rs)
    return rs
