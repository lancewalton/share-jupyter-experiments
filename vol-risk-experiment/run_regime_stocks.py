"""Task 2: regime object per stock -- where does regime-timing actually help?

FTSE (index): turbulent regime preceded REBOUNDS (turbulent forward return > calm),
so de-risking into turbulence HURT. Is the sign different stock-by-stock? Fit a
2-state HMM per stock, and (causally) compare forward return in turbulent vs calm,
and test whether exiting a stock while it is turbulent beats buy-and-hold.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from mc.data import load_close
from mc.returns import log_returns

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
SCALE, H, ANN = 100.0, 20, 252


def _filter(x, means, vars_, trans, start):
    n, k = len(x), len(means)
    lp = -0.5 * (np.log(2 * np.pi * vars_) + (x[:, None] - means) ** 2 / vars_)
    pdf = np.exp(lp - lp.max(axis=1, keepdims=True))
    a = np.empty((n, k)); a[0] = start * pdf[0]; a[0] /= a[0].sum()
    for t in range(1, n):
        a[t] = pdf[t] * (a[t - 1] @ trans); a[t] /= a[t].sum()
    return a


def _sharpe(x):
    x = x[np.isfinite(x)]
    return x.mean() / x.std() * np.sqrt(ANN) if x.std() > 0 else 0.0


def main():
    rows = []
    for pat in DIRS:
        for path in sorted(glob.glob(pat)):
            try:
                r = log_returns(load_close(path).to_numpy())
            except Exception:
                continue
            if len(r) < 1500 or np.abs(r).max() > 0.6:
                continue
            split = len(r) // 2
            try:
                hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=120,
                                  random_state=0).fit((r[:split] * SCALE).reshape(-1, 1))
            except Exception:
                continue
            turb = int(np.argmax(hmm.covars_.ravel()))
            P = _filter(r * SCALE, hmm.means_.ravel(), hmm.covars_.ravel(),
                        hmm.transmat_, hmm.startprob_)[:, turb]
            fwd = np.full(len(r), np.nan)
            for t in range(len(r) - H):
                fwd[t] = r[t + 1:t + 1 + H].sum()
            te = np.arange(split, len(r) - H)
            hot = P[te] >= 0.5
            if hot.sum() < 30 or (~hot).sum() < 30:
                continue
            f_turb, f_calm = np.nanmean(fwd[te][hot]), np.nanmean(fwd[te][~hot])
            # regime-timed strategy (causal: lag regime 1 day): in market only when calm
            w = (np.concatenate([[0.5], P[:-1]]) < 0.5).astype(float)
            idx = np.arange(split, len(r))
            sh_bh = _sharpe(r[idx])
            sh_rt = _sharpe((w * r)[idx])
            rows.append((f_turb, f_calm, sh_bh, sh_rt))
    A = np.array(rows)
    n = len(A)
    print(f"stocks with a fitted per-stock regime: {n}\n")
    ft, fc = A[:, 0], A[:, 1]
    print("Forward-20d return by own-stock regime (OOS, causal):")
    print(f"  mean in CALM      : {fc.mean():+.4f}")
    print(f"  mean in TURBULENT : {ft.mean():+.4f}")
    print(f"  stocks where turbulent < calm (de-risking HELPS): {np.mean(ft < fc):.0%}")
    print(f"  stocks where turbulent > calm (rebound, like FTSE): {np.mean(ft > fc):.0%}\n")
    print("Regime-timed (exit while turbulent) vs buy-and-hold Sharpe (OOS):")
    print(f"  mean buy-hold      : {A[:,2].mean():+.2f}")
    print(f"  mean regime-timed  : {A[:,3].mean():+.2f}")
    print(f"  regime-timing wins : {np.mean(A[:,3] > A[:,2]):.0%} of stocks")


if __name__ == "__main__":
    main()
