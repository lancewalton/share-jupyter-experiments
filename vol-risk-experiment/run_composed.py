"""The full composed strategy, one backtest, net of costs, no look-ahead.

Stacks the validated pieces:
  CORE     low-vol long-only equity tilt
  SLEEVE   + market-neutral 12-1 momentum (diversifying, ~uncorrelated w/ core)
  VOLTGT   + vol-target the combined book (constant-risk sizing)
  REGIME   + systemic (market-level) regime overlay, de-risk in crises (lagged)
Net of 10 bps turnover throughout; regime signal lagged one day (no look-ahead).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

from mc.data import load_close
from mc.returns import log_returns
from mc.forecasters import _ewma_vol_series

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
H, QUINT, ANN, COST, SCALE = 20, 0.2, 252, 0.001, 100.0
W_MOM, TARGET, MAXLEV = 0.5, 0.12, 2.5


def _load():
    out = {}
    for pat in DIRS:
        for path in sorted(glob.glob(pat)):
            try:
                s = load_close(path)
            except Exception:
                continue
            r = log_returns(s.to_numpy())
            if len(r) < 1500 or np.abs(r).max() > 0.6:
                continue
            out[os.path.splitext(os.path.basename(path))[0]] = pd.Series(r, index=s.index[1:])
    return pd.DataFrame(out)


def _book(signal, R, rebal, dates, mode):
    """daily NET book return (rebalance turnover charged) for 'ls' or 'low'."""
    W = pd.DataFrame(0.0, index=dates, columns=R.columns)
    cost = pd.Series(0.0, index=dates); prev = None
    for i, d in enumerate(rebal):
        if d not in signal.index:
            continue
        s = signal.loc[d].dropna()
        if len(s) < 30:
            continue
        k = max(1, int(QUINT * len(s))); rank = s.rank()
        w = pd.Series(0.0, index=s.index)
        if mode == "ls":
            w[rank > len(s) - k] = 0.5 / k; w[rank <= k] = -0.5 / k
        else:
            w[rank <= k] = 1.0 / k
        end = rebal[i + 1] if i + 1 < len(rebal) else dates[-1]
        W.loc[(dates > d) & (dates <= end), w.index] = w.values
        if prev is not None:
            t = (w.reindex(prev.index.union(w.index)).fillna(0)
                 - prev.reindex(prev.index.union(w.index)).fillna(0)).abs().sum()
            cost.loc[d] = COST * t
        prev = w
    return (R * W).sum(axis=1).to_numpy() - cost.to_numpy()


def _stats(x):
    x = np.nan_to_num(x); ar = x.mean() * ANN; av = x.std() * np.sqrt(ANN)
    W = np.exp(np.cumsum(x)); peak = np.maximum.accumulate(W)   # compounded wealth
    dd = float(((W - peak) / peak).min())                        # true % drawdown
    return ar, av, (ar / av if av > 0 else 0.0), dd


def _filter(x, means, vars_, trans, start):
    n, k = len(x), len(means)
    lp = -0.5 * (np.log(2 * np.pi * vars_) + (x[:, None] - means) ** 2 / vars_)
    pdf = np.exp(lp - lp.max(axis=1, keepdims=True))
    a = np.empty((n, k)); a[0] = start * pdf[0]; a[0] /= a[0].sum()
    for t in range(1, n):
        a[t] = pdf[t] * (a[t - 1] @ trans); a[t] /= a[t].sum()
    return a


def main():
    R = _load(); dates = R.index.sort_values(); R = R.loc[dates]
    mom = R.rolling(231).sum().shift(21)
    vol = R.apply(lambda c: pd.Series(np.sqrt(_ewma_vol_series(c.fillna(0).to_numpy(),
                  lam=0.94, burn=60)), index=R.index))
    rebal = dates[260::H]
    core = _book(vol, R, rebal, dates, "low")           # low-vol long-only (net)
    sleeve = _book(mom, R, rebal, dates, "ls")           # momentum L/S (net)
    mkt = R.mean(axis=1).to_numpy()

    combined = core + W_MOM * sleeve                     # core + momentum sleeve
    # vol-target the combined book
    cvol = np.sqrt(_ewma_vol_series(np.nan_to_num(combined), lam=0.94, burn=60)) * np.sqrt(ANN)
    m_vt = np.nan_to_num(np.clip(TARGET / np.where(cvol > 0, cvol, np.nan), 0, MAXLEV))
    # systemic regime overlay (market HMM, causal, LAGGED one day)
    split = len(mkt) // 2
    hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=200,
                      random_state=0).fit((mkt[:split] * SCALE).reshape(-1, 1))
    turb = int(np.argmax(hmm.covars_.ravel()))
    P = _filter(mkt * SCALE, hmm.means_.ravel(), hmm.covars_.ravel(),
                hmm.transmat_, hmm.startprob_)[:, turb]
    m_rg = 1.0 - 0.7 * np.concatenate([[P[0]], P[:-1]])

    def overlay(book, mult):
        oc = COST * np.abs(np.diff(mult, prepend=mult[0]))    # cost of resizing
        return mult * book - oc

    stages = [("market (equal-wt)", mkt),
              ("low-vol core", core),
              ("+ momentum sleeve", combined),
              ("+ vol-target", overlay(combined, m_vt)),
              ("+ regime (FULL)", overlay(combined, m_vt * m_rg))]
    print(f"universe {R.shape[1]}; {dates[260].date()}..{dates[-1].date()};  "
          f"W_mom {W_MOM}  target {TARGET:.0%}  cost {COST*1e4:.0f}bps  regime LAGGED\n")
    print(f"{'stage':<22}{'ann.ret':>9}{'ann.vol':>9}{'Sharpe':>8}{'maxDD':>8}{'mkt-beta':>10}")
    curves = {}
    for nm, x in stages:
        ar, av, sh, dd = _stats(x)
        beta = np.polyfit(np.nan_to_num(mkt), np.nan_to_num(x), 1)[0]
        curves[nm] = np.nancumsum(np.nan_to_num(x))
        print(f"{nm:<22}{ar:>+9.3f}{av:>9.3f}{sh:>+8.2f}{dd:>+8.2f}{beta:>+10.2f}")
    full = overlay(combined, m_vt * m_rg)
    print("\nFULL strategy sub-period Sharpe (thirds):",
          "  ".join(f"{_stats(full[a:b])[2]:+.2f}" for a, b in
                    [(0, len(full)//3), (len(full)//3, 2*len(full)//3), (2*len(full)//3, len(full))]))

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for nm, cur in curves.items():
        ax.plot(dates, cur, label=nm, lw=1.1 if "FULL" in nm else 0.9)
    ax.axhline(0, color="k", lw=.6, ls=":"); ax.legend(fontsize=8)
    ax.set_title("Composed defensive strategy (net of 10bps, no look-ahead)")
    ax.set_ylabel("cumulative net return"); fig.tight_layout(); fig.savefig("composed.png", dpi=110)
    print("\nsaved composed.png")


if __name__ == "__main__":
    main()
