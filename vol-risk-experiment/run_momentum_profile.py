"""Momentum-sleeve risk profile: the path the Sharpe hides.

12-1 momentum has net Sharpe +0.34 (Task 4) -- but its defining risk is the
momentum CRASH (sharp losses in post-bottom rebounds). Here: equity curve, worst
drawdown, sub-period Sharpes, turnover, correlation with the low-vol core and the
market, and whether vol-scaling the sleeve tames the crash.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.data import load_close
from mc.returns import log_returns
from mc.forecasters import _ewma_vol_series

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
H, QUINT, ANN = 20, 0.2, 252


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
    """Daily book return + turnover. mode: 'ls' (dollar-neutral quintiles) or
    'low' (long-only bottom quintile = low signal)."""
    W = pd.DataFrame(0.0, index=dates, columns=R.columns)
    turn = []
    prev = None
    for i, d in enumerate(rebal):
        if d not in signal.index:
            continue
        s = signal.loc[d].dropna()
        if len(s) < 30:
            continue
        k = max(1, int(QUINT * len(s))); rank = s.rank()
        w = pd.Series(0.0, index=s.index)
        if mode == "ls":
            w[rank > len(s) - k] = 0.5 / k
            w[rank <= k] = -0.5 / k
        else:  # long-only lowest quintile
            w[rank <= k] = 1.0 / k
        end = rebal[i + 1] if i + 1 < len(rebal) else dates[-1]
        W.loc[(dates > d) & (dates <= end), w.index] = w.values
        if prev is not None:
            turn.append(float((w.reindex(prev.index.union(w.index)).fillna(0)
                               - prev.reindex(prev.index.union(w.index)).fillna(0)).abs().sum()))
        prev = w
    book = (R * W).sum(axis=1).to_numpy()
    return book, np.mean(turn) if turn else np.nan


def _stats(x):
    x = np.nan_to_num(x); ar = x.mean() * ANN; av = x.std() * np.sqrt(ANN)
    g = np.cumsum(x); dd = float((g - np.maximum.accumulate(g)).min())
    return ar, av, (ar / av if av > 0 else 0.0), dd


def main():
    R = _load(); dates = R.index.sort_values(); R = R.loc[dates]
    mom = R.rolling(231).sum().shift(21)
    vol = R.apply(lambda c: pd.Series(np.sqrt(_ewma_vol_series(c.fillna(0).to_numpy(),
                  lam=0.94, burn=60)), index=R.index))
    rebal = dates[260::H]
    mom_r, mom_turn = _book(mom, R, rebal, dates, "ls")
    low_r, _ = _book(vol, R, rebal, dates, "low")   # lowest-vol quintile (long-only)
    mkt_r = R.mean(axis=1).to_numpy()

    # vol-scaled momentum (constant-vol): scale by 10% / trailing ann vol of the sleeve
    mvol = np.sqrt(_ewma_vol_series(np.nan_to_num(mom_r), lam=0.94, burn=60)) * np.sqrt(ANN)
    wsc = np.nan_to_num(np.clip(0.10 / np.where(mvol > 0, mvol, np.nan), 0, 3))
    mom_vs = wsc * mom_r

    print(f"universe {R.shape[1]}; {dates[260].date()}..{dates[-1].date()}\n")
    print(f"{'book':<24}{'ann.ret':>9}{'ann.vol':>9}{'Sharpe':>8}{'maxDD':>8}")
    for nm, x in [("market", mkt_r), ("low-vol core", low_r),
                  ("momentum L/S", mom_r), ("momentum L/S vol-scaled", mom_vs)]:
        ar, av, sh, dd = _stats(x)
        print(f"{nm:<24}{ar:>+9.3f}{av:>9.3f}{sh:>+8.2f}{dd:>+8.2f}")
    print(f"\nmomentum avg turnover/rebalance: {mom_turn:.2f}")

    # worst single-month move (the crash), and sub-period Sharpes
    mseries = pd.Series(mom_r, index=dates)
    monthly = mseries.groupby([dates.year, dates.to_series().dt.month.values]).sum()
    print(f"worst momentum month: {monthly.min():+.3f}   best: {monthly.max():+.3f}")
    print("sub-period Sharpe (thirds, momentum L/S):",
          "  ".join(f"{_stats(mom_r[a:b])[2]:+.2f}" for a, b in
                    [(0, len(mom_r)//3), (len(mom_r)//3, 2*len(mom_r)//3), (2*len(mom_r)//3, len(mom_r))]))

    # correlations (the diversification case)
    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b); return float(np.corrcoef(a[m], b[m])[0, 1])
    print(f"\ncorrelation of momentum L/S with:")
    print(f"  low-vol core : {corr(mom_r, low_r):+.2f}")
    print(f"  market       : {corr(mom_r, mkt_r):+.2f}")

    fig, ax = plt.subplots(figsize=(11, 5))
    for nm, x, c in [("market", mkt_r, "tab:gray"), ("low-vol core", low_r, "tab:green"),
                     ("momentum L/S", mom_r, "tab:blue"),
                     ("momentum vol-scaled", mom_vs, "tab:orange")]:
        ax.plot(dates, np.nancumsum(np.nan_to_num(x)), label=nm, lw=1)
    ax.axhline(0, color="k", lw=.6, ls=":"); ax.legend(fontsize=8)
    ax.set_title("Momentum sleeve risk profile (UK equities)")
    ax.set_ylabel("cumulative return"); fig.tight_layout(); fig.savefig("momentum_profile.png", dpi=110)
    print("\nsaved momentum_profile.png")


if __name__ == "__main__":
    main()
