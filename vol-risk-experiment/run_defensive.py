"""Do our working pieces COMPOUND? A regime-aware, vol-targeted low-vol book.

Combine the three validated tools:
  - cross-sectional low-vol SELECTION (the one edge that worked)
  - time-series VOL-TARGETING of the book (constant-risk sizing)
  - a REGIME overlay (cut exposure when the market HMM says turbulent)
and compare against the market and the plain low-vol book. The question: do these
independent tools stack, or is each redundant given the others?
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
H, QUINT, ANN, TARGET, MAXLEV, SCALE = 20, 0.2, 252, 0.12, 2.0, 100.0


def _load():
    ser = {}
    for pat in DIRS:
        for path in sorted(glob.glob(pat)):
            try:
                s = load_close(path)
            except Exception:
                continue
            r = log_returns(s.to_numpy())
            if len(r) < 1500 or np.abs(r).max() > 0.6:
                continue
            ser[os.path.splitext(os.path.basename(path))[0]] = pd.Series(r, index=s.index[1:])
    return pd.DataFrame(ser)


def _stats(x):
    x = x[np.isfinite(x)]
    ar, av = x.mean() * ANN, x.std() * np.sqrt(ANN)
    sh = ar / av if av > 0 else 0.0
    W = np.exp(np.nancumsum(x)); peak = np.maximum.accumulate(W)   # compounded wealth
    dd = float(((W - peak) / peak).min())                           # true % drawdown
    return ar, av, sh, dd


def _causal_filter(x, means, vars_, trans, start):
    n, k = len(x), len(means)
    lp = -0.5 * (np.log(2 * np.pi * vars_) + (x[:, None] - means) ** 2 / vars_)
    pdf = np.exp(lp - lp.max(axis=1, keepdims=True))
    a = np.empty((n, k)); a[0] = start * pdf[0]; a[0] /= a[0].sum()
    for t in range(1, n):
        a[t] = pdf[t] * (a[t - 1] @ trans); a[t] /= a[t].sum()
    return a


def main():
    R = _load()
    dates = R.index.sort_values(); R = R.loc[dates]
    # causal forecast vol per stock
    VOL = R.apply(lambda c: pd.Series(
        np.sqrt(_ewma_vol_series(c.fillna(0).to_numpy(), lam=0.94, burn=60)), index=R.index))
    rebal = dates[250::H]

    # build daily weight matrix for the low-vol book (hold quintile for the month)
    W = pd.DataFrame(0.0, index=dates, columns=R.columns)
    for i, d in enumerate(rebal):
        v = VOL.loc[d][R.loc[d].notna() & (VOL.loc[d] > 0)]
        if len(v) < 30:
            continue
        k = max(1, int(QUINT * len(v)))
        low = v.nsmallest(k).index
        end = rebal[i + 1] if i + 1 < len(rebal) else dates[-1]
        seg = (dates > d) & (dates <= end)
        W.loc[seg, low] = 1.0 / k
    book = (R * W).sum(axis=1).to_numpy()          # low-vol book daily return
    mkt = R.mean(axis=1).to_numpy()

    # time-series vol-target overlay on the book
    bvol = np.sqrt(_ewma_vol_series(np.nan_to_num(book), lam=0.94, burn=60)) * np.sqrt(ANN)
    wv = np.clip(TARGET / np.where(bvol > 0, bvol, np.nan), 0, MAXLEV)
    wv = np.nan_to_num(wv)
    book_vt = wv * book

    # regime overlay: HMM on the market, cut exposure when turbulent
    split = len(mkt) // 2
    hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=200,
                      random_state=0).fit((mkt[:split] * SCALE).reshape(-1, 1))
    turb = int(np.argmax(hmm.covars_.ravel()))
    p_turb = _causal_filter(mkt * SCALE, hmm.means_.ravel(), hmm.covars_.ravel(),
                            hmm.transmat_, hmm.startprob_)[:, turb]
    p_lag = np.concatenate([[p_turb[0]], p_turb[:-1]])   # LAG 1 day: no look-ahead
    wr = 1.0 - 0.7 * p_lag                          # scale exposure 1.0 -> 0.3 in turbulence
    book_rg = wr * book
    book_both = wr * book_vt

    print(f"universe {R.shape[1]}; {len(dates)} days; target {TARGET:.0%} vol\n")
    print(f"{'book':<28}{'ann.ret':>9}{'ann.vol':>9}{'Sharpe':>8}{'maxDD':>8}")
    books = [("market (equal-wt)", mkt), ("low-vol book", book),
             ("low-vol + vol-target", book_vt), ("low-vol + regime", book_rg),
             ("low-vol + both", book_both)]
    curves = {}
    for name, x in books:
        ar, av, sh, dd = _stats(x)
        curves[name] = np.nancumsum(np.nan_to_num(x))
        print(f"{name:<28}{ar:>+9.3f}{av:>9.3f}{sh:>+8.2f}{dd:>+8.2f}")

    print("\nSub-period Sharpe (thirds):")
    for name, x in books:
        segs = [_stats(x[a:b])[2] for a, b in
                [(0, len(x)//3), (len(x)//3, 2*len(x)//3), (2*len(x)//3, len(x))]]
        print(f"  {name:<28}" + "  ".join(f"{s:+.2f}" for s in segs))

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, cur in curves.items():
        ax.plot(dates, cur, label=name, lw=1)
    ax.axhline(0, color="k", lw=0.6, ls=":"); ax.legend(fontsize=8)
    ax.set_title("Do the pieces compound? Regime-aware vol-targeted low-vol book")
    ax.set_ylabel("cumulative log return"); ax.set_xlabel("date")
    fig.tight_layout(); fig.savefig("defensive.png", dpi=110)
    print("\nsaved defensive.png")


if __name__ == "__main__":
    main()
