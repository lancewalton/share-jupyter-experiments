"""The low-volatility anomaly: a cross-sectional strategy on the FORECASTABLE axis.

Documented effect: low-volatility stocks earn higher *risk-adjusted* returns than
high-volatility ones. We rank the universe each month by forecast volatility
(causal EWMA), go long the low-vol quintile / short the high-vol quintile
(dollar-neutral), and measure the market-neutral ALPHA -- no directional forecast,
only the vol ranking. Also the long-only low-vol book (the defensive version).
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
H, QUINT, COST, MIN_NAMES = 20, 0.2, 0.001, 30
ANN = 252


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
            tk = os.path.splitext(os.path.basename(path))[0]
            out[tk] = pd.Series(r, index=s.index[1:])
    return out


def _stats(x):
    per = ANN / H                              # ~12.6 H-day periods per year
    ar, av = x.mean() * per, x.std() * np.sqrt(per)
    sh = ar / av if av > 0 else 0.0
    g = np.cumsum(x); dd = float((g - np.maximum.accumulate(g)).min())
    return ar, av, sh, dd


def main():
    data = _load()
    vol, fwd = {}, {}
    for tk, r in data.items():
        vol[tk] = pd.Series(np.sqrt(_ewma_vol_series(r.to_numpy(), lam=0.94, burn=60)),
                            index=r.index)                       # causal vol forecast
        fwd[tk] = r.rolling(H).sum().shift(-H)                   # forward H-day return
    VOL, FWD = pd.DataFrame(vol), pd.DataFrame(fwd)
    dates = VOL.index.sort_values()
    rebal = dates[250::H]
    print(f"universe {len(data)} equities; {len(rebal)} monthly rebalances "
          f"{rebal[0].date()}..{rebal[-1].date()}\n")

    ls, lo, mkt, turn = [], [], [], []
    prev = pd.Series(dtype=float)
    for d in rebal:
        if d not in VOL.index or d not in FWD.index:
            continue
        v, y = VOL.loc[d], FWD.loc[d]
        ok = v.notna() & y.notna() & (v > 0)
        v, y = v[ok], y[ok]
        if len(v) < MIN_NAMES:
            continue
        mkt.append(float(y.mean()))
        k = max(1, int(QUINT * len(v)))
        rank = v.rank()
        low = rank <= k                          # low vol
        high = rank > len(v) - k                 # high vol
        w = pd.Series(0.0, index=v.index)
        w[low] = 0.5 / low.sum(); w[high] = -0.5 / high.sum()    # dollar-neutral
        ls.append(float((w * y).sum()))
        lo.append(float(y[low].mean()))          # long-only low-vol quintile
        al = prev.index.union(w.index)
        turn.append(float((w.reindex(al).fillna(0) - prev.reindex(al).fillna(0)).abs().sum()))
        prev = w
    ls, lo, mkt, turn = map(np.array, (ls, lo, mkt, turn))
    ls_net = ls - COST * turn

    # market-neutral alpha of the long-short book (OLS on the equal-weight market)
    beta, alpha = np.polyfit(mkt, ls_net, 1)
    resid = ls_net - (alpha + beta * mkt)
    t_alpha = alpha / (resid.std() / np.sqrt(len(resid)))
    per = ANN / H

    print(f"{'book':<26}{'ann.ret':>9}{'ann.vol':>9}{'Sharpe':>8}{'maxDD':>8}")
    for name, x in [("market (equal-wt)", mkt), ("long low-vol quintile", lo),
                    ("L/S low-minus-high (gross)", ls),
                    ("L/S low-minus-high (net)", ls_net)]:
        ar, av, sh, dd = _stats(x)
        print(f"{name:<26}{ar:>+9.3f}{av:>9.3f}{sh:>+8.2f}{dd:>+8.2f}")
    print(f"\nLong-short market exposure:  beta {beta:+.2f}   "
          f"annualised alpha {alpha*per:+.3f}   alpha t-stat {t_alpha:+.2f}")
    print(f"avg turnover/rebalance {turn.mean():.2f};  break-even cost "
          f"{1e4*ls.mean()/turn.mean():.0f} bps")

    print("\nSub-period Sharpe (L/S net):")
    for a, b in [(0, len(ls)//3), (len(ls)//3, 2*len(ls)//3), (2*len(ls)//3, len(ls))]:
        print(f"  {a:>3}-{b:<3}: {_stats(ls_net[a:b])[2]:+.2f}")

    _plot(ls_net, lo, mkt, rebal[:len(ls)])
    print("\nsaved lowvol.png")


def _plot(ls_net, lo, mkt, dts):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dts, np.cumsum(mkt), label="market (equal-wt)", color="tab:gray")
    ax.plot(dts, np.cumsum(lo), label="long low-vol quintile", color="tab:green")
    ax.plot(dts, np.cumsum(ls_net), label="L/S low-minus-high (net)", color="tab:blue")
    ax.axhline(0, color="k", lw=0.6, ls=":"); ax.legend()
    ax.set_title("Low-volatility anomaly (UK equities)")
    ax.set_ylabel("cumulative log return"); ax.set_xlabel("date")
    fig.tight_layout(); fig.savefig("lowvol.png", dpi=110)


if __name__ == "__main__":
    main()
