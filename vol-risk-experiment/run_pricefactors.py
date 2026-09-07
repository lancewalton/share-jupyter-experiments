"""Task 4: longer-term PRICE factors on the UK equity cross-section.

Price-based only (no fundamentals in our data). Three cross-sectional signals:
  MOM   12-1 month momentum   (past 12m return, skip most recent month)
  REV   long-term reversal    (-(5y..1y past return); long past losers)
  LOWV  low volatility        (-trailing 60d vol; the tilt that already worked)
Measure each honestly (rank IC, market-neutral L/S net Sharpe + beta) and whether
combining them beats the best single factor. Monthly rebalance, ~150 UK equities.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from mc.data import load_close
from mc.returns import log_returns
from mc.forecasters import _ewma_vol_series

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
H, QUINT, COST, MIN_NAMES, ANN = 20, 0.2, 0.001, 30, 252


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


def _rank_xs(df):
    return df.rank(axis=1).sub(0.5).div(df.notna().sum(axis=1), axis=0) - 0.5


def _sh(x):
    x = np.array(x); per = ANN / H
    return x.mean() / x.std() * np.sqrt(per) if x.std() > 0 else 0.0


def main():
    R = _load()
    dates = R.index.sort_values(); R = R.loc[dates]
    mom = R.rolling(231).sum().shift(21)                       # 12-1 momentum
    rev = -(R.rolling(1008).sum().shift(252))                  # -(5y..1y): long losers
    lowv = -R.apply(lambda c: pd.Series(
        np.sqrt(_ewma_vol_series(c.fillna(0).to_numpy(), lam=0.94, burn=60)), index=R.index))
    fwd = R.rolling(H).sum().shift(-H)
    sigs = {"MOM 12-1": mom, "REV 5y-1y": rev, "LOWV": lowv}
    Z = {k: _rank_xs(v.reindex(dates)) for k, v in sigs.items()}
    Z["COMBINED"] = pd.concat(list(Z.values())).groupby(level=0).mean()
    Z["MOM+LOWV"] = pd.concat([Z["MOM 12-1"], Z["LOWV"]]).groupby(level=0).mean()
    FWD, VOL = fwd, -lowv                                      # VOL for inverse-vol sizing
    rebal = dates[1260::H]
    print(f"universe {R.shape[1]}; {len(rebal)} monthly rebalances "
          f"{rebal[0].date()}..{rebal[-1].date()}\n")

    def xs_ic(sig):
        ics = []
        for d in rebal:
            if d in sig.index and d in FWD.index:
                a, b = sig.loc[d].to_numpy(), FWD.loc[d].to_numpy()
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() >= MIN_NAMES:
                    ics.append(np.corrcoef(np.argsort(np.argsort(a[m])),
                                           np.argsort(np.argsort(b[m])))[0, 1])
        ics = np.array(ics)
        return ics.mean(), ics.mean() / ics.std() * np.sqrt(len(ics))

    def backtest(sig):
        rets, turn, mkt = [], [], []
        prev = pd.Series(dtype=float)
        for d in rebal:
            if d not in sig.index or d not in FWD.index:
                continue
            s, y, v = sig.loc[d], FWD.loc[d], VOL.loc[d]
            ok = s.notna() & y.notna() & v.notna() & (v > 0)
            s, y, v = s[ok], y[ok], v[ok]
            if len(s) < MIN_NAMES:
                continue
            mkt.append(float(y.mean()))
            k = max(1, int(QUINT * len(s))); rank = s.rank()
            w = pd.Series(0.0, index=s.index)
            lo, hi = rank > len(s) - k, rank <= k
            w[lo] = (1 / v[lo]); w[lo] /= w[lo].sum() * 2
            w[hi] = -(1 / v[hi]); w[hi] /= -w[hi].sum() * 2
            rets.append(float((w * y).sum()))
            al = prev.index.union(w.index)
            turn.append(float((w.reindex(al).fillna(0) - prev.reindex(al).fillna(0)).abs().sum()))
            prev = w
        rets, turn, mkt = np.array(rets), np.array(turn), np.array(mkt)
        net = rets - COST * turn
        beta = np.polyfit(mkt, net, 1)[0]
        return _sh(rets), _sh(net), beta

    print(f"{'factor':<12}{'rank IC':>9}{'t':>6}{'L/S gross':>11}{'L/S net':>9}{'beta':>7}")
    for nm, sig in Z.items():
        ic, t = xs_ic(sig)
        g, n, b = backtest(sig)
        print(f"{nm:<12}{ic:>+9.4f}{t:>+6.1f}{g:>+11.2f}{n:>+9.2f}{b:>+7.2f}")


if __name__ == "__main__":
    main()
