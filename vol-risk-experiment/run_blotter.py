"""Trade ledger for The Defensive Stack: one row per monthly rebalance cycle.

Reconstructs the recommended stack (low-vol core + 0.5x momentum sleeve +
vol-target, NO regime overlay), net of 10 bps, no look-ahead, then compounds a
GBP 100,000 notional through each rebalance period: period return, P&L, and the
running portfolio value. Writes defensive_stack_blotter.csv.
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
H, QUINT, ANN, COST = 20, 0.2, 252, 0.001
W_MOM, TARGET, MAXLEV, V0 = 0.5, 0.12, 2.5, 100_000.0


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


def main():
    R = _load(); dates = R.index.sort_values(); R = R.loc[dates]
    mom = R.rolling(231).sum().shift(21)
    vol = R.apply(lambda c: pd.Series(np.sqrt(_ewma_vol_series(c.fillna(0).to_numpy(),
                  lam=0.94, burn=60)), index=R.index))
    rebal = list(dates[260::H])
    core = _book(vol, R, rebal, dates, "low")
    sleeve = _book(mom, R, rebal, dates, "ls")
    combined = core + W_MOM * sleeve
    cvol = np.sqrt(_ewma_vol_series(np.nan_to_num(combined), lam=0.94, burn=60)) * np.sqrt(ANN)
    m_vt = np.nan_to_num(np.clip(TARGET / np.where(cvol > 0, cvol, np.nan), 0, MAXLEV))
    oc = COST * np.abs(np.diff(m_vt, prepend=m_vt[0]))
    daily = pd.Series(m_vt * combined - oc, index=dates)   # strategy daily net log return

    # aggregate to per-rebalance-cycle periods
    rows = []
    V = V0
    for i in range(len(rebal) - 1):
        d0, d1 = rebal[i], rebal[i + 1]
        seg = daily[(daily.index > d0) & (daily.index <= d1)]
        if len(seg) == 0:
            continue
        pr = float(seg.sum())                 # period log return
        pnl = V * (np.exp(pr) - 1)
        V += pnl
        rows.append({"rebalance": d1.date(), "days": len(seg),
                     "period_return_%": round((np.exp(pr) - 1) * 100, 2),
                     "pnl_gbp": round(pnl, 0), "portfolio_value_gbp": round(V, 0)})
    led = pd.DataFrame(rows)
    led["cum_return_%"] = (led["portfolio_value_gbp"] / V0 - 1).mul(100).round(1)
    out = "defensive_stack_blotter.csv"
    led.to_csv(out, index=False)

    # summary
    n = len(led); ret = led["period_return_%"].to_numpy() / 100
    cum = led["portfolio_value_gbp"].to_numpy()
    dd = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)
    yrs = (rebal[-1] - rebal[0]).days / 365.25
    cagr = (V / V0) ** (1 / yrs) - 1
    print(f"The Defensive Stack — trade ledger ({n} monthly cycles, "
          f"{rebal[0].date()} -> {rebal[-1].date()})\n")
    print(f"  starting capital     GBP {V0:>12,.0f}")
    print(f"  ending value         GBP {V:>12,.0f}")
    print(f"  total return         {(V/V0-1)*100:>12.0f}%   ({cagr*100:.1f}% CAGR over {yrs:.0f}y)")
    print(f"  winning months       {np.mean(ret > 0)*100:.0f}%")
    print(f"  best / worst month   {ret.max()*100:+.1f}% / {ret.min()*100:+.1f}%")
    print(f"  worst drawdown (value) {dd.min()*100:.0f}%")
    print(f"\nsaved {out}  ({n} rows)\n")
    print("first 5 cycles:\n", led.head(5).to_string(index=False))
    print("\nlast 5 cycles:\n", led.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
