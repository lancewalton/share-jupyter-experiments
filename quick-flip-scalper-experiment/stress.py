"""Stress-test the promising ORB cell: liquidity >= 0.75x ATR, box-end hold.

Four independent checks, because this result was cherry-picked from many cells
on a single 60-day window:
  1. Day-block bootstrap CI  - trades on one day share the market move, so we
     resample whole DAYS, not individual trades (a per-trade CI would be falsely
     tight).
  2. Concentration          - is the edge a handful of names or days?
  3. Drift / beta control    - subtract each day's market move in the trade's
     direction; is there edge left once you remove "shorts won in a down tape"?
  4. Direction permutation   - randomise long/short; where does the real edge sit
     in that null?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from orb import preload, C1_BARS, BOX_BARS

RNG = np.random.default_rng(20260901)
COST = 5.0 / 1e4          # 5 bps round-trip
THR = 0.75
N_BOOT = 20000


def net_R(tr: pd.DataFrame) -> pd.Series:
    return (tr.gross_ret - COST) / tr.risk_frac


def market_ret_by_date() -> dict:
    """Equal-weight 75-min morning return (C1 close -> box end) across all
    name-days, per date - a beta proxy for the box window."""
    rows = []
    for t, d, c, a in preload():
        be = min(len(c) - 1, C1_BARS + BOX_BARS - 1)
        start = c[C1_BARS - 1].close
        if start > 0:
            rows.append((d, c[be].close / start - 1.0))
    m = pd.DataFrame(rows, columns=["date", "r"]).groupby("date")["r"].mean()
    return m.to_dict()


def day_block_boot(tr: pd.DataFrame, value: pd.Series) -> tuple[float, float, float]:
    """Bootstrap the mean of `value` by resampling whole dates with replacement."""
    by_date = {d: value[tr.date == d].to_numpy() for d in tr.date.unique()}
    dates = list(by_date)
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = RNG.choice(len(dates), size=len(dates), replace=True)
        pooled = np.concatenate([by_date[dates[j]] for j in pick])
        means[i] = pooled.mean()
    return float(value.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    tr_all = pd.read_parquet("orb_trades_box_end.parquet")
    tr = tr_all[tr_all.liq_ratio >= THR].copy().reset_index(drop=True)
    tr["net_R"] = net_R(tr)
    print(f"Cell: liq>={THR}x ATR, box-end, 5bps cost.  n={len(tr)}  "
          f"tickers={tr.ticker.nunique()}  dates={tr.date.nunique()}\n")

    # ---- 1. headline + day-block bootstrap ------------------------------- #
    print("== 1. Expectancy & day-block bootstrap (95% CI) ==")
    for label, sub in [("BOTH", tr), ("LONG", tr[tr.side == "long"]),
                       ("SHORT", tr[tr.side == "short"])]:
        m, lo, hi = day_block_boot(sub, sub.net_R)
        star = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
        print(f"  {label:5s} n={len(sub):5d}  mean net {m:+.4f}R  "
              f"[{lo:+.4f}, {hi:+.4f}]{star}")

    # ---- 2. concentration ------------------------------------------------ #
    print("\n== 2. Concentration ==")
    by_tk = tr.groupby("ticker").net_R.sum().sort_values()
    by_dt = tr.groupby("date").net_R.sum().sort_values()
    total = tr.net_R.sum()
    print(f"  total net-R summed: {total:+.2f}")
    print(f"  tickers: {tr.ticker.nunique()}  |  net-positive names "
          f"{(by_tk>0).sum()} vs negative {(by_tk<0).sum()}")
    print(f"  top-10 names contribute {by_tk.tail(10).sum()/total*100:.0f}% of total net-R")
    print(f"  days: {tr.date.nunique()}  |  net-positive days "
          f"{(by_dt>0).sum()} vs negative {(by_dt<0).sum()}")
    print(f"  top-3 days contribute {by_dt.tail(3).sum()/total*100:.0f}% of total net-R")
    top3 = by_dt.tail(3).index
    rest = tr[~tr.date.isin(top3)]
    print(f"  mean net R excluding those 3 best days: {rest.net_R.mean():+.4f}R "
          f"(n={len(rest)})")

    # ---- 3. drift / beta control ---------------------------------------- #
    print("\n== 3. Drift / beta control (market-neutral alpha) ==")
    mkt = market_ret_by_date()
    drift = np.mean(list(mkt.values()))
    print(f"  ambient 75-min morning drift (equal-weight): {drift*1e4:+.1f} bps")
    sign = np.where(tr.side.to_numpy() == "long", 1.0, -1.0)
    mret = tr.date.map(mkt).to_numpy()
    beta_comp = sign * mret                      # market move in the trade's direction
    tr["alpha_net_R"] = (tr.gross_ret.to_numpy() - beta_comp - COST) / tr.risk_frac.to_numpy()
    m, lo, hi = day_block_boot(tr, tr["alpha_net_R"])
    star = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
    print(f"  BOTH market-neutral mean net {m:+.4f}R  [{lo:+.4f}, {hi:+.4f}]{star}")
    for label, sub in [("LONG", tr[tr.side == "long"]), ("SHORT", tr[tr.side == "short"])]:
        print(f"  {label:5s} market-neutral mean {sub.alpha_net_R.mean():+.4f}R")

    # ---- 4. direction permutation --------------------------------------- #
    print("\n== 4. Direction-permutation null (random long/short) ==")
    base = tr.gross_ret.to_numpy()
    risk = tr.risk_frac.to_numpy()
    actual = tr.net_R.mean()
    null = np.empty(N_BOOT)
    for i in range(N_BOOT):
        eps = RNG.choice([-1.0, 1.0], size=len(tr))
        null[i] = ((eps * base - COST) / risk).mean()
    p = (null >= actual).mean()
    print(f"  actual mean net {actual:+.4f}R  |  null mean {null.mean():+.4f}R  "
          f"[{np.percentile(null,2.5):+.4f}, {np.percentile(null,97.5):+.4f}]")
    print(f"  p(null >= actual) = {p:.4f}")
    print("  (note: this conflates real direction-signal with 'shorts won in a "
          "down tape' - read alongside check 3)")


if __name__ == "__main__":
    main()
