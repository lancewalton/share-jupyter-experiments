"""Diagnose data quality of the survivorship-free universe before backtesting:
is the liquidity filter picking real large-caps, and how extreme are the monthly
returns / prices (which blew up the first run)?

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u momentum_diagnose.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SP = Path("/private/tmp/claude-501/-Users-lance-Projects-share-jupyter-experiments/"
          "ef57bf07-743d-4448-a1f7-5248924dfeaf/scratchpad")
OHLCV = HERE / "eodhd_uk_ohlcv.parquet"


def main() -> None:
    df = pd.read_parquet(OHLCV, columns=["date", "code", "close", "adjusted_close", "volume"])
    df = df[df["date"] >= "1998-01-01"]
    ccy, name = {}, {}
    for fn in ("lse_active.json", "lse_delisted.json"):
        for x in json.load(open(SP / fn)):
            ccy.setdefault(x.get("Code"), x.get("Currency"))
            name.setdefault(x.get("Code"), x.get("Name"))
    fac = df["code"].map(lambda c: 0.01 if ccy.get(c) == "GBX" else 1.0).astype("float32")
    df["turnover"] = df["close"] * df["volume"] * fac
    adj = df.pivot(index="date", columns="code", values="adjusted_close")
    turn = df.pivot(index="date", columns="code", values="turnover")
    M = adj.resample("ME").last()
    mret = M.pct_change()
    liq = turn.resample("ME").mean().rolling(12).mean()

    print("=== top-15 by trailing liquidity at 2024-06 (should be real large-caps) ===")
    t = M.index[M.index.get_indexer([pd.Timestamp('2024-06-30')], method='nearest')[0]]
    top = liq.loc[t].dropna().nlargest(15)
    for c, v in top.items():
        print(f"  {c:8} {str(name.get(c,''))[:34]:34} £/day~{v/1e6:8.2f}m  last_px={M.loc[t,c]:.1f}")

    print("\n=== monthly return distribution (all names) ===")
    r = mret.stack()
    for q in [0.001, 0.01, 0.5, 0.99, 0.999, 0.9999]:
        print(f"  q{q}: {r.quantile(q):+.3f}")
    print(f"  count |r|>1 (>100%): {(r.abs()>1).sum()}   >5 (>500%): {(r>5).sum()}   "
          f"==-1 (to zero): {(r<=-0.999).sum()}   total obs: {len(r):,}")

    print("\n=== price (adjusted_close) levels ===")
    p = adj.stack()
    print(f"  frac < 1 (sub-unit): {(p<1).mean():.3f}   < 5: {(p<5).mean():.3f}   "
          f"< 20: {(p<20).mean():.3f}")

    print("\n=== within top-350 liquid, monthly return extremes (2001+) ===")
    ex = []
    for i in range(1, len(M.index)):
        tprev = M.index[i-1]
        e = liq.loc[tprev].dropna().nlargest(350).index
        rr = mret.loc[M.index[i], e].dropna()
        if len(rr):
            ex.append(rr.abs().max())
    ex = pd.Series(ex)
    print(f"  max |monthly return| within top-350: median {ex.median():.2f}  p95 {ex.quantile(.95):.2f}  max {ex.max():.2f}")
    print(f"  months where top-350 had a |return|>2 (+200%): {(ex>2).sum()} of {len(ex)}")


if __name__ == "__main__":
    main()
