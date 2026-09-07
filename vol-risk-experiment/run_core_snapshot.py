"""Current low-vol core: the lowest-vol quintile as of the latest date, with the
per-name £ split for a £10,000 long-only book (fractional shares).

Point-in-time snapshot of the STRATEGY.md core (§3a): rank the eligible active
cross-section by EWMA volatility (λ=0.94, burn 60), go long the lowest 20%,
equal-weight. Excludes names that are frozen/delisted (no recent price) — their
returns would otherwise decay to ~0 and masquerade as ultra-low-vol.
"""
from __future__ import annotations

import csv, glob, os
import numpy as np
import pandas as pd
from mc.forecasters import _ewma_vol_series

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
ANN, QUINT, BOOK = 252, 0.2, 10_000.0
STALE_DAYS = 7          # exclude names whose last price is older than this


def load():
    close = {}
    for pat in DIRS:
        for path in sorted(glob.glob(pat)):
            recs = []
            with open(path, encoding="utf-8-sig") as f:
                rd = csv.reader(f); next(rd, None)
                for r in rd:
                    if len(r) < 5:
                        continue
                    try:
                        recs.append((pd.to_datetime(r[0], dayfirst=True),
                                     float(r[1].replace(",", ""))))
                    except Exception:
                        pass
            s = pd.Series(dict(recs)).sort_index()
            r = np.diff(np.log(s.values))
            if len(s) < 1500 or (np.abs(r).max() if len(r) else 1) > 0.6 or (s <= 0).any():
                continue
            close[os.path.splitext(os.path.basename(path))[0]] = s
    return close


def main():
    close = load()
    latest = max(s.index[-1] for s in close.values())
    rows, pinned = [], []
    for tk, s in close.items():
        if (latest - s.index[-1]).days > STALE_DAYS:      # frozen/delisted -> skip
            continue
        r = np.log(s.values[1:] / s.values[:-1])
        vann = float(np.sqrt(_ewma_vol_series(r, lam=0.94, burn=60)[-1]) * np.sqrt(ANN))
        last40 = s.values[-40:]
        rng = (last40.max() - last40.min()) / last40.mean()   # pinned/takeover proxy
        if rng < 0.03:                                    # flat = deal-pinned, not calm
            pinned.append((tk, vann, rng)); continue
        rows.append((tk, vann, s.values[-1] / 100.0))       # price pence -> £
    df = pd.DataFrame(rows, columns=["ticker", "ann_vol", "price_gbp"]).sort_values("ann_vol")
    n = len(df); k = max(1, int(QUINT * n))
    core = df.head(k).copy()
    if pinned:
        print("excluded (pinned near a takeover price — deal risk, not low vol): "
              + ", ".join(f"{t} ({v*100:.1f}% vol, {g*100:.1f}% range)" for t, v, g in pinned) + "\n")
    alloc = BOOK / k
    core["alloc_gbp"] = round(alloc, 2)
    core["shares"] = (alloc / core["price_gbp"]).round(3)

    print(f"as of {latest.date()};  {n} active eligible names;  low-vol quintile = {k} names")
    print(f"£{BOOK:,.0f} equal-weight  ->  £{alloc:,.2f} per name  (long-only core, §3a)\n")
    print(f"  {'#':>2} {'ticker':<7}{'ann vol':>9}{'price £':>10}{'alloc £':>10}{'shares':>10}")
    for i, (_, x) in enumerate(core.iterrows(), 1):
        print(f"  {i:>2} {x.ticker:<7}{x.ann_vol*100:>8.1f}%{x.price_gbp:>10.2f}"
              f"{x.alloc_gbp:>10.2f}{x.shares:>10.3f}")
    print(f"\n  portfolio ann vol (equal-weight, ignoring correlation): "
          f"{core.ann_vol.mean()*100:.1f}%  |  book total £{alloc*k:,.0f}")
    core.to_csv("core_snapshot_10k.csv", index=False)
    print("  saved core_snapshot_10k.csv")


if __name__ == "__main__":
    main()
