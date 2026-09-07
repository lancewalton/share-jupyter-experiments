"""Effective bid-ask spreads from daily OHLC — does low-vol = tight-spread, and
do the small low-priced names breach the 10 bps the backtest assumes?

Two high/low spread estimators (no quote data needed):
  Corwin-Schultz (2012)  — separates spread from variance via 1-day vs 2-day H/L
  Abdi-Ranaldo  (2017)   — close vs high/low mid-range; usually the cleaner one
Both return a PROPORTIONAL full spread S. A one-way rebalance trade costs ~S/2
(crossing from mid), so compare S/2 against the backtest's COST = 10 bps.
Window: last 252 trading days (what you'd actually trade against now).
"""
from __future__ import annotations

import csv, glob, os
import numpy as np
import pandas as pd
from mc.forecasters import _ewma_vol_series

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
ANN, WIN, STALE_DAYS = 252, 252, 7
CORE = ["DCC", "CWK", "ZIG", "BNZL", "BLND", "GRI", "UU", "SVT", "AV", "SMIN",
        "LGEN", "HMSO", "WKP", "LAND", "WTB", "PAG", "DLN", "NG", "INF", "GPE",
        "HSBA", "SSE"]
DEN = 3 - 2 * np.sqrt(2)


def load():
    out = {}
    for pat in DIRS:
        for path in sorted(glob.glob(pat)):
            rows = []
            with open(path, encoding="utf-8-sig") as f:
                rd = csv.reader(f); next(rd, None)
                for r in rd:
                    if len(r) < 5:
                        continue
                    try:
                        d = pd.to_datetime(r[0], dayfirst=True)
                        c = float(r[1].replace(",", "")); h = float(r[3].replace(",", ""))
                        l = float(r[4].replace(",", ""))
                        rows.append((d, c, h, l))
                    except Exception:
                        pass
            df = pd.DataFrame(rows, columns=["d", "C", "H", "L"]).set_index("d").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            if len(df) < 1500 or (df[["C", "H", "L"]] <= 0).any().any():
                continue
            r = np.diff(np.log(df["C"].values))
            if np.abs(r).max() > 0.6:
                continue
            out[os.path.splitext(os.path.basename(path))[0]] = df
    return out


def corwin_schultz(H, L):
    h, l = np.log(H), np.log(L)
    hl = (h - l) ** 2
    beta = hl[:-1] + hl[1:]
    H2 = np.maximum(H[:-1], H[1:]); L2 = np.minimum(L[:-1], L[1:])
    gamma = (np.log(H2 / L2)) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / DEN - np.sqrt(gamma / DEN)
    S = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return np.nanmean(np.clip(S, 0, None))            # clip negatives to 0


def abdi_ranaldo(C, H, L):
    c = np.log(C); eta = (np.log(H) + np.log(L)) / 2
    term = 4 * (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    return np.sqrt(max(0.0, np.nanmean(term)))


def main():
    data = load()
    latest = max(df.index[-1] for df in data.values())
    rows = []
    for tk, df in data.items():
        if (latest - df.index[-1]).days > STALE_DAYS:
            continue
        last40 = df["C"].values[-40:]
        if (last40.max() - last40.min()) / last40.mean() < 0.03:   # pinned -> skip
            continue
        r = np.diff(np.log(df["C"].values))
        vann = float(np.sqrt(_ewma_vol_series(r, lam=0.94, burn=60)[-1]) * np.sqrt(ANN))
        w = df.tail(WIN)
        cs = corwin_schultz(w["H"].values, w["L"].values) * 1e4        # bps, full spread
        ar = abdi_ranaldo(w["C"].values, w["H"].values, w["L"].values) * 1e4
        rows.append((tk, vann, df["C"].values[-1] / 100.0, cs, ar))
    df = pd.DataFrame(rows, columns=["tk", "vol", "price", "cs", "ar"]).sort_values("vol")
    n = len(df); k = max(1, int(0.2 * n))

    print(f"universe {n} active names, spreads over last {WIN}d (as of {latest.date()})")
    print(f"spread = full proportional; one-way cost ~ spread/2 vs backtest COST 10 bps\n")
    lo, hi = df.head(k), df.tail(k)
    print("VOL-AXIS CHECK — mean effective spread by vol quintile:")
    print(f"  lowest-vol quintile   vol {lo.vol.mean()*100:4.1f}%   AR spread {lo.ar.mean():5.1f} bps   CS {lo.cs.mean():5.1f} bps")
    print(f"  highest-vol quintile  vol {hi.vol.mean()*100:4.1f}%   AR spread {hi.ar.mean():5.1f} bps   CS {hi.cs.mean():5.1f} bps")
    print(f"  corr(ann vol, AR spread) across universe = {np.corrcoef(df.vol, df.ar)[0,1]:+.2f}")
    print(f"  corr(1/price, AR spread)                 = {np.corrcoef(1/df.price, df.ar)[0,1]:+.2f}  (price-level axis)\n")

    core = df[df.tk.isin(CORE)].sort_values("ar")
    print("THE 22-NAME CORE — effective spread per name (AR = primary, CS = upper-ish):")
    print(f"  {'ticker':<7}{'ann vol':>8}{'price £':>9}{'AR bps':>8}{'CS bps':>8}{'1-way≈':>8}  flag")
    for _, x in core.iterrows():
        oneway = x.ar / 2
        flag = "WIDE >10bps" if oneway > 10 else ("watch" if oneway > 6 else "")
        print(f"  {x.tk:<7}{x.vol*100:>7.1f}%{x.price:>9.2f}{x.ar:>8.1f}{x.cs:>8.1f}{oneway:>7.1f}b  {flag}")
    br = core[core.ar / 2 > 10]
    print(f"\n  core names breaching 10 bps one-way: {len(br)}/{len(core)}"
          + (f" -> {', '.join(br.tk)}" if len(br) else ""))
    print(f"  core mean one-way cost (AR/2): {core.ar.mean()/2:.1f} bps  "
          f"(equal-weight book pays ~this per rebalance leg)")


if __name__ == "__main__":
    main()
