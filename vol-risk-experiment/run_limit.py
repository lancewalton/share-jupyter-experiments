"""Limit-at-decision-close execution: does skipping the gap-ups help or hurt?

Rule: place a buy-limit at the decision close C_d. It fills AT C_d if the next
day trades down to it (Low_{d+1} <= C_d); otherwise the name gapped up and is
skipped. So the fill PRICE is a non-issue (= C_d); the only effect is SELECTION —
you never hold the names that gapped up and ran. Test: forward return of FILLED
vs SKIPPED entries, for the low-vol and momentum longs, and the portfolio impact.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from mc.forecasters import _ewma_vol_series

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
H, QUINT, ANN = 20, 0.2, 252


def _load_cl():
    close, low = {}, {}
    for pat in DIRS:
        for path in sorted(glob.glob(pat)):
            try:
                df = pd.read_csv(path, thousands=",", encoding="utf-8-sig")
            except Exception:
                continue
            df.columns = [c.strip().strip('"') for c in df.columns]
            pc = "Close" if "Close" in df.columns else ("Price" if "Price" in df.columns else None)
            if pc is None or "Low" not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            for c in (pc, "Low"):
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
            df = df.dropna(subset=["Date", pc, "Low"]).sort_values("Date")
            df = df[~df["Date"].duplicated(keep="last")].set_index("Date")
            if len(df) < 1500 or (df[pc] <= 0).any():
                continue
            tk = os.path.splitext(os.path.basename(path))[0]
            close[tk] = df[pc]; low[tk] = df["Low"]
    C = pd.DataFrame(close).sort_index()
    return C, pd.DataFrame(low).reindex_like(C)


def main():
    C, L = _load_cl(); dates = C.index
    logC = np.log(C); R = logC.diff()
    mom = R.rolling(231).sum().shift(21)
    vol = R.apply(lambda c: pd.Series(np.sqrt(_ewma_vol_series(c.fillna(0).to_numpy(),
                  lam=0.94, burn=60)), index=dates))
    fwdH = logC.shift(-H) - logC                       # forward H-day return from C_d
    print(f"universe {C.shape[1]}; {dates[0].date()} -> {dates[-1].date()}\n")

    rebal = list(dates[260::H])
    recs = []   # (kind, filled(bool), fwd_return)
    for i, d in enumerate(rebal[:-1]):
        di = dates.get_loc(d)
        if di + 1 >= len(dates):
            continue
        d1 = dates[di + 1]
        cd = C.loc[d]; low1 = L.loc[d1]; fwd = fwdH.loc[d]
        for kind, sig, top in [("low-vol", vol.loc[d], False), ("momentum", mom.loc[d], True)]:
            s = sig.dropna()
            if len(s) < 30:
                continue
            k = max(1, int(QUINT * len(s)))
            names = (s.nlargest(k) if top else s.nsmallest(k)).index
            for nm in names:
                if nm in cd.index and nm in low1.index and np.isfinite(cd[nm]) \
                        and np.isfinite(low1[nm]) and np.isfinite(fwd.get(nm, np.nan)):
                    filled = low1[nm] <= cd[nm]
                    recs.append((kind, filled, float(fwd[nm])))
    df = pd.DataFrame(recs, columns=["kind", "filled", "fwd"])
    print("Forward 20-day return of ENTERED names, filled (Low<=close) vs skipped (gapped up):")
    print(f"  {'sleeve':<10}{'fill rate':>10}{'filled fwd':>12}{'skipped fwd':>13}{'skip−fill':>11}")
    for kind in ["low-vol", "momentum"]:
        g = df[df.kind == kind]
        fr = g.filled.mean()
        ff = g[g.filled].fwd.mean(); sf = g[~g.filled].fwd.mean()
        print(f"  {kind:<10}{fr:>9.0%}{ff*100:>+11.2f}%{sf*100:>+12.2f}%{(sf-ff)*100:>+10.2f}%")
    print("\n  (skipped − filled > 0  =>  the gap-ups you SKIP were the WINNERS = adverse selection)")

    # portfolio impact: low-vol core, market (all filled@close) vs limit (skip gap-ups,
    # cash for the unfilled slots) vs limit-renormalised (isolates pure selection)
    print("\nLow-vol core, mean per-cycle forward return (equal-weight quintile):")
    mkt, lim_cash, lim_renorm = [], [], []
    for d in rebal[:-1]:
        di = dates.get_loc(d); d1 = dates[dates.get_loc(d) + 1] if di + 1 < len(dates) else None
        if d1 is None: continue
        s = vol.loc[d].dropna()
        if len(s) < 30: continue
        k = max(1, int(QUINT * len(s))); names = s.nsmallest(k).index
        cd, low1, fwd = C.loc[d], L.loc[d1], fwdH.loc[d]
        f = np.array([np.exp(fwd[nm]) - 1 for nm in names if np.isfinite(fwd.get(nm, np.nan))])
        fill = np.array([np.isfinite(low1.get(nm, np.nan)) and low1[nm] <= cd[nm]
                         for nm in names if np.isfinite(fwd.get(nm, np.nan))])
        if len(f) == 0: continue
        mkt.append(f.mean())                                        # hold all
        lim_cash.append((f * fill).sum() / len(f))                  # unfilled -> cash
        lim_renorm.append(f[fill].mean() if fill.any() else 0.0)    # only filled, full weight
    def sh(x):
        x = np.array(x); return x.mean() / x.std() * np.sqrt(ANN / H) if x.std() > 0 else 0
    print(f"  market (hold all)          mean {np.mean(mkt)*100:+.3f}%/cycle  Sharpe {sh(mkt):+.2f}")
    print(f"  limit, cash on unfilled    mean {np.mean(lim_cash)*100:+.3f}%/cycle  Sharpe {sh(lim_cash):+.2f}")
    print(f"  limit, renormalised        mean {np.mean(lim_renorm)*100:+.3f}%/cycle  Sharpe {sh(lim_renorm):+.2f}")


if __name__ == "__main__":
    main()
