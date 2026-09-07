"""Does adding VALUE and/or QUALITY to 12-1 momentum improve the tilt?

Point-in-time factors (as-of filing date, no lookahead) on the survivorship-free
top-350 universe:
  - momentum: 12-1 return
  - quality : ROE (net income / book equity) + gross profitability (GP / assets) -- currency-neutral
  - value   : earnings yield (E/P) + book yield (B/P) -- GBX-quoted GBP/GBX reporters only
Composite = equal-weight of available component z-scores (momentum always required).
Long top quintile, monthly, net tiered costs, winsorised returns. Compare to momentum alone.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u momentum_multifactor.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from momentum_tradeability import build, tier_spread, cagr, sharpe, maxdd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "eodhd"
FUND = DATA / "eodhd_uk_fundamentals_panel.parquet"
N, FRAC, BT_START = 350, 0.2, "2001-01-01"
OUT = HERE / "momentum_multifactor_results.txt"
_lines: list[str] = []


def say(s=""):
    print(s, flush=True); _lines.append(s)


def asof(panel, field, mi):
    """Point-in-time raster: latest reported value known at each month-end.

    NB reindex(mi, method="ffill") on a DataFrame is WRONG here -- it maps each
    month to a single most-recent filing ROW, so a month only shows the few names
    that filed on that exact date. We must forward-fill DOWN EACH COLUMN: reindex
    to the union of filing dates + month-ends, ffill per column, then select months.
    """
    piv = panel.pivot_table(index="avail_date", columns="code", values=field, aggfunc="last").sort_index()
    return piv.reindex(piv.index.union(mi)).ffill().reindex(mi)


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd and np.isfinite(sd) else s * 0.0


def run_multi(M, mret, liq, signal, comps: dict):
    """comps: dict name->raster(month x code); 'mom' required. Long top quintile of the
    equal-weight z-composite of available components, monthly, net tiered costs."""
    months = M.index
    lo_w = None
    r_out, uni_out, idx = [], [], []
    for i in range(1, len(months)):
        t, th = months[i - 1], months[i]
        elig = liq.loc[t].dropna().nlargest(N).index
        r_e = mret.loc[th, elig].clip(lower=-1.0)
        r_e = r_e.clip(r_e.quantile(0.01), r_e.quantile(0.99))
        # composite z over eligible. The 'mom' slot is the REQUIRED component
        # (usually momentum, but "X only" books pass X here to isolate it).
        zs = []
        req = zscore(comps["mom"].loc[t, elig])
        zs.append(req)
        for nm, ras in comps.items():
            if nm == "mom":
                continue
            zs.append(zscore(ras.loc[t, elig]))
        Z = pd.concat(zs, axis=1)
        comp = Z.mean(axis=1, skipna=True)
        comp = comp[req.notna()]                         # require the mandatory component
        comp = comp.reindex(r_e.dropna().index).dropna()
        if len(comp) < 30:
            continue
        k = max(1, int(len(comp) * FRAC))
        top = comp.nlargest(k).index
        new_w = pd.Series(1.0 / k, index=top)
        sp = tier_spread(liq.loc[t, elig].dropna())
        allc = new_w.index if lo_w is None else new_w.index.union(lo_w.index)
        nw = new_w.reindex(allc).fillna(0.0)
        ow = (nw * 0.0) if lo_w is None else lo_w.reindex(allc).fillna(0.0)
        cost = float(((nw - ow).abs() * sp.reindex(allc).fillna(sp.median()) / 2 / 1e4).sum())
        lo_w = new_w
        r_out.append(float(r_e.reindex(top).mean()) - cost)
        uni_out.append(float(r_e.mean())); idx.append(th)
    P = pd.DataFrame({"r": r_out, "uni": uni_out}, index=pd.DatetimeIndex(idx))
    return P[P.index >= BT_START]


def build_factors(M):
    """Point-in-time QUALITY and VALUE rasters (month x M.columns) on the price panel.

    quality = 0.5*rank(ROE) + 0.5*rank(GP/assets), currency-neutral (covers ~all names).
    value   = 0.5*rank(B/P) + 0.5*rank(E/P), GBX-quoted GBP/GBX reporters only (price and
    statement must share currency; mega-cap USD reporters are excluded -> thinner coverage).
    """
    mi = M.index
    f = pd.read_parquet(FUND)
    book, ni = asof(f, "book_equity", mi), asof(f, "net_income", mi)
    gp, assets, shares = asof(f, "gross_profit", mi), asof(f, "total_assets", mi), asof(f, "shares", mi)
    ccy = f.sort_values("avail_date").groupby("code").agg(stmt=("stmt_ccy", "last"), quote=("quote_ccy", "last"))
    cols = book.columns
    inf = [np.inf, -np.inf]
    roe = (ni / book).replace(inf, np.nan).where(book > 0)              # quality (currency-neutral)
    gpa = (gp / assets).replace(inf, np.nan).where(assets > 0)
    stmt_fac = ccy["stmt"].map({"GBP": 100.0, "GBX": 1.0}).reindex(cols)
    gbx = ccy["quote"].map(lambda c: c == "GBX").reindex(cols).fillna(False)
    mcap = M.reindex(columns=cols) * shares.reindex(columns=cols)        # pence mcap (valid where GBX quote)
    bp = (book.mul(stmt_fac, axis=1) / mcap).replace(inf, np.nan).where(book > 0)
    ep = (ni.mul(stmt_fac, axis=1) / mcap).replace(inf, np.nan)
    bad = [c for c in cols if not (bool(gbx.get(c, False)) and pd.notna(stmt_fac.get(c)))]
    bp[bad] = np.nan; ep[bad] = np.nan                                   # value: GBX-quoted GBP/GBX reporters only
    quality = (0.5 * roe.rank(axis=1, pct=True) + 0.5 * gpa.rank(axis=1, pct=True)).reindex(columns=M.columns)
    value = (0.5 * bp.rank(axis=1, pct=True) + 0.5 * ep.rank(axis=1, pct=True)).reindex(columns=M.columns)
    return quality, value, len(cols)


def main():
    say("Building price panel + point-in-time factor rasters...")
    M, mret, liq, signal = build()
    quality, value, ncols = build_factors(M)
    say(f"factor coverage (names with a value in 2020): quality {quality.loc['2020-06':'2020-07'].iloc[0].notna().sum()}, "
        f"value {value.loc['2020-06':'2020-07'].iloc[0].notna().sum()} of {ncols}\n")

    books = {
        "momentum only": {"mom": signal},
        "mom + quality": {"mom": signal, "q": quality},
        "mom + value": {"mom": signal, "v": value},
        "mom + value + quality": {"mom": signal, "v": value, "q": quality},
        "quality only": {"mom": quality},   # factor goes in the required slot to isolate it
        "value only": {"mom": value},
    }
    base = run_multi(M, mret, liq, signal, {"mom": signal})
    uni = base["uni"]
    say("#" * 82)
    say("# 12-1 MOMENTUM + VALUE/QUALITY (survivorship-free top-350, net tiered costs)")
    say("#" * 82)
    say(f"eligible-universe EW B&H: CAGR {100*cagr(uni):+.2f}%  Sharpe {sharpe(uni):.2f}\n")
    say(f"{'strategy':24} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}")
    for name, comps in books.items():
        P = run_multi(M, mret, liq, signal, comps)
        r = P["r"]
        say(f"{name:24} {100*cagr(r):>+6.2f}% {sharpe(r):>7.2f} {100*maxdd(r):>6.0f}%")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
