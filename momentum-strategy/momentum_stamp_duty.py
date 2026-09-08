"""How much does UK stamp duty (SDRT) cost the SHARE implementation?

The headline backtest charged only the tiered bid/offer spread (15/40/80 bps). A real UK
cash-share account also pays 0.5% Stamp Duty Reserve Tax on every PURCHASE (buys only, not
sells). This re-runs the key builds with stamp duty added to the cost, so the net figures
reflect a taxable share account. Also shows the drag at higher turnover (concentrated book)
and confirms the edge still clears buy-and-hold.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u momentum_stamp_duty.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from momentum_tradeability import build, tier_spread, cagr, sharpe, maxdd, N as UNIV_N, BT_START
from momentum_multifactor import build_factors, zscore
from vol_target_momentum import vol_target

HERE = Path(__file__).resolve().parent
STAMP_BPS = 50.0            # 0.5% SDRT on purchases (buys only)
OUT = HERE / "momentum_stamp_duty_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True); _lines.append(s)


def run(M, mret, liq, facs, want, stamp_bps):
    """Long-only top-`want` composite tilt. Cost = tiered half-spread on |Δw| + stamp on BUYS."""
    months = M.index
    lo = None
    r_out, uni, idx, buys = [], [], [], []
    for i in range(1, len(months)):
        t, th = months[i - 1], months[i]
        elig = liq.loc[t].dropna().nlargest(UNIV_N).index
        r_e = mret.loc[th, elig].clip(lower=-1.0)
        r_e = r_e.clip(r_e.quantile(0.01), r_e.quantile(0.99))
        zs = [zscore(f.loc[t, elig]) for f in facs]
        comp = pd.concat(zs, axis=1).mean(axis=1, skipna=True)
        comp = comp[zs[0].notna()].reindex(r_e.dropna().index).dropna()
        if len(comp) < 30:
            continue
        k = min(want, len(comp))
        top = comp.nlargest(k).index
        nw_ = pd.Series(1.0 / k, index=top)
        sp = tier_spread(liq.loc[t, elig].dropna())
        allc = nw_.index if lo is None else nw_.index.union(lo.index)
        nw = nw_.reindex(allc).fillna(0.0)
        ow = (nw * 0.0) if lo is None else lo.reindex(allc).fillna(0.0)
        dz = nw - ow
        spread_cost = float((dz.abs() * sp.reindex(allc).fillna(sp.median()) / 2 / 1e4).sum())
        buy = float(dz.clip(lower=0.0).sum())                 # fraction of book purchased
        stamp_cost = buy * stamp_bps / 1e4
        if lo is not None:
            buys.append(buy)
        lo = nw_
        r_out.append(float(r_e.reindex(top).mean()) - spread_cost - stamp_cost)
        uni.append(float(r_e.mean())); idx.append(th)
    P = pd.DataFrame({"r": r_out, "uni": uni}, index=pd.DatetimeIndex(idx))
    P = P[P.index >= BT_START]
    return P, (np.mean(buys) * 12 if buys else float("nan"))


def line(label, P, buys_yr):
    r = P["r"]
    say(f"  {label:34} CAGR {100*cagr(r):>+6.2f}%  Sharpe {sharpe(r):>5.2f}  maxDD {100*maxdd(r):>4.0f}%"
        + (f"  buys/yr {buys_yr:.1f}x" if not np.isnan(buys_yr) else ""))


def main() -> None:
    say("Building panel + quality raster...")
    M, mret, liq, signal = build()
    quality, _v, _ = build_factors(M)
    books = {"momentum only": [signal], "mom + quality": [signal, quality]}

    say("#" * 84)
    say(f"# STAMP DUTY IMPACT (0.5% on buys) -- share account, net tiered spread + SDRT")
    say("#" * 84)
    uniP, _ = run(M, mret, liq, [signal], 69, 0.0)
    say(f"eligible-universe EW B&H: CAGR {100*cagr(uniP['uni']):+.2f}%  Sharpe {sharpe(uniP['uni']):.2f}\n")

    for name, facs in books.items():
        say(f"{name} (top quintile, ~69 names):")
        P0, b0 = run(M, mret, liq, facs, 69, 0.0)
        Ps, bs = run(M, mret, liq, facs, 69, STAMP_BPS)
        line("spread only (as reported)", P0, b0)
        line("+ stamp duty (0.5% on buys)", Ps, bs)
        say(f"  {'':34} stamp drag: {100*(cagr(P0['r'])-cagr(Ps['r'])):.2f} pp/yr\n")

    say("Concentrated book (20 names, higher turnover -> more stamp drag):")
    for name, facs in books.items():
        P0, b0 = run(M, mret, liq, facs, 20, 0.0)
        Ps, bs = run(M, mret, liq, facs, 20, STAMP_BPS)
        say(f"  {name}: {100*cagr(P0['r']):+.2f}% -> {100*cagr(Ps['r']):+.2f}%  "
            f"(drag {100*(cagr(P0['r'])-cagr(Ps['r'])):.2f}pp, buys {bs:.1f}x/yr)")

    say("\nBest build -- mom+quality vol-targeted (tilt net spread+stamp, then vol-target):")
    Ps, _ = run(M, mret, liq, [signal, quality], 69, STAMP_BPS)
    r = Ps["r"]; bv = r.std() * np.sqrt(12)
    for cap, lab in ((1.0, "no leverage"), (1.5, "lev<=1.5")):
        vt, _w = vol_target(r, bv, cap)
        say(f"  {lab:12} CAGR {100*cagr(vt):>+6.2f}%  Sharpe {sharpe(vt):.2f}  maxDD {100*maxdd(vt):.0f}%")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
