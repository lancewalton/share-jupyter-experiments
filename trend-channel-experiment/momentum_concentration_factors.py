"""Does VALUE earn its place in a CONCENTRATED book? (breadth x factor interaction)

At the wide (~69-name) quintile, adding value gives the best Sharpe (mom+value+quality 0.84).
This tests whether that survives concentration: run each factor book at 20 / 30 / 69 names,
net tiered costs, long-only. Also vol-targets the concentrated three-factor book (the
maximum-aggression corner). Finding: value needs breadth; quality suits concentration.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u momentum_concentration_factors.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from momentum_tradeability import build, tier_spread, cagr, sharpe, maxdd, N as UNIV_N, BT_START
from momentum_multifactor import build_factors, zscore
from vol_target_momentum import vol_target

HERE = Path(__file__).resolve().parent
COUNTS = (20, 30, 69)
OUT = HERE / "momentum_concentration_factors_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True); _lines.append(s)


def run_book(M, mret, liq, facs, want):
    """facs: list of rasters to z-average (facs[0] is the mandatory momentum slot).
    Long the top `want` names of the composite, monthly, net tiered costs."""
    months = M.index
    lo = None
    r_out, uni, idx, tw = [], [], [], []
    for i in range(1, len(months)):
        t, th = months[i - 1], months[i]
        elig = liq.loc[t].dropna().nlargest(UNIV_N).index
        r_e = mret.loc[th, elig].clip(lower=-1.0)
        r_e = r_e.clip(r_e.quantile(0.01), r_e.quantile(0.99))
        zs = [zscore(f.loc[t, elig]) for f in facs]
        req = zs[0]
        comp = pd.concat(zs, axis=1).mean(axis=1, skipna=True)
        comp = comp[req.notna()].reindex(r_e.dropna().index).dropna()
        if len(comp) < 30:
            continue
        k = min(want, len(comp))
        top = comp.nlargest(k).index
        nw_ = pd.Series(1.0 / k, index=top)
        sp = tier_spread(liq.loc[t, elig].dropna())
        allc = nw_.index if lo is None else nw_.index.union(lo.index)
        nw = nw_.reindex(allc).fillna(0.0)
        ow = (nw * 0.0) if lo is None else lo.reindex(allc).fillna(0.0)
        dz = (nw - ow).abs()
        cost = float((dz * sp.reindex(allc).fillna(sp.median()) / 2 / 1e4).sum())
        if lo is not None:
            tw.append(float(dz.sum()))
        lo = nw_
        r_out.append(float(r_e.reindex(top).mean()) - cost)
        uni.append(float(r_e.mean())); idx.append(th)
    P = pd.DataFrame({"r": r_out, "uni": uni}, index=pd.DatetimeIndex(idx))
    P = P[P.index >= BT_START]
    return P["r"], (np.mean(tw) * 12 if tw else float("nan"))


def main() -> None:
    say("Building panel + factor rasters...")
    M, mret, liq, signal = build()
    quality, value, _ = build_factors(M)
    books = {
        "mom+quality":       [signal, quality],
        "mom+value":         [signal, value],
        "mom+value+quality": [signal, value, quality],
    }

    say("#" * 74)
    say("# BREADTH x FACTOR: does value earn its place when concentrated? (net costs)")
    say("#" * 74)
    say(f"{'book':20} {'#held':>5} {'CAGR':>8} {'Sharpe':>7} {'maxDD':>7} {'turn/yr':>8}")
    for name, facs in books.items():
        for want in COUNTS:
            r, turn = run_book(M, mret, liq, facs, want)
            say(f"{name:20} {want:>5} {100*cagr(r):>+7.2f}% {sharpe(r):>7.2f} {100*maxdd(r):>6.0f}% {turn:>7.1f}x")
        say("")

    say("--- vol-target on 20-name mom+value+quality (maximum-aggression corner) ---")
    r20, _ = run_book(M, mret, liq, books["mom+value+quality"], 20)
    bv = r20.std() * np.sqrt(12)
    for cap, lab in ((1.0, "no leverage"), (1.5, "lev<=1.5")):
        vt, w = vol_target(r20, bv, cap)
        say(f"  {lab:12} CAGR {100*cagr(vt):>+6.2f}%  Sharpe {sharpe(vt):.2f}  "
            f"maxDD {100*maxdd(vt):.0f}%  avg exposure {w.reindex(vt.index).mean():.2f}")

    say("\nRule: VALUE needs BREADTH (best Sharpe diversified, ~69), QUALITY suits CONCENTRATION.")
    OUT.write_text("\n".join(_lines) + "\n")
    say(f"saved -> {OUT.name}")


if __name__ == "__main__":
    main()
