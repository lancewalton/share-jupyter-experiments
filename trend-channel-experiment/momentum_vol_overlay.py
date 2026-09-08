"""Refine entry/weighting with VOLATILITY (linking the vol work to the momentum build).

Two independent uses of each name's trailing volatility (causal 12m std of monthly returns):
  (A) SELECTION: add a low-vol factor  z(-vol)  to the composite (defensive momentum).
  (B) WEIGHTING: weight selected names by 1/vol (risk-parity-ish) instead of equal weight.
Tested on momentum-only and mom+quality, top-quintile, survivorship-free top-350, net tiered
costs. Report CAGR/Sharpe/maxDD + pre/post-2013 to guard against overfit.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u momentum_vol_overlay.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from momentum_tradeability import build, tier_spread, cagr, sharpe, maxdd, N as UNIV_N, BT_START
from momentum_multifactor import build_factors, zscore

HERE = Path(__file__).resolve().parent
OUT = HERE / "momentum_vol_overlay_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True); _lines.append(s)


def run(M, mret, liq, facs, vol, weight_mode, frac=0.2):
    """facs: list of selection rasters (z-averaged). weight_mode: 'ew' or 'invvol'."""
    months = M.index
    lo = None
    r_out, uni, idx = [], [], []
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
        k = max(1, int(len(comp) * frac))
        top = comp.nlargest(k).index
        if weight_mode == "invvol":
            v = vol.loc[t, top].reindex(top)
            iv = (1.0 / v).replace([np.inf, -np.inf], np.nan)
            iv = iv.fillna(iv.median())
            w = iv / iv.sum()
        else:
            w = pd.Series(1.0 / k, index=top)
        sp = tier_spread(liq.loc[t, elig].dropna())
        allc = w.index if lo is None else w.index.union(lo.index)
        nw = w.reindex(allc).fillna(0.0)
        ow = (nw * 0.0) if lo is None else lo.reindex(allc).fillna(0.0)
        dz = (nw - ow).abs()
        cost = float((dz * sp.reindex(allc).fillna(sp.median()) / 2 / 1e4).sum())
        lo = w
        r_out.append(float((r_e.reindex(w.index) * w).sum()) - cost)
        uni.append(float(r_e.mean())); idx.append(th)
    P = pd.DataFrame({"r": r_out, "uni": uni}, index=pd.DatetimeIndex(idx))
    return P[P.index >= BT_START]


def line(label, P):
    r = P["r"]; pre, post = r[r.index < "2013-01-01"], r[r.index >= "2013-01-01"]
    say(f"  {label:38} {100*cagr(r):>+6.2f}% {sharpe(r):>6.2f} {100*maxdd(r):>5.0f}%  "
        f"pre {100*cagr(pre):>+5.1f}%  post {100*cagr(post):>+5.1f}%")


def main() -> None:
    say("Building panel + rasters...")
    M, mret, liq, signal = build()
    quality, _v, _ = build_factors(M)
    vol = mret.rolling(12).std().shift(1)                 # causal trailing monthly vol
    lowvol = -vol                                          # higher = calmer (z of this = low-vol factor)

    say(f"{'':40} {'CAGR':>7} {'Sharpe':>6} {'maxDD':>6}   OOS split")
    say("#" * 92)
    say("# (A) LOW-VOL as a SELECTION factor (equal-weight)")
    say("#" * 92)
    line("momentum (incumbent)", run(M, mret, liq, [signal], vol, "ew"))
    line("momentum + lowvol", run(M, mret, liq, [signal, lowvol], vol, "ew"))
    line("mom + quality (incumbent build)", run(M, mret, liq, [signal, quality], vol, "ew"))
    line("mom + quality + lowvol", run(M, mret, liq, [signal, quality, lowvol], vol, "ew"))

    say("\n" + "#" * 92)
    say("# (B) INVERSE-VOL WEIGHTING (same selection, weight by 1/vol instead of equal)")
    say("#" * 92)
    line("momentum, equal-weight", run(M, mret, liq, [signal], vol, "ew"))
    line("momentum, inverse-vol weight", run(M, mret, liq, [signal], vol, "invvol"))
    line("mom + quality, equal-weight", run(M, mret, liq, [signal, quality], vol, "ew"))
    line("mom + quality, inverse-vol weight", run(M, mret, liq, [signal, quality], vol, "invvol"))

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
