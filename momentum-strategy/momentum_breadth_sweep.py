"""How sensitive is the recommended build to the NUMBER OF NAMES HELD?

The default holds ~69 names -- not a chosen number, but floor(0.20 x ~347 eligible). This
sweeps the holding count directly (top-M of the mom+quality composite) to see the
concentration/diversification trade-off: CAGR, Sharpe, maxDD, turnover, avg #names.
Long-only, survivorship-free top-350, net tiered costs (same engine as the headline build).

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u momentum_breadth_sweep.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from momentum_tradeability import build, tier_spread, cagr, sharpe, maxdd, N as UNIV_N, BT_START
from momentum_multifactor import build_factors, zscore

HERE = Path(__file__).resolve().parent
COUNTS = [20, 30, 50, 69, 100, 150, 200]     # absolute holding counts to test
OUT = HERE / "momentum_breadth_sweep_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True); _lines.append(s)


def run_count(M, mret, liq, signal, quality, want):
    """Long-only mom+quality tilt holding the top `want` names of the composite."""
    months = M.index
    lo_w = None
    r_out, uni_out, idx, tw = [], [], [], []
    for i in range(1, len(months)):
        t, th = months[i - 1], months[i]
        elig = liq.loc[t].dropna().nlargest(UNIV_N).index
        r_e = mret.loc[th, elig].clip(lower=-1.0)
        r_e = r_e.clip(r_e.quantile(0.01), r_e.quantile(0.99))
        z_mom = zscore(signal.loc[t, elig])
        comp = pd.concat([z_mom, zscore(quality.loc[t, elig])], axis=1).mean(axis=1, skipna=True)
        comp = comp[z_mom.notna()].reindex(r_e.dropna().index).dropna()
        if len(comp) < 30:
            continue
        k = min(want, len(comp))
        top = comp.nlargest(k).index
        new_w = pd.Series(1.0 / k, index=top)
        sp = tier_spread(liq.loc[t, elig].dropna())
        allc = new_w.index if lo_w is None else new_w.index.union(lo_w.index)
        nw = new_w.reindex(allc).fillna(0.0)
        ow = (nw * 0.0) if lo_w is None else lo_w.reindex(allc).fillna(0.0)
        dz = (nw - ow).abs()
        cost = float((dz * sp.reindex(allc).fillna(sp.median()) / 2 / 1e4).sum())
        if lo_w is not None:
            tw.append(float(dz.sum()))
        lo_w = new_w
        r_out.append(float(r_e.reindex(top).mean()) - cost)
        uni_out.append(float(r_e.mean())); idx.append(th)
    P = pd.DataFrame({"r": r_out, "uni": uni_out}, index=pd.DatetimeIndex(idx))
    P = P[P.index >= BT_START]
    turn = np.mean(tw) * 12 if tw else float("nan")
    return P, turn


def main() -> None:
    say("Building panel + quality raster...")
    M, mret, liq, signal = build()
    quality, _v, _ = build_factors(M)
    say(f"universe top-{UNIV_N}; sweeping holding counts {COUNTS}\n")

    say("#" * 82)
    say("# HOLDING-COUNT SWEEP -- mom+quality long-only tilt (net tiered costs)")
    say("#" * 82)
    say(f"{'#held':>6} {'~pctile':>8} {'CAGR':>8} {'Sharpe':>7} {'maxDD':>7} {'turn/yr':>8}")
    for want in COUNTS:
        P, turn = run_count(M, mret, liq, signal, quality, want)
        r = P["r"]
        pct = 100 * want / 347
        say(f"{want:>6} {pct:>7.0f}% {100*cagr(r):>+7.2f}% {sharpe(r):>7.2f} {100*maxdd(r):>6.0f}% {turn:>7.1f}x")
    say(f"\n  (reference: eligible-universe EW B&H CAGR {100*cagr(P['uni']):+.2f}%, Sharpe {sharpe(P['uni']):.2f})")
    say("  default build holds ~69 (= floor(0.20 x ~347 eligible)).")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
