"""Vol CHANGE as a selection factor: does a shift in volatility predict momentum won't persist?

Hypothesis (Lance): a change in a name's volatility signals its regime may be shifting, so its
recent momentum is less likely to continue -- deprioritise it. Two readings of "change", each
built from a short/long vol ratio (causal):
  - SIGNED  : penalise vol INCREASES  (favour calming/stable)          factor = z(-ratio)
  - ABSOLUTE: penalise ANY change up or down (favour stable)           factor = z(-|ln ratio|)
Windows: short 3m or 6m vs long 12m. Added to mom+quality, equal-weight, net tiered costs.
Compared to mom+quality (incumbent) and mom+quality+lowvol (vol LEVEL, previous experiment).
Judged by Sharpe + pre/post-2013 consistency.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u momentum_vol_change.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from momentum_tradeability import build, tier_spread, cagr, sharpe, maxdd, N as UNIV_N, BT_START
from momentum_multifactor import build_factors, zscore
from momentum_vol_overlay import run

HERE = Path(__file__).resolve().parent
OUT = HERE / "momentum_vol_change_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True); _lines.append(s)


def line(label, P):
    r = P["r"]; pre, post = r[r.index < "2013-01-01"], r[r.index >= "2013-01-01"]
    say(f"  {label:40} {100*cagr(r):>+6.2f}% {sharpe(r):>6.2f} {100*maxdd(r):>5.0f}%  "
        f"pre {100*cagr(pre):>+5.1f}%  post {100*cagr(post):>+5.1f}%")


def main() -> None:
    say("Building panel + rasters...")
    M, mret, liq, signal = build()
    quality, _v, _ = build_factors(M)
    vol = mret.rolling(12).std().shift(1)
    v3 = mret.rolling(3).std().shift(1)
    v6 = mret.rolling(6).std().shift(1)
    v12 = mret.rolling(12).std().shift(1)
    eps = 1e-9
    r312 = (v3 + eps) / (v12 + eps)
    r612 = (v6 + eps) / (v12 + eps)

    signed312 = -r312                              # higher = calming (vol falling vs year)
    signed612 = -r612
    abs312 = -(np.log(r312)).abs()                 # higher = most stable (ratio near 1)
    abs612 = -(np.log(r612)).abs()
    lowvol = -vol                                  # LEVEL (reference, previous experiment)

    say(f"{'':42} {'CAGR':>7} {'Sharpe':>6} {'maxDD':>6}   OOS split")
    say("#" * 94)
    say("# VOL-CHANGE as a selection factor, added to mom+quality (equal-weight, net costs)")
    say("#" * 94)
    line("mom+quality (incumbent)", run(M, mret, liq, [signal, quality], vol, "ew"))
    say("  -- signed (penalise vol INCREASES) --")
    line("+ signed change 3m/12m", run(M, mret, liq, [signal, quality, signed312], vol, "ew"))
    line("+ signed change 6m/12m", run(M, mret, liq, [signal, quality, signed612], vol, "ew"))
    say("  -- absolute (penalise ANY change) --")
    line("+ abs change 3m/12m", run(M, mret, liq, [signal, quality, abs312], vol, "ew"))
    line("+ abs change 6m/12m", run(M, mret, liq, [signal, quality, abs612], vol, "ew"))
    say("  -- reference: vol LEVEL (previous experiment) --")
    line("+ lowvol (level)", run(M, mret, liq, [signal, quality, lowvol], vol, "ew"))

    # also test the sign the other way, as a falsification check
    say("\n  -- sign check: FAVOUR vol increases (opposite of hypothesis) --")
    line("+ FAVOUR increases 3m/12m", run(M, mret, liq, [signal, quality, r312], vol, "ew"))

    # GATE form: EXCLUDE the biggest vol-changers, then rank survivors by mom+quality (fixed 69)
    say("\n" + "#" * 94)
    say("# GATE: drop the biggest vol-changers, then pick top-69 by mom+quality (matches intent)")
    say("#" * 94)

    def run_gated(drop_frac):
        change = np.log(r612).abs()                 # magnitude of 6m/12m vol change
        months = M.index; lo = None; r_out, uni, idx = [], [], []
        for i in range(1, len(months)):
            t, th = months[i - 1], months[i]
            elig = liq.loc[t].dropna().nlargest(UNIV_N).index
            r_e = mret.loc[th, elig].clip(lower=-1.0)
            r_e = r_e.clip(r_e.quantile(0.01), r_e.quantile(0.99))
            ch = change.loc[t, elig].dropna()
            keep = ch.nsmallest(int(len(ch) * (1 - drop_frac))).index if drop_frac > 0 else elig
            comp = pd.concat([zscore(signal.loc[t, keep]), zscore(quality.loc[t, keep])], axis=1).mean(axis=1, skipna=True)
            comp = comp[zscore(signal.loc[t, keep]).notna()].reindex(r_e.dropna().index).dropna()
            if len(comp) < 30:
                continue
            k = min(69, len(comp)); top = comp.nlargest(k).index
            nw_ = pd.Series(1.0 / k, index=top)
            sp = tier_spread(liq.loc[t, elig].dropna())
            allc = nw_.index if lo is None else nw_.index.union(lo.index)
            nw = nw_.reindex(allc).fillna(0.0); ow = (nw * 0.0) if lo is None else lo.reindex(allc).fillna(0.0)
            cost = float(((nw - ow).abs() * sp.reindex(allc).fillna(sp.median()) / 2 / 1e4).sum())
            lo = nw_
            r_out.append(float(r_e.reindex(top).mean()) - cost); uni.append(float(r_e.mean())); idx.append(th)
        P = pd.DataFrame({"r": r_out, "uni": uni}, index=pd.DatetimeIndex(idx))
        return P[P.index >= BT_START]

    line("no gate (top-69 mom+quality)", run_gated(0.0))
    line("drop biggest-change quartile", run_gated(0.25))
    line("drop biggest-change third", run_gated(0.33))
    line("drop biggest-change half", run_gated(0.50))

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
