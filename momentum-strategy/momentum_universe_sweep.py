"""Widen the universe and take a smaller fraction of the best -- does it help?

Currently: top-350 by liquidity, top-quintile (~69 names). This sweeps the universe size N
(deeper into less-liquid names) under two rules: (a) fixed fraction 0.2, (b) fixed count ~69
from the wider pool. Momentum-only, net tiered costs. Reports CAGR/Sharpe/maxDD/turnover, avg
names held, and the liquidity of the *marginal* eligible name (a capacity/realism flag).

CAVEAT: the tiered cost model caps at 80 bps (low tier). Real micro-cap spreads are far wider
and bad ticks worse, so any GROSS gain from a wide N is optimistic -- read net, and note the
marginal-liquidity column.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u momentum_universe_sweep.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from momentum_tradeability import build, tier_spread, cagr, sharpe, maxdd, BT_START

HERE = Path(__file__).resolve().parent
NS = [200, 350, 500, 750, 1000, 1500]
OUT = HERE / "momentum_universe_sweep_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True); _lines.append(s)


def run(M, mret, liq, signal, N, frac=None, count=None):
    """Long-only momentum tilt over top-N liquid names; hold frac*elig or a fixed count."""
    months = M.index
    lo = None
    r_out, uni, idx, tw, held, marg = [], [], [], [], [], []
    for i in range(1, len(months)):
        t, th = months[i - 1], months[i]
        lrow = liq.loc[t].dropna()
        elig = lrow.nlargest(N).index
        if len(elig) < 30:
            continue
        marg.append(float(lrow.nlargest(N).iloc[-1]))          # turnover of the least-liquid held-eligible name
        r_e = mret.loc[th, elig].clip(lower=-1.0)
        r_e = r_e.clip(r_e.quantile(0.01), r_e.quantile(0.99))
        s = signal.loc[t, elig].dropna().reindex(r_e.dropna().index).dropna()
        if len(s) < 30:
            continue
        k = count if count is not None else max(1, int(len(s) * frac))
        k = min(k, len(s))
        top = s.nlargest(k).index
        nw_ = pd.Series(1.0 / k, index=top)
        sp = tier_spread(lrow.reindex(elig).dropna())
        allc = nw_.index if lo is None else nw_.index.union(lo.index)
        nw = nw_.reindex(allc).fillna(0.0)
        ow = (nw * 0.0) if lo is None else lo.reindex(allc).fillna(0.0)
        dz = (nw - ow).abs()
        cost = float((dz * sp.reindex(allc).fillna(sp.median()) / 2 / 1e4).sum())
        if lo is not None:
            tw.append(float(dz.sum()))
        lo = nw_
        r_out.append(float(r_e.reindex(top).mean()) - cost)
        uni.append(float(r_e.mean())); idx.append(th); held.append(k)
    P = pd.DataFrame({"r": r_out, "uni": uni}, index=pd.DatetimeIndex(idx))
    P = P[P.index >= BT_START]
    return P, (np.mean(tw) * 12 if tw else float("nan")), np.mean(held), np.median(marg)


def main() -> None:
    say("Building panel...")
    M, mret, liq, signal = build()
    avail = liq.notna().sum(axis=1)
    say(f"names with a liquidity value: median {int(avail.median())}, max {int(avail.max())} "
        f"(so N beyond that just takes all available)\n")

    for rule, kw in (("fixed fraction 0.2", dict(frac=0.2)), ("fixed count 69", dict(count=69))):
        say("#" * 82)
        say(f"# UNIVERSE SIZE SWEEP -- {rule}, momentum-only, net tiered costs")
        say("#" * 82)
        say(f"{'N':>6} {'#held':>6} {'CAGR':>8} {'Sharpe':>7} {'maxDD':>7} {'turn/yr':>8} "
            f"{'pre13':>7} {'post13':>7} {'margLiq£/day':>13}")
        for N in NS:
            P, turn, avgheld, marg = run(M, mret, liq, signal, N, **kw)
            r = P["r"]; pre, post = r[r.index < "2013-01-01"], r[r.index >= "2013-01-01"]
            say(f"{N:>6} {avgheld:>6.0f} {100*cagr(r):>+7.2f}% {sharpe(r):>7.2f} {100*maxdd(r):>6.0f}% "
                f"{turn:>7.1f}x {100*cagr(pre):>+6.1f}% {100*cagr(post):>+6.1f}% {marg:>12,.0f}")
        say("")

    say("(marginal-liquidity = median trailing daily GBP turnover of the least-liquid eligible name;")
    say(" it collapses as N widens -> the extra names are micro-caps the 80bps cost cap flatters.)")
    OUT.write_text("\n".join(_lines) + "\n")
    say(f"saved -> {OUT.name}")


if __name__ == "__main__":
    main()
