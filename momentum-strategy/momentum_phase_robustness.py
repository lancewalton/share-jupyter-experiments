"""Calendar-PHASE robustness: is the edge an artefact of rebalancing on month-ends?

The whole strategy resamples to month-end ("ME"). If you had started it on a different
day of the month, every rebalance would land on a different date. A real edge must not
depend on that phase. We rebuild the monthly grid anchored on different DAYS OF THE MONTH
(1st, 4th, ... 28th) -- one rebalance per month at that day, priced as-of the last trading
day on/before it -- and re-run the headline builds on each phase:
  - long-only 12-1 momentum tilt, net tiered costs
  - mom + quality, net tiered costs
Report CAGR / Sharpe / maxDD per phase, then the spread across phases. Stable => robust.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u momentum_phase_robustness.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_tradeability import DATA, OHLCV, cagr, sharpe, maxdd, run
from momentum_multifactor import build_factors, run_multi

HERE = Path(__file__).resolve().parent
SHIFTS = list(range(0, 28, 3))   # calendar-phase shifts in days (0 == month-end baseline)
OUT = HERE / "momentum_phase_robustness_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True); _lines.append(s)


def load_daily():
    """Daily adjusted-close and (currency-adjusted) turnover panels."""
    df = pd.read_parquet(OHLCV, columns=["date", "code", "close", "adjusted_close", "volume"])
    df = df[df["date"] >= "1998-01-01"]
    ccy = {}
    for fn in ("lse_active.json", "lse_delisted.json"):
        for x in json.load(open(DATA / fn)):
            ccy.setdefault(x.get("Code"), x.get("Currency"))
    fac = df["code"].map(lambda c: 0.01 if ccy.get(c) == "GBX" else 1.0).astype("float32")
    df["turnover"] = df["close"] * df["volume"] * fac
    adj = df.pivot(index="date", columns="code", values="adjusted_close").sort_index()
    turn = df.pivot(index="date", columns="code", values="turnover").sort_index()
    return adj, turn


def phase_panel(adj, turn, k):
    """build()'s resample("ME").last() at a different calendar phase: shift the daily index
    back by k days first, so the monthly sampling point moves ~k days earlier in the month
    (k=0 == month-end baseline). resample.last() yields NaN for an empty bin, so delisted
    names correctly drop out -- unlike reindex-ffill, which would zombie them at a stale price.
    ('offset=' is silently ignored for the non-tick 'ME' frequency, hence the index shift.)"""
    shift = pd.Timedelta(days=k)
    A = adj.set_axis(adj.index - shift)
    T = turn.set_axis(turn.index - shift)
    M = A.resample("ME").last()
    mret = M.pct_change()
    signal = M.shift(1) / M.shift(12) - 1          # 12-1 momentum on the monthly grid
    liq = T.resample("ME").mean().rolling(12).mean()
    return M, mret, liq, signal


def main() -> None:
    say("Loading daily panels...")
    adj, turn = load_daily()
    say(f"{adj.shape[1]} names x {adj.shape[0]} trading days; testing {len(SHIFTS)} calendar phases (day-shift; 0=month-end)\n")

    for build_label, factor_book in (("long-only 12-1 momentum tilt", None),
                                     ("mom + quality", "quality")):
        say("#" * 80)
        say(f"# PHASE ROBUSTNESS: {build_label} (net tiered costs)")
        say("#" * 80)
        say(f"{'shift':>5} {'#rebal':>7} {'CAGR':>8} {'Sharpe':>7} {'maxDD':>7} {'B&H CAGR':>9}")
        rows = []
        for o in SHIFTS:
            M, mret, liq, signal = phase_panel(adj, turn, o)
            if factor_book is None:
                P, _ = run(M, mret, liq, signal, 1, "tier", 0.0)
                r, uni = P["lo"], P["uni"]
            else:
                quality, _value, _ = build_factors(M)
                P = run_multi(M, mret, liq, signal, {"mom": signal, "q": quality})
                r, uni = P["r"], P["uni"]
            c, s, d, bh = cagr(r), sharpe(r), maxdd(r), cagr(uni)
            rows.append((c, s, d, bh))
            say(f"{o:>5} {len(r):>7} {100*c:>+7.2f}% {s:>7.2f} {100*d:>6.0f}% {100*bh:>+8.2f}%")
        C = np.array([x[0] for x in rows]); S = np.array([x[1] for x in rows])
        D = np.array([x[2] for x in rows]); B = np.array([x[3] for x in rows])
        say(f"\n  across phases: CAGR {100*C.mean():+.2f}% +/- {100*C.std():.2f}pp  "
            f"(range {100*C.min():+.2f}..{100*C.max():+.2f})")
        say(f"                Sharpe {S.mean():.2f} +/- {S.std():.2f}  (range {S.min():.2f}..{S.max():.2f})")
        say(f"                maxDD {100*D.mean():.0f}% +/- {100*D.std():.1f}pp  (range {100*D.min():.0f}..{100*D.max():.0f})")
        say(f"  beats B&H on CAGR in {int((C > B).sum())}/{len(rows)} phases "
            f"(mean edge {100*(C-B).mean():+.2f}pp)\n")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"saved -> {OUT.name}")


if __name__ == "__main__":
    main()
