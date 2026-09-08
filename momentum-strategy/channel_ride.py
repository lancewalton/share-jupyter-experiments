"""Ride-the-winner exit: buy the bottom of a rising channel, then HOLD until price
breaks the BOTTOM of the channel (lower band) by BREAK_FRAC of the channel height
- optionally also exit if the channel stops rising (gradient < 0). No sell at the
top, so winners ride the whole trend; cash is rotated on exit.

This directly attacks the winner-capping that sank the sell-at-the-top version.
Tested in the rotation portfolio vs equal-weight buy-and-hold, and by whether it
now captures the big multi-year winners.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u channel_ride.py
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import load_ftse
from channel_portfolio import run_portfolio
from channel_ratchet import ENTRY_FRAC, FIN, G_MIN, L, SPREAD, precompute

HERE = Path(__file__).resolve().parent
BARS_YR = 252
OUT = HERE / "channel_ride_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def trades_for_ride(df, arr, break_frac: float, grad_exit: bool):
    up, lo, ann = arr
    close = df["close"].to_numpy()
    dates = df.index
    n = len(df)
    out = []
    pos = False
    e = grad = None
    for i in range(L, n - 1):
        u, l, a = up[i], lo[i], ann[i]
        if not np.isfinite(u):
            continue
        w = u - l
        c = close[i]
        if not pos:
            if a >= G_MIN and w > 0 and c <= l + ENTRY_FRAC * w:
                pos, e, grad = True, i + 1, a
        else:
            if (c < l - break_frac * w) or (grad_exit and a < 0):
                out.append((dates[e], dates[i + 1], grad)); pos = False
    if pos:
        out.append((dates[e], dates[n - 1], grad))
    return out


def main() -> None:
    names = load_ftse.universe()
    say(f"Loading {len(names)} names; precomputing channels...")
    data, closes = {}, {}
    for t in names:
        try:
            df = load_ftse.load(t)
        except Exception:
            continue
        arr = precompute(df, L)
        if arr is None:
            continue
        data[t] = (df, arr)
        closes[t] = df["close"]
    say(f"{len(data)} names.\n")

    dates = pd.DatetimeIndex(sorted(set().union(*[set(closes[t].index) for t in closes])))
    retmap = {t: {d: float(v) for d, v in closes[t].pct_change().items() if np.isfinite(v)} for t in closes}
    ew = pd.DataFrame(closes).reindex(dates).pct_change().mean(axis=1).fillna(0.0).to_numpy()
    ew_total = np.prod(1 + ew) - 1
    yrs = len(dates) / BARS_YR
    ew_cagr = (1 + ew_total) ** (1 / yrs) - 1
    ew_sh = float(np.mean(ew) / np.std(ew) * np.sqrt(BARS_YR))
    say(f"equal-weight B&H: total {100*ew_total:+.0f}%  CAGR {100*ew_cagr:+.2f}%  Sharpe {ew_sh:+.2f}\n")

    say("#" * 90)
    say("# RIDE-THE-WINNER portfolio vs B&H  (exit on lower-band break by BREAK_FRAC of height)")
    say("#" * 90)
    say(f"{'break':>6} {'gradExit':>9} {'slots':>6} {'total%':>9} {'CAGR%':>7} {'Sharpe':>7} "
        f"{'expo':>6} {'medHold_d':>10}  vs B&H")
    best = None
    for grad_exit in (True, False):
        for bf in (0.0, 0.5, 1.0):
            ebd = defaultdict(list)
            holds = []
            for t, (df, arr) in data.items():
                for ed, xd, grad in trades_for_ride(df, arr, bf, grad_exit):
                    ebd[ed].append((grad, t, xd))
                    holds.append((xd - ed).days)
            medhold = float(np.median(holds)) if holds else float("nan")
            for S in (5, 10, 20):
                res = run_portfolio(ebd, retmap, dates, S)
                cagr = (1 + res["total"]) ** (1 / yrs) - 1
                beat = "WINS" if res["total"] > ew_total else "loses"
                say(f"{bf:>6} {str(grad_exit):>9} {S:>6} {100*res['total']:>9.0f} {100*cagr:>7.2f} "
                    f"{res['sharpe']:>7.2f} {100*res['exposure']:>5.0f}% {medhold:>10.0f}  {beat}")
                if best is None or res["total"] > best[0]:
                    best = (res["total"], bf, grad_exit, S)

    # winner capture for the best config
    bf, ge, S = best[1], best[2], best[3]
    say(f"\nBest: break={bf}, gradExit={ge}, slots={S} (total {100*best[0]:+.0f}% vs B&H {100*ew_total:+.0f}%)")
    say("\n" + "#" * 74)
    say(f"# Winner capture under the RIDE exit (break={bf}, gradExit={ge}) — top-10 B&H names")
    say("#" * 74)
    per = {}
    for t, (df, arr) in data.items():
        close = df["close"].to_numpy(); dts = df.index; pos = {d: i for i, d in enumerate(dts)}
        bh = close[-1] / close[0] - 1.0
        mult = 1.0
        for ed, xd, _ in trades_for_ride(df, arr, bf, ge):
            ei, xi = pos[ed], pos[xd]
            days = max(1, (xd - ed).days)
            mult *= (1 + close[xi] / close[ei] - 1 - 2 * SPREAD - FIN * days / 365)
        per[t] = (bh, mult - 1.0)
    say(f"    {'name':6} {'B&H%':>9} {'strat%':>9} {'capture':>8}")
    for t in sorted(per, key=lambda k: per[k][0], reverse=True)[:10]:
        bh, st = per[t]
        say(f"    {t:6} {100*bh:>9.0f} {100*st:>9.0f} {st/bh if bh>0 else float('nan'):>8.2f}")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
