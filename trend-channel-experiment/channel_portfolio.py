"""Fix 1: capital rotation. A per-name channel trade sits in cash ~68% of the time,
but a portfolio redeploys that cash into the next name at the bottom of its channel.
The right benchmark is not per-name total return but whether a continuously-rotated
book beats buy-and-hold of the whole universe.

Scheduler: S slots. Each day, active positions earn their name's return (equal
weight 1/S); when a slot frees, fill it with the highest-gradient available entry
signal (a name whose channel qualifies and whose close is in the bottom quarter).
Entry/exit follow the corrected per-name rule (no premature disaster stop). Costs:
10bps/side + 5%/yr financing on the deployed weight.

Benchmark: equal-weight buy-and-hold of all 120 FTSE names (always fully invested).
Also reports the key diagnostic Lance's argument rests on: return per year *while
deployed* vs the market's drift rate.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u channel_portfolio.py
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import load_ftse
from channel_ratchet import (ENTRY_FRAC, FIN, G_MIN, L, SPREAD, precompute)

HERE = Path(__file__).resolve().parent
BARS_YR = 252
SLOTS = [1, 3, 5, 10, 20]
OUT = HERE / "channel_portfolio_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def trades_for(df: pd.DataFrame, arrays) -> list[tuple]:
    """Per-name trades (entry_date, exit_date, entry_gradient) under the corrected
    top-baseline rule: enter bottom quarter of a qualifying channel, exit top
    quarter or when the gradient turns negative."""
    up, lo, ann = arrays
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
            if a < 0 or c >= u - ENTRY_FRAC * w:
                out.append((dates[e], dates[i + 1], grad)); pos = False
    if pos:
        out.append((dates[e], dates[n - 1], grad))
    return out


def run_portfolio(entries_by_date, retmap, dates, S: int) -> dict:
    active: dict[str, pd.Timestamp] = {}
    port = np.zeros(len(dates))
    exposure = np.zeros(len(dates))
    n_taken = 0
    for k, t in enumerate(dates):
        day = 0.0
        for name in active:
            day += (retmap[name].get(t, 0.0) - FIN / 365) / S
        for name in [nm for nm, xd in active.items() if xd == t]:
            day -= SPREAD / S
            del active[name]
        free = S - len(active)
        if free > 0 and t in entries_by_date:
            for grad, name, xd in sorted(entries_by_date[t], reverse=True):
                if free <= 0:
                    break
                if name in active or xd <= t:
                    continue
                active[name] = xd
                day -= SPREAD / S
                n_taken += 1
                free -= 1
        port[k] = day
        exposure[k] = len(active) / S
    eq = np.prod(1 + port) - 1
    sh = float(np.mean(port) / np.std(port) * np.sqrt(BARS_YR)) if np.std(port) > 0 else 0.0
    return {"total": eq, "sharpe": sh, "exposure": float(exposure.mean()),
            "n_taken": n_taken, "port": port}


def main() -> None:
    names = load_ftse.universe()
    say(f"Loading {len(names)} names; precomputing channels + trades...")
    retmap, entries_by_date = {}, defaultdict(list)
    closes = {}
    holds = []
    for t in names:
        try:
            df = load_ftse.load(t)
        except Exception:
            continue
        arr = precompute(df, L)
        if arr is None:
            continue
        r = df["close"].pct_change()
        retmap[t] = {d: float(v) for d, v in r.items() if np.isfinite(v)}
        closes[t] = df["close"]
        for ed, xd, grad in trades_for(df, arr):
            entries_by_date[ed].append((grad, t, xd))
            holds.append((xd - ed).days)
    say(f"{len(retmap)} names, {sum(len(v) for v in entries_by_date.values())} candidate entries, "
        f"median hold {np.median(holds):.0f} calendar days.")

    dates = pd.DatetimeIndex(sorted(set().union(*[set(closes[t].index) for t in closes])))

    # equal-weight buy-and-hold of the universe (always fully invested)
    ew = pd.DataFrame(closes).reindex(dates).pct_change().mean(axis=1).fillna(0.0).to_numpy()
    ew_total = np.prod(1 + ew) - 1
    ew_sh = float(np.mean(ew) / np.std(ew) * np.sqrt(BARS_YR))
    yrs = len(dates) / BARS_YR
    ew_cagr = (1 + ew_total) ** (1 / yrs) - 1

    say("\n" + "#" * 78)
    say(f"# CAPITAL-ROTATION PORTFOLIO vs equal-weight buy-and-hold  ({yrs:.0f}y)")
    say("#" * 78)
    say(f"  equal-weight B&H: total {100*ew_total:+.0f}%  CAGR {100*ew_cagr:+.2f}%  Sharpe {ew_sh:+.2f}")
    say(f"\n{'slots':>5} {'total%':>10} {'CAGR%':>8} {'Sharpe':>7} {'exposure':>9} {'trades':>7}  vs B&H")
    for S in SLOTS:
        res = run_portfolio(entries_by_date, retmap, dates, S)
        cagr = (1 + res["total"]) ** (1 / yrs) - 1
        beat = "WINS" if res["total"] > ew_total else "loses"
        say(f"{S:>5} {100*res['total']:>10.0f} {100*cagr:>8.2f} {res['sharpe']:>7.2f} "
            f"{100*res['exposure']:>8.0f}% {res['n_taken']:>7}  {beat} (B&H {100*ew_total:+.0f}%)")

    # the diagnostic Lance's argument rests on: return per year WHILE deployed
    say("\n" + "#" * 78)
    say("# Diagnostic: return per year WHILE DEPLOYED vs market drift")
    say("#" * 78)
    res1 = run_portfolio(entries_by_date, retmap, dates, 1)   # sequential, ~always in one name
    p = res1["port"]
    deployed = p[p != 0]
    if len(deployed):
        rate_deployed = (1 + np.sum(deployed)) ** (1 / (len(deployed) / BARS_YR)) - 1 \
            if (1 + np.sum(deployed)) > 0 else float("nan")
        say(f"  S=1 book: exposure {100*res1['exposure']:.0f}%, "
            f"~{100*np.mean(deployed)*BARS_YR:+.1f}%/yr annualised over deployed days")
    say(f"  market drift (equal-weight CAGR): {100*ew_cagr:+.2f}%/yr")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
