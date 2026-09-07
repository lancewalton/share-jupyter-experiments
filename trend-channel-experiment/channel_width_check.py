"""Is the min-width lever a real edge, or degeneration toward buy-and-hold?

As the required channel width rises, the ride stop (0.5x width below support) moves
far away, so positions are held for years on a shrinking set of volatile names. If
the strategy is merely becoming buy-and-hold of those names, its return should
converge to the equal-weight B&H of the *traded names* (not beat it). Test: sweep
w_min at L=100 and compare strategy CAGR to (a) B&H of the traded names, (b) B&H of
the whole universe.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u channel_width_check.py
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import load_ftse
from channel_entry_sweep import precompute_rich, ride_trades
from channel_portfolio import run_portfolio

HERE = Path(__file__).resolve().parent
BARS_YR = 252
L = 100
G_MIN = 0.10
S = 5
W_MINS = [0.10, 0.15, 0.25, 0.35, 0.45, 0.60]
OUT = HERE / "channel_width_check_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def cagr_of(names, closes, dates) -> float:
    if not names:
        return float("nan")
    sub = pd.DataFrame({t: closes[t] for t in names}).reindex(dates)
    r = sub.pct_change().mean(axis=1).fillna(0.0).to_numpy()
    total = np.prod(1 + r) - 1
    return (1 + total) ** (1 / (len(dates) / BARS_YR)) - 1


def main() -> None:
    names = load_ftse.universe()
    say(f"Loading {len(names)} names; precomputing L={L} channels...")
    dfs, closes, arrs = {}, {}, {}
    for t in names:
        try:
            df = load_ftse.load(t)
        except Exception:
            continue
        a = precompute_rich(df, L)
        if a is None:
            continue
        dfs[t] = df; closes[t] = df["close"]; arrs[t] = a
    dates = pd.DatetimeIndex(sorted(set().union(*[set(closes[t].index) for t in closes])))
    retmap = {t: {d: float(v) for d, v in closes[t].pct_change().items() if np.isfinite(v)} for t in closes}
    univ_cagr = cagr_of(list(closes), closes, dates)
    say(f"{len(dfs)} names. Universe equal-weight B&H CAGR {100*univ_cagr:+.2f}%\n")

    say("#" * 92)
    say("# DEGENERATION CHECK: strategy vs B&H of the SAME names it trades (L=100, ride 0.5)")
    say("#" * 92)
    say(f"{'w_min':>6} {'names':>6} {'trades':>7} {'medHold_d':>10} {'expo':>6} "
        f"{'stratCAGR':>10} {'tradedBH':>9} {'edge':>7} {'Sharpe':>7}")
    for w in W_MINS:
        ebd = defaultdict(list)
        traded = set()
        holds = []
        for t, a in arrs.items():
            trs = ride_trades(dfs[t], a, L, G_MIN, w)
            for ed, xd, grad in trs:
                ebd[ed].append((grad, t, xd)); traded.add(t); holds.append((xd - ed).days)
        ntr = sum(len(v) for v in ebd.values())
        if ntr < 20:
            say(f"{w:>6.2f} {len(traded):>6} {ntr:>7}  (too few)"); continue
        res = run_portfolio(ebd, retmap, dates, S)
        scagr = (1 + res["total"]) ** (1 / (len(dates) / BARS_YR)) - 1
        tbh = cagr_of(sorted(traded), closes, dates)
        say(f"{w:>6.2f} {len(traded):>6} {ntr:>7} {np.median(holds):>10.0f} {100*res['exposure']:>5.0f}% "
            f"{100*scagr:>9.2f}% {100*tbh:>8.2f}% {100*(scagr-tbh):>+6.2f}% {res['sharpe']:>7.2f}")

    say("\n  edge = strategy CAGR - buy&hold CAGR of the same traded names.")
    say("  If edge -> 0 (or negative) as width rises while names shrink and holds lengthen,")
    say("  the 'improvement' is degeneration into buy-and-hold of volatile names, not timing skill.")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
