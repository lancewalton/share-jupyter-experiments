"""Decompose the channel strategy's shortfall vs buy-and-hold into:
  (1) SELECTION  - do the names that form qualifying channels grow slower than the
      universe average? (compare buy-and-hold CAGR of formers vs non-formers.)
  (2) WINNER-CAPPING - among the names it does trade, how much of each name's
      buy-and-hold does the strategy capture, and is capture worse for the big
      winners? Plus: how far does a name keep rising AFTER the strategy sells at the
      channel top (return left on the table)?

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u channel_decompose.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import load_ftse
from channel_portfolio import trades_for
from channel_ratchet import FIN, L, SPREAD, precompute

HERE = Path(__file__).resolve().parent
BARS_YR = 252
POST = 60          # trading days after a top-exit to measure continuation
OUT = HERE / "channel_decompose_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def ew_cagr(names: list[str], closes: dict) -> tuple[float, float]:
    if not names:
        return float("nan"), float("nan")
    sub = pd.DataFrame({t: closes[t] for t in names}).sort_index()
    r = sub.pct_change().mean(axis=1).fillna(0.0).to_numpy()
    total = float(np.prod(1 + r) - 1)
    yrs = len(r) / BARS_YR
    return (1 + total) ** (1 / yrs) - 1, total


def main() -> None:
    names = load_ftse.universe()
    say(f"Loading {len(names)} names; precomputing channels...")
    closes, per = {}, {}
    for t in names:
        try:
            df = load_ftse.load(t)
        except Exception:
            continue
        arr = precompute(df, L)
        if arr is None:
            continue
        closes[t] = df["close"]
        close = df["close"].to_numpy()
        dates = df.index
        yrs = (dates[-1] - dates[0]).days / 365.25
        bh_total = close[-1] / close[0] - 1.0
        pos = {d: i for i, d in enumerate(dates)}
        trades = trades_for(df, arr)
        mult = 1.0
        post = []
        for ed, xd, _ in trades:
            ei, xi = pos[ed], pos[xd]
            gross = close[xi] / close[ei] - 1.0
            days = max(1, (xd - ed).days)
            mult *= (1 + gross - 2 * SPREAD - FIN * days / 365)
            if xi + POST < len(close):
                post.append(close[xi + POST] / close[xi] - 1.0)
        strat_total = mult - 1.0
        per[t] = {
            "bh_total": bh_total, "bh_cagr": (1 + bh_total) ** (1 / yrs) - 1,
            "strat_total": strat_total, "n_trades": len(trades),
            "post": post,
        }
    say(f"{len(per)} names processed.\n")

    formers = [t for t, p in per.items() if p["n_trades"] > 0]
    nonformers = [t for t, p in per.items() if p["n_trades"] == 0]

    say("#" * 74)
    say("# (1) SELECTION — do channel-forming names grow slower than the universe?")
    say("#" * 74)
    for label, group in [("ALL names", list(per)), ("channel FORMERS", formers),
                         ("NON-formers", nonformers)]:
        if not group:
            say(f"  {label:16}: none"); continue
        cagrs = np.array([per[t]["bh_cagr"] for t in group])
        ewc, ewt = ew_cagr(group, closes)
        say(f"  {label:16} n={len(group):>3}  median single-name B&H CAGR {100*np.median(cagrs):+.2f}%  "
            f"mean {100*np.mean(cagrs):+.2f}%  |  equal-weight-hold CAGR {100*ewc:+.2f}%")

    # do the biggest winners form channels?
    top = sorted(per, key=lambda t: per[t]["bh_total"], reverse=True)[:10]
    say("\n  top-10 B&H winners — do they form tradeable channels?")
    say(f"    {'name':6} {'B&H%':>9} {'trades':>7} {'strat%':>9} {'capture':>8}")
    for t in top:
        p = per[t]
        cap = p["strat_total"] / p["bh_total"] if p["bh_total"] > 0 else float("nan")
        say(f"    {t:6} {100*p['bh_total']:>9.0f} {p['n_trades']:>7} {100*p['strat_total']:>9.0f} "
            f"{cap:>8.2f}")

    say("\n" + "#" * 74)
    say("# (2) WINNER-CAPPING — capture of B&H by the size of the winner (formers)")
    say("#" * 74)
    fr = [t for t in formers if per[t]["bh_total"] > 0]
    bt = np.array([per[t]["bh_total"] for t in fr])
    q = np.quantile(bt, [0, 0.25, 0.5, 0.75, 1.0])
    say(f"  {'B&H quartile':>16} {'n':>4} {'med B&H%':>10} {'med strat%':>11} {'med capture':>12}")
    labels = ["Q1 (smallest)", "Q2", "Q3", "Q4 (biggest)"]
    for i, lab in enumerate(labels):
        lo, hi = q[i], q[i + 1]
        sel = [t for t in fr if (lo <= per[t]["bh_total"] <= hi if i == 3 else lo <= per[t]["bh_total"] < hi)]
        if not sel:
            continue
        bh = np.array([per[t]["bh_total"] for t in sel])
        stt = np.array([per[t]["strat_total"] for t in sel])
        cap = np.array([per[t]["strat_total"] / per[t]["bh_total"] for t in sel])
        say(f"  {lab:>16} {len(sel):>4} {100*np.median(bh):>10.0f} {100*np.median(stt):>11.0f} "
            f"{np.median(cap):>12.2f}")

    allpost = np.array([x for t in formers for x in per[t]["post"]])
    say(f"\n  continuation after a channel-top exit ({POST} trading days later):")
    say(f"    median {100*np.median(allpost):+.2f}%  mean {100*np.mean(allpost):+.2f}%  "
        f"share still rising {100*np.mean(allpost>0):.0f}%  (n={len(allpost)})")
    say("  (positive => the strategy sold and the name kept climbing — capped upside)")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
