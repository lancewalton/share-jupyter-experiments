"""Stress the channel-SELECTION edge (hold names currently in a qualifying rising
channel, L=250) before believing it:
  - out-of-sample split at 2013 (is it just the pre-2013 regime / decaying momentum?)
  - concentration: drop the top-1 and top-3 contributing names -- does the edge survive
    or was it a couple of monster winners?

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u channel_select_stress.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import load_ftse
from channel_entry_sweep import precompute_rich

HERE = Path(__file__).resolve().parent
BARS_YR = 252
L = 250
G_MIN, R2_MIN = 0.10, 0.80
SPREAD = 10 / 1e4
W_MINS = [0.0, 0.15, 0.25]
SPLIT = pd.Timestamp("2013-01-01")
OUT = HERE / "channel_select_stress_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def cagr(daily: np.ndarray) -> float:
    if len(daily) == 0:
        return float("nan")
    total = np.prod(1 + daily) - 1
    return (1 + total) ** (1 / (len(daily) / BARS_YR)) - 1 if total > -1 else float("nan")


def sharpe(daily: np.ndarray) -> float:
    return float(np.mean(daily) / np.std(daily) * np.sqrt(BARS_YR)) if np.std(daily) > 0 else 0.0


def sel_daily(qual: pd.DataFrame, RETS: pd.DataFrame) -> np.ndarray:
    held = qual.shift(1).fillna(False)
    n = held.sum(axis=1)
    mret = RETS.where(held).mean(axis=1).fillna(0.0)
    changes = held.astype(int).diff().abs().sum(axis=1)
    cost = (SPREAD * changes / n.replace(0, np.nan)).fillna(0.0)
    return (mret - cost).to_numpy()


def main() -> None:
    names = load_ftse.universe()
    say(f"Loading {len(names)} names; precomputing L={L}...")
    closes, ANN, RELW, R2 = {}, {}, {}, {}
    for t in names:
        try:
            df = load_ftse.load(t)
        except Exception:
            continue
        a = precompute_rich(df, L)
        if a is None:
            continue
        closes[t] = df["close"]
        _, _, ann, relw, r2 = a
        ANN[t] = pd.Series(ann, index=df.index)
        RELW[t] = pd.Series(relw, index=df.index)
        R2[t] = pd.Series(r2, index=df.index)
    dates = pd.DatetimeIndex(sorted(set().union(*[set(closes[t].index) for t in closes])))
    RETS = pd.DataFrame(closes).reindex(dates).pct_change()
    ANN = pd.DataFrame(ANN).reindex(dates); RELW = pd.DataFrame(RELW).reindex(dates); R2 = pd.DataFrame(R2).reindex(dates)
    base = (ANN >= G_MIN) & (R2 >= R2_MIN)
    uni = RETS.mean(axis=1).fillna(0.0).to_numpy()
    pre = dates < SPLIT
    post = dates >= SPLIT
    say(f"{len(closes)} names. Universe B&H CAGR {100*cagr(uni):+.2f}% | pre {100*cagr(uni[pre]):+.2f}% "
        f"post {100*cagr(uni[post]):+.2f}%  Sharpe {sharpe(uni):.2f}\n")

    say("#" * 96)
    say("# SELECTION STRESS (L=250): OOS split @2013 and drop-top-contributors")
    say("#" * 96)
    say(f"{'w_min':>6} {'names':>6} {'CAGR':>7} {'Sharpe':>7} {'pre13':>7} {'post13':>7} "
        f"{'exTop1':>7} {'exTop3':>7}")
    for w in W_MINS:
        qual = (base & (RELW >= w)).fillna(False)
        d = sel_daily(qual, RETS)
        # per-name contribution to the (equal-weight, time-varying) book
        held = qual.shift(1).fillna(False)
        n = held.sum(axis=1).replace(0, np.nan)
        contrib = (RETS.where(held).div(n, axis=0)).sum(axis=0).sort_values(ascending=False)
        top1 = list(contrib.index[:1]); top3 = list(contrib.index[:3])
        d_ex1 = sel_daily(qual.drop(columns=top1), RETS.drop(columns=top1))
        d_ex3 = sel_daily(qual.drop(columns=top3), RETS.drop(columns=top3))
        n_ever = int((qual.any(axis=0)).sum())
        say(f"{w:>6.2f} {n_ever:>6} {100*cagr(d):>6.2f}% {sharpe(d):>7.2f} {100*cagr(d[pre]):>6.2f}% "
            f"{100*cagr(d[post]):>6.2f}% {100*cagr(d_ex1):>6.2f}% {100*cagr(d_ex3):>6.2f}%")
        say(f"        top contributors: {', '.join(contrib.index[:3])}")

    say("\n  pre13/post13 = selection net CAGR in each era (universe: "
        f"{100*cagr(uni[pre]):.1f}% / {100*cagr(uni[post]):.1f}%).")
    say("  exTop1/exTop3 = selection net CAGR after removing the 1/3 biggest-contributing names.")
    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
