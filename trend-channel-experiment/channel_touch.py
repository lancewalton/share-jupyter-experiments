"""Idea #2: define/validate the channel by NUMBER OF TOUCHES of the support and
resistance bands, rather than by fit quality (R2). A channel qualifies when, over
the window, the high has touched the upper band >= N times AND the low has touched
the lower band >= N times. Everything else is the best pipeline (ride exit,
rotation) plus the decisive degeneration check: strategy vs buy-and-hold of the
same names it trades.

(A full variable-window "grow until N touches" is O(n^2)/bar; we use a touch gate
at fixed windows, which still replaces the fit/day criterion with a touch count.)

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u channel_touch.py
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import load_ftse
from channel import fit_channel
from channel_portfolio import run_portfolio
from channel_ratchet import ENTRY_FRAC, K

HERE = Path(__file__).resolve().parent
BARS_YR = 252
G_MIN = 0.10
BREAK_FRAC = 0.5
TOUCH_FRAC = 0.25      # a "touch" = within 0.25*(K*sigma) of the band, in log space
S = 5
WINDOWS = [100, 250]
N_TOUCHES = [2, 3, 4]
OUT = HERE / "channel_touch_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def _edges(mask: np.ndarray) -> int:
    if len(mask) == 0:
        return 0
    return int(np.sum(mask & ~np.concatenate(([False], mask[:-1]))))


def precompute_touch(df: pd.DataFrame, Lw: int):
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    if n < Lw + 30:
        return None
    logc, logh, logl = np.log(close), np.log(high), np.log(low)
    up = np.full(n, np.nan); lo = np.full(n, np.nan); ann = np.full(n, np.nan)
    tup = np.zeros(n, int); tlo = np.zeros(n, int)
    for i in range(Lw, n):
        w0 = i - Lw + 1
        ch = fit_channel(logc[w0:i + 1])
        x = Lw - 1
        up[i], lo[i], ann[i] = ch.upper(x, K), ch.lower(x, K), ch.annual_gradient(BARS_YR)
        xs = np.arange(Lw)
        upper = ch.intercept + ch.gradient * xs + K * ch.sigma
        lower = ch.intercept + ch.gradient * xs - K * ch.sigma
        tol = TOUCH_FRAC * K * ch.sigma
        tup[i] = _edges(logh[w0:i + 1] >= upper - tol)
        tlo[i] = _edges(logl[w0:i + 1] <= lower + tol)
    return up, lo, ann, tup, tlo


def ride_trades(df, arr, Lw, n_touch):
    up, lo, ann, tup, tlo = arr
    close = df["close"].to_numpy(); dates = df.index; n = len(df)
    out = []
    pos = False; e = grad = None
    for i in range(Lw, n - 1):
        u, l, a = up[i], lo[i], ann[i]
        if not np.isfinite(u):
            continue
        w = u - l
        c = close[i]
        if not pos:
            if a >= G_MIN and tup[i] >= n_touch and tlo[i] >= n_touch and w > 0 and c <= l + ENTRY_FRAC * w:
                pos, e, grad = True, i + 1, a
        else:
            if c < l - BREAK_FRAC * w:
                out.append((dates[e], dates[i + 1], grad)); pos = False
    if pos:
        out.append((dates[e], dates[n - 1], grad))
    return out


def cagr_of(names, closes, dates) -> float:
    if not names:
        return float("nan")
    sub = pd.DataFrame({t: closes[t] for t in names}).reindex(dates)
    r = sub.pct_change().mean(axis=1).fillna(0.0).to_numpy()
    total = np.prod(1 + r) - 1
    return (1 + total) ** (1 / (len(dates) / BARS_YR)) - 1


def main() -> None:
    names = load_ftse.universe()
    say(f"Loading {len(names)} names...")
    dfs, closes = {}, {}
    for t in names:
        try:
            dfs[t] = load_ftse.load(t); closes[t] = dfs[t]["close"]
        except Exception:
            pass
    dates = pd.DatetimeIndex(sorted(set().union(*[set(closes[t].index) for t in closes])))
    retmap = {t: {d: float(v) for d, v in closes[t].pct_change().items() if np.isfinite(v)} for t in closes}
    univ = cagr_of(list(closes), closes, dates)
    say(f"{len(dfs)} names. Universe equal-weight B&H CAGR {100*univ:+.2f}%\n")

    say("#" * 94)
    say("# TOUCH-DEFINED channel (>=N touches each band), ride exit, degeneration check")
    say("#" * 94)
    say(f"{'L':>4} {'Ntouch':>7} {'names':>6} {'trades':>7} {'expo':>6} {'stratCAGR':>10} "
        f"{'tradedBH':>9} {'edge':>7} {'Sharpe':>7}  vsUniv")
    for Lw in WINDOWS:
        arrs = {}
        for t, df in dfs.items():
            a = precompute_touch(df, Lw)
            if a is not None:
                arrs[t] = a
        for nt in N_TOUCHES:
            ebd = defaultdict(list); traded = set()
            for t, a in arrs.items():
                for ed, xd, grad in ride_trades(dfs[t], a, Lw, nt):
                    ebd[ed].append((grad, t, xd)); traded.add(t)
            ntr = sum(len(v) for v in ebd.values())
            if ntr < 20:
                say(f"{Lw:>4} {nt:>7} {len(traded):>6} {ntr:>7}  (too few)"); continue
            res = run_portfolio(ebd, retmap, dates, S)
            scagr = (1 + res["total"]) ** (1 / (len(dates) / BARS_YR)) - 1
            tbh = cagr_of(sorted(traded), closes, dates)
            beat = "WINS" if scagr > univ else "loses"
            say(f"{Lw:>4} {nt:>7} {len(traded):>6} {ntr:>7} {100*res['exposure']:>5.0f}% "
                f"{100*scagr:>9.2f}% {100*tbh:>8.2f}% {100*(scagr-tbh):>+6.2f}% {res['sharpe']:>7.2f}  {beat}")

    say("\n  edge = strategy CAGR - B&H CAGR of the same traded names (the decisive test).")
    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
