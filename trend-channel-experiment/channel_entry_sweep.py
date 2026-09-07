"""Sweep the channel ENTRY definition to attack the late-entry residual:
  (1) window length L (shorter -> catch a winner's launch earlier)
  (3) minimum channel width (relative)
  (4) minimum channel gradient (steeper -> stronger trend)
All with the best exit (ride: hold until price breaks the lower band by 0.5 of the
channel height) and rotation across S slots, compared to equal-weight buy-and-hold.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u channel_entry_sweep.py
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import load_ftse
from channel import fit_channel
from channel_portfolio import run_portfolio
from channel_ratchet import ENTRY_FRAC, FIN, K, SPREAD

HERE = Path(__file__).resolve().parent
BARS_YR = 252
R2_MIN = 0.80
BREAK_FRAC = 0.5
S = 5
WINDOWS = [100, 150, 250]
G_MINS = [0.10, 0.20, 0.30]
W_MINS = [0.0, 0.15, 0.25]
OUT = HERE / "channel_entry_sweep_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def precompute_rich(df: pd.DataFrame, Lw: int):
    close = df["close"].to_numpy()
    n = len(df)
    if n < Lw + 30:
        return None
    logc = np.log(close)
    up = np.full(n, np.nan); lo = np.full(n, np.nan)
    ann = np.full(n, np.nan); relw = np.full(n, np.nan); r2 = np.full(n, np.nan)
    for i in range(Lw, n):
        ch = fit_channel(logc[i - Lw + 1:i + 1])
        x = Lw - 1
        u, l, m = ch.upper(x, K), ch.lower(x, K), ch.mid(x)
        up[i], lo[i] = u, l
        ann[i] = ch.annual_gradient(BARS_YR)
        relw[i] = (u - l) / m if m > 0 else 0.0
        r2[i] = ch.r2
    return up, lo, ann, relw, r2


def ride_trades(df, arr, Lw, g_min, w_min):
    up, lo, ann, relw, r2 = arr
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
            if a >= g_min and r2[i] >= R2_MIN and relw[i] >= w_min and w > 0 and c <= l + ENTRY_FRAC * w:
                pos, e, grad = True, i + 1, a
        else:
            if c < l - BREAK_FRAC * w:
                out.append((dates[e], dates[i + 1], grad)); pos = False
    if pos:
        out.append((dates[e], dates[n - 1], grad))
    return out


def main() -> None:
    names = load_ftse.universe()
    say(f"Loading {len(names)} names...")
    dfs, closes = {}, {}
    for t in names:
        try:
            dfs[t] = load_ftse.load(t)
            closes[t] = dfs[t]["close"]
        except Exception:
            pass
    dates = pd.DatetimeIndex(sorted(set().union(*[set(closes[t].index) for t in closes])))
    retmap = {t: {d: float(v) for d, v in closes[t].pct_change().items() if np.isfinite(v)} for t in closes}
    ew = pd.DataFrame(closes).reindex(dates).pct_change().mean(axis=1).fillna(0.0).to_numpy()
    ew_total = np.prod(1 + ew) - 1
    yrs = len(dates) / BARS_YR
    ew_cagr = (1 + ew_total) ** (1 / yrs) - 1
    ew_sh = float(np.mean(ew) / np.std(ew) * np.sqrt(BARS_YR))
    say(f"{len(dfs)} names. equal-weight B&H: total {100*ew_total:+.0f}%  "
        f"CAGR {100*ew_cagr:+.2f}%  Sharpe {ew_sh:+.2f}\n")

    say("#" * 92)
    say(f"# ENTRY SWEEP (ride exit break=0.5, {S} slots) vs B&H CAGR {100*ew_cagr:.2f}%")
    say("#" * 92)
    say(f"{'L':>4} {'g_min':>6} {'w_min':>6} {'trades':>7} {'expo':>6} {'total%':>9} "
        f"{'CAGR%':>7} {'Sharpe':>7}  vs B&H")
    best = None
    for Lw in WINDOWS:
        arrs = {}
        for t, df in dfs.items():
            a = precompute_rich(df, Lw)
            if a is not None:
                arrs[t] = a
        for g_min in G_MINS:
            for w_min in W_MINS:
                ebd = defaultdict(list)
                for t, a in arrs.items():
                    for ed, xd, grad in ride_trades(dfs[t], a, Lw, g_min, w_min):
                        ebd[ed].append((grad, t, xd))
                ntr = sum(len(v) for v in ebd.values())
                if ntr < 30:
                    say(f"{Lw:>4} {g_min:>6.2f} {w_min:>6.2f} {ntr:>7}  (too few)"); continue
                res = run_portfolio(ebd, retmap, dates, S)
                cagr = (1 + res["total"]) ** (1 / yrs) - 1
                beat = "WINS" if res["total"] > ew_total else "loses"
                flag = "  <==" if (best is None or res["total"] > best[0]) else ""
                say(f"{Lw:>4} {g_min:>6.2f} {w_min:>6.2f} {ntr:>7} {100*res['exposure']:>5.0f}% "
                    f"{100*res['total']:>9.0f} {100*cagr:>7.2f} {res['sharpe']:>7.2f}  {beat}{flag}")
                if best is None or res["total"] > best[0]:
                    best = (res["total"], Lw, g_min, w_min, cagr, res["sharpe"])
    say(f"\nBest: L={best[1]}, g_min={best[2]}, w_min={best[3]} -> total {100*best[0]:+.0f}% "
        f"(CAGR {100*best[4]:.2f}%, Sharpe {best[5]:.2f}) vs B&H total {100*ew_total:+.0f}% (CAGR {100*ew_cagr:.2f}%)")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
