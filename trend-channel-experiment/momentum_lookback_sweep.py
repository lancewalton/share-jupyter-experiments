"""Is 12-1 optimal? Sweep formation length (lookback) x skip length (pullback).

Momentum signal from t-L to t-S:  signal = adj_close[t-S] / adj_close[t-L] - 1  (needs L > S).
Long-only top-quintile tilt, survivorship-free top-350, net tiered costs (same engine). Report
CAGR / Sharpe / maxDD for each (L, S), plus pre/post-2013 CAGR to check the winner isn't an
in-sample artefact. Judge by a robust PLATEAU, not the single best cell.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u momentum_lookback_sweep.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from momentum_tradeability import build, run, cagr, sharpe, maxdd

HERE = Path(__file__).resolve().parent
LOOKBACKS = [3, 6, 9, 12, 15, 18, 24]
SKIPS = [0, 1, 2, 3]
OUT = HERE / "momentum_lookback_sweep_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True); _lines.append(s)


def main() -> None:
    say("Building panel...")
    M, mret, liq, _sig = build()

    def sig(L, S):
        return M.shift(S) / M.shift(L) - 1

    base, _ = run(M, mret, liq, sig(12, 1), 1, "tier", 0.0)
    say(f"eligible-universe EW B&H: CAGR {100*cagr(base['uni']):+.2f}%  Sharpe {sharpe(base['uni']):.2f}\n")

    say("#" * 78)
    say("# FORMATION x SKIP -- CAGR% (Sharpe) [maxDD%], momentum-only, net tiered costs")
    say("#" * 78)
    say(f"{'look\\skip':>9} " + " ".join(f"{'S='+str(s):>16}" for s in SKIPS))
    grid = {}
    for L in LOOKBACKS:
        cells = []
        for S in SKIPS:
            if S >= L:
                cells.append(f"{'—':>16}"); continue
            P, _ = run(M, mret, liq, sig(L, S), 1, "tier", 0.0)
            r = P["lo"]
            grid[(L, S)] = r
            cells.append(f"{100*cagr(r):>5.1f}({sharpe(r):.2f})[{100*maxdd(r):>3.0f}]")
        say(f"{'L='+str(L):>9} " + " ".join(cells))

    # incumbent vs best-by-Sharpe, with OOS split
    say("\n--- robustness: pre/post-2013 CAGR (guards against in-sample overfit) ---")
    ranked = sorted(grid.items(), key=lambda kv: sharpe(kv[1]), reverse=True)
    show = [(12, 1)] + [k for k, _ in ranked[:4] if k != (12, 1)]
    say(f"{'(L,S)':>8} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7} {'pre-13':>7} {'post-13':>8}")
    for k in show:
        r = grid[k]
        pre, post = r[r.index < "2013-01-01"], r[r.index >= "2013-01-01"]
        tag = "  <- incumbent" if k == (12, 1) else ""
        say(f"{str(k):>8} {100*cagr(r):>+6.2f}% {sharpe(r):>7.2f} {100*maxdd(r):>6.0f}% "
            f"{100*cagr(pre):>+6.2f}% {100*cagr(post):>+7.2f}%{tag}")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
