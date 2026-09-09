"""Phase 5 - separate exit parameters (third modification).

Entry fixed at the standard 12/26 MACD cross; the *exit* uses its own MACD lengths —
a faster exit cuts losers quicker, a slower exit holds winners longer. Tested under
the concentrated book, net of costs. Prior (Phase 2 null): the crossover is
anti-predictive, so a different exit MACD should not manufacture a net edge.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u run_phase5_exitparams.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macd.data import load_panel, top_n_mask
from macd.indicator import macd_cross_state, long_state, hysteresis_frame
from macd.backtest import portfolio, extract_trades
from macd.metrics import cagr, sharpe, hit_rate, mean_payoff

HERE = Path(__file__).resolve().parent
COST = 10 / 1e4
MIN_OBS = 750
UNIVERSE_N = 350
LIQ_WINDOW = 252
ENTRY = (12, 26)
EXITS = [(12, 26), (6, 13), (5, 20), (19, 40), (24, 52)]   # first = baseline
OUT = HERE / "phase5_exitparams_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def trades_of(positions, returns, eligible) -> np.ndarray:
    pos_e = positions.astype(bool) & eligible
    out: list[float] = []
    for code in pos_e.columns:
        p = pos_e[code]
        if p.any():
            out.extend(extract_trades(p, returns[code].fillna(0.0)))
    return np.asarray(out, dtype=float)


def main() -> None:
    say("Loading panel and fixing the investable universe...")
    prices, returns, turnover = load_panel(min_obs=MIN_OBS)
    liq = turnover.rolling(LIQ_WINDOW, min_periods=LIQ_WINDOW // 2).mean().shift(1)
    eligible = top_n_mask(liq, UNIVERSE_N)
    ever = eligible.any()
    prices, returns, eligible = prices.loc[:, ever], returns.loc[:, ever], eligible.loc[:, ever]
    returns_masked = returns.where(eligible)
    bh = returns_masked.mean(axis=1)
    say(f"universe: {int(ever.sum())} ever-eligible names; B&H CAGR {100*cagr(bh):+.2f}%, "
        f"Sharpe {sharpe(bh):.2f}\n")

    entry_cross = macd_cross_state(prices, fast=ENTRY[0], slow=ENTRY[1])

    say("#" * 84)
    say(f"# SEPARATE EXIT PARAMS - entry fixed {ENTRY[0]}/{ENTRY[1]}, concentrated, net 10bps")
    say("#" * 84)
    say(f"{'exit fast/slow':>16} {'excessCAGR':>11} {'Sharpe':>7} {'hold-days':>11} "
        f"{'hit-rate':>9} {'meanPayoff':>11}")

    for fx, sx in EXITS:
        exit_cross = macd_cross_state(prices, fast=fx, slow=sx)
        pos = hysteresis_frame(entry_cross, exit_cross).shift(1, fill_value=False)
        tag = f"{fx}/{sx}" + (" *" if (fx, sx) == ENTRY else "")
        strat = portfolio(pos, returns_masked, cost=COST, concentrate=True)
        exc = cagr(strat["strat"]) - cagr(bh)
        tr = trades_of(pos, returns, eligible)
        say(f"{tag:>16} {100*exc:>+10.2f}% {sharpe(strat['strat']):>+7.2f} "
            f"{strat['hold_days']:>11,} {100*hit_rate(tr):>8.1f}% {100*mean_payoff(tr):>+10.3f}%")

    say("\n* baseline (exit params == entry params) reproduces Phase 1.")
    OUT.write_text("\n".join(_lines) + "\n")
    say(f"saved -> {OUT.name}")


if __name__ == "__main__":
    main()
