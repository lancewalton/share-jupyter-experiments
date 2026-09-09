"""Phase 4 - low-for-long / high-for-short price sourcing (second modification).

Enter on a MACD cross computed from LOW prices (a stricter uptrend confirmation) and
exit on a MACD cross from HIGH prices (the 'short signal' closing the long). Tests all
four entry/exit sourcings to isolate which leg, if any, helps, under the correct
concentrated portfolio, net of costs. Prior (Phase 2 null, p=1.000): the crossover is
anti-predictive, so reshaping it should not create a net edge.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u run_phase4_hilo.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macd.data import load_ohlc, top_n_mask
from macd.indicator import macd_cross_state, long_state, hysteresis_frame
from macd.backtest import portfolio, extract_trades
from macd.metrics import cagr, sharpe, hit_rate, mean_payoff

HERE = Path(__file__).resolve().parent
COST = 10 / 1e4
MIN_OBS = 750
UNIVERSE_N = 350
LIQ_WINDOW = 252
OUT = HERE / "phase4_hilo_results.txt"
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
    say("Loading OHLC panel and fixing the investable universe...")
    close, high, low, returns, turnover = load_ohlc(min_obs=MIN_OBS)
    liq = turnover.rolling(LIQ_WINDOW, min_periods=LIQ_WINDOW // 2).mean().shift(1)
    eligible = top_n_mask(liq, UNIVERSE_N)
    ever = eligible.any()
    close, high, low = close.loc[:, ever], high.loc[:, ever], low.loc[:, ever]
    returns, eligible = returns.loc[:, ever], eligible.loc[:, ever]
    returns_masked = returns.where(eligible)
    bh = returns_masked.mean(axis=1)
    say(f"universe: {int(ever.sum())} ever-eligible names; B&H CAGR {100*cagr(bh):+.2f}%, "
        f"Sharpe {sharpe(bh):.2f}\n")

    cc = macd_cross_state(close)
    ch = macd_cross_state(high)
    cl = macd_cross_state(low)

    variants = {
        "close/close (baseline)": (cc, cc),
        "LOW/HIGH (modification)": (cl, ch),
        "low/close (entry only)": (cl, cc),
        "close/high (exit only)": (cc, ch),
    }

    # integration gate: close/close hysteresis must reproduce the plain baseline
    base_pos = hysteresis_frame(cc, cc).shift(1, fill_value=False)
    assert base_pos.equals(long_state(close)), "close/close must equal long_state"

    say("#" * 88)
    say("# LOW-FOR-LONG / HIGH-FOR-SHORT (12/26/9, concentrated, net 10bps)")
    say("#" * 88)
    say(f"{'entry/exit':<26} {'excessCAGR':>11} {'Sharpe':>7} {'hold-days':>11} "
        f"{'hit-rate':>9} {'meanPayoff':>11}")

    for name, (entry, exit_) in variants.items():
        pos = hysteresis_frame(entry, exit_).shift(1, fill_value=False)
        strat = portfolio(pos, returns_masked, cost=COST, concentrate=True)
        exc = cagr(strat["strat"]) - cagr(bh)
        tr = trades_of(pos, returns, eligible)
        say(f"{name:<26} {100*exc:>+10.2f}% {sharpe(strat['strat']):>+7.2f} "
            f"{strat['hold_days']:>11,} {100*hit_rate(tr):>8.1f}% {100*mean_payoff(tr):>+10.3f}%")

    say("\nRead: a helpful sourcing would push excess CAGR toward 0 vs the baseline.")
    OUT.write_text("\n".join(_lines) + "\n")
    say(f"saved -> {OUT.name}")


if __name__ == "__main__":
    main()
