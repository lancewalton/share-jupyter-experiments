"""Phase 3 - volatility-regime gate on standard MACD (Lance's first modification).

Falsification test: does gating MACD long entries to higher-volatility regimes
improve the *net* economics, or does it only move *hit-rate* while leaving net
payoff untouched (the programme's prior for direction filters)? Sweeps the gate
percentile at 12/26/9 and reports net excess CAGR against the hit-rate/payoff split.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u run_phase3_volgate.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macd.data import load_panel, top_n_mask
from macd.indicator import long_state, gated_long_state
from macd.backtest import portfolio, extract_trades
from macd.metrics import cagr, sharpe, hit_rate, mean_payoff

HERE = Path(__file__).resolve().parent
COST = 10 / 1e4
MIN_OBS = 750
UNIVERSE_N = 350
LIQ_WINDOW = 252
VOL_WINDOW = 20
Q_WINDOW = 252
QS = [0.0, 0.3, 0.5, 0.7]           # 0.0 = ungated baseline
OUT = HERE / "phase3_volgate_results.txt"
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

    bh_bps = 1e4 * float(bh.mean())    # what an average market day earns (gross)

    say("#" * 82)
    say("# VOL-REGIME GATE on 12/26/9 (net 10bps) - does it move net payoff or hit-rate?")
    say("#" * 82)
    say(f"average market day (B&H, gross): {bh_bps:+.3f} bps/day\n")
    say(f"{'gate q':>7} {'excessCAGR':>11} {'Sharpe':>7} {'hold-days':>10} "
        f"{'hit-rate':>9} {'meanPayoff':>11} {'bps/holdday':>12}")

    for q in QS:
        if q == 0.0:
            pos = long_state(prices, fast=12, slow=26, signal=9)
            label = "0.00*"
        else:
            pos = gated_long_state(prices, returns, fast=12, slow=26, signal=9,
                                   vol_window=VOL_WINDOW, vol_q=q, q_window=Q_WINDOW)
            label = f"{q:.2f}"
        strat = portfolio(pos, returns_masked, cost=COST, concentrate=True)
        exc = cagr(strat["strat"]) - cagr(bh)
        tr = trades_of(pos, returns, eligible)
        held = pos.astype(bool) & eligible
        gross_per_holdday = 1e4 * float((held * returns_masked.fillna(0.0)).to_numpy().sum()
                                        / strat["hold_days"])
        say(f"{label:>7} {100*exc:>+10.2f}% {sharpe(strat['strat']):>+7.2f} "
            f"{strat['hold_days']:>10,} {100*hit_rate(tr):>8.1f}% "
            f"{100*mean_payoff(tr):>+10.3f}% {gross_per_holdday:>+11.3f}")

    say("\n* q=0.00 is the ungated Phase 1 baseline.")
    say("Read: a gate that adds *timing skill* makes bps/hold-day exceed the average")
    say("market day; a gate that merely trades less leaves it at or below the market.")
    OUT.write_text("\n".join(_lines) + "\n")
    say(f"saved -> {OUT.name}")


if __name__ == "__main__":
    main()
