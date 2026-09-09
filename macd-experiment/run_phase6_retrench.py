"""Phase 6 - ignore-first-signal / retrenchment re-entry (fourth modification).

Skip the first MACD up-cross of an episode and enter on the second one within a
window (the README's "wait for a retrenchment, enter on the next signal"). Exit is the
standard cross-down. Swept over the retrenchment window, concentrated book, net of
costs. Prior (Phase 2 null): the crossover is anti-predictive, so waiting for a second
one should not manufacture a net edge.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u run_phase6_retrench.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macd.data import load_panel, top_n_mask
from macd.indicator import macd_cross_state, long_state, hysteresis_frame, retrench_entry
from macd.backtest import portfolio, extract_trades
from macd.metrics import cagr, sharpe, hit_rate, mean_payoff

HERE = Path(__file__).resolve().parent
COST = 10 / 1e4
MIN_OBS = 750
UNIVERSE_N = 350
LIQ_WINDOW = 252
WINDOWS = [5, 10, 20, 40]
OUT = HERE / "phase6_retrench_results.txt"
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


def retrench_entry_frame(cross: pd.DataFrame, window: int) -> pd.DataFrame:
    return pd.DataFrame(
        {c: retrench_entry(cross[c], window=window) for c in cross.columns},
        index=cross.index,
    )


def report(tag, pos, returns_masked, returns, eligible, bh) -> None:
    strat = portfolio(pos, returns_masked, cost=COST, concentrate=True)
    exc = cagr(strat["strat"]) - cagr(bh)
    tr = trades_of(pos, returns, eligible)
    say(f"{tag:>18} {100*exc:>+10.2f}% {sharpe(strat['strat']):>+7.2f} "
        f"{strat['hold_days']:>11,} {100*hit_rate(tr):>8.1f}% {100*mean_payoff(tr):>+10.3f}%")


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

    cross = macd_cross_state(prices, fast=12, slow=26)

    say("#" * 84)
    say("# RETRENCHMENT RE-ENTRY - skip first up-cross, enter 2nd within window (net 10bps)")
    say("#" * 84)
    say(f"{'variant':>18} {'excessCAGR':>11} {'Sharpe':>7} {'hold-days':>11} "
        f"{'hit-rate':>9} {'meanPayoff':>11}")

    report("baseline *", long_state(prices), returns_masked, returns, eligible, bh)
    for w in WINDOWS:
        entry = retrench_entry_frame(cross, w)
        pos = hysteresis_frame(entry, cross).shift(1, fill_value=False)
        report(f"retrench w={w}", pos, returns_masked, returns, eligible, bh)

    say("\n* baseline enters on every up-cross (Phase 1).")
    OUT.write_text("\n".join(_lines) + "\n")
    say(f"saved -> {OUT.name}")


if __name__ == "__main__":
    main()
