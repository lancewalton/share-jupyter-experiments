"""Phase 1 - standard 12/26/9 MACD, long-only, portfolio level vs equal-weight B&H.

The decisive cheap test (see README): does textbook MACD timing beat always-invested
buy-and-hold on the survivorship-free EODHD UK universe, and does any edge live in
*payoff* or only in *hit-rate*? Reports gross and net (10bps turnover), full-sample
and a pre/post-2013 out-of-sample split.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u run_phase1_baseline.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from macd.data import load_panel, top_n_mask
from macd.indicator import long_state
from macd.backtest import portfolio, extract_trades
from macd.metrics import sharpe, cagr, max_drawdown, hit_rate, mean_payoff

HERE = Path(__file__).resolve().parent
FAST, SLOW, SIGNAL = 12, 26, 9
COST = 10 / 1e4
MIN_OBS = 750
UNIVERSE_N = 350
LIQ_WINDOW = 252
SPLIT = pd.Timestamp("2013-01-01")
OUT = HERE / "phase1_baseline_results.txt"
PNG = HERE / "phase1_baseline.png"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def summarise(label: str, r: pd.Series) -> None:
    say(f"  {label:<26} CAGR {100*cagr(r):+6.2f}%  Sharpe {sharpe(r):+5.2f}  "
        f"maxDD {100*max_drawdown(r):+6.1f}%")


def all_trades(positions: pd.DataFrame, returns: pd.DataFrame, eligible: pd.DataFrame) -> np.ndarray:
    """Per-name gross trade P&Ls, counting only days the name is in the universe."""
    pos_e = positions.astype(bool) & eligible
    trades: list[float] = []
    for code in pos_e.columns:
        pos = pos_e[code]
        if pos.any():
            trades.extend(extract_trades(pos, returns[code].fillna(0.0)))
    return np.asarray(trades, dtype=float)


def main() -> None:
    say("Loading and cleaning EODHD UK panel...")
    prices, returns, turnover = load_panel(min_obs=MIN_OBS)

    # point-in-time investable universe: top-N by trailing-1y average turnover,
    # lagged one day so eligibility on day t uses only prior data
    liq = turnover.rolling(LIQ_WINDOW, min_periods=LIQ_WINDOW // 2).mean().shift(1)
    eligible = top_n_mask(liq, UNIVERSE_N)
    returns = returns.where(eligible)

    positions = long_state(prices, fast=FAST, slow=SLOW, signal=SIGNAL)
    say(f"panel: {prices.shape[1]} names x {prices.shape[0]} days "
        f"({prices.index.min().date()} -> {prices.index.max().date()}); "
        f"investable universe top-{UNIVERSE_N} by turnover\n")

    gross = portfolio(positions, returns, cost=0.0, concentrate=True)
    net = portfolio(positions, returns, cost=COST, concentrate=True)
    invested_days = gross["hold_days"]
    gross_pnl = (positions.astype(bool) & returns.notna()).astype(float) * returns.fillna(0.0)
    per_hold_day = float(gross_pnl.to_numpy().sum() / invested_days)

    say("#" * 78)
    say(f"# STANDARD MACD {FAST}/{SLOW}/{SIGNAL}, LONG-ONLY, EQUAL-WEIGHT PORTFOLIO")
    say("#" * 78)
    say(f"invested name-days: {invested_days:,}  | mean return per hold-day: "
        f"{1e4*per_hold_day:+.3f} bps (gross)\n")

    say("FULL SAMPLE")
    summarise("buy-and-hold (EW)", gross["bnh"])
    summarise("MACD timing (gross)", gross["strat"])
    summarise("MACD timing (net 10bps)", net["strat"])
    say("")

    for name, lo, hi in [("pre-2013", None, SPLIT), ("2013-on (OOS)", SPLIT, None)]:
        m = pd.Series(True, index=gross["bnh"].index)
        if lo is not None:
            m &= gross["bnh"].index >= lo
        if hi is not None:
            m &= gross["bnh"].index < hi
        say(f"{name}")
        summarise("buy-and-hold (EW)", gross["bnh"][m])
        summarise("MACD timing (net)", net["strat"][m])
        say("")

    trades = all_trades(positions, returns, eligible)
    say("TRADE-LEVEL DECOMPOSITION (gross, per-name spells)")
    say(f"  trades: {len(trades):,}  hit-rate: {100*hit_rate(trades):.1f}%  "
        f"mean payoff: {100*mean_payoff(trades):+.3f}%  "
        f"median: {100*np.median(trades):+.3f}%")
    say("")

    verdict = ("BEATS" if cagr(net["strat"]) > cagr(gross["bnh"]) else "LOSES TO")
    say(f"VERDICT: net MACD timing {verdict} equal-weight buy-and-hold on CAGR.")

    cum_bnh = (1 + gross["bnh"].fillna(0)).cumprod()
    cum_net = (1 + net["strat"].fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cum_bnh.index, cum_bnh, label="Equal-weight buy-and-hold", lw=1.3)
    ax.plot(cum_net.index, cum_net, label=f"MACD {FAST}/{SLOW}/{SIGNAL} net", lw=1.3)
    ax.set_yscale("log")
    ax.set_title("Standard MACD long-only vs buy-and-hold (EODHD UK, survivorship-free)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"saved -> {OUT.name}, {PNG.name}")


if __name__ == "__main__":
    main()
