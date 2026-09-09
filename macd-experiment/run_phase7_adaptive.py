"""Phase 7 - walk-forward adaptive MACD ('re-tuning until it works', done honestly).

Every 6 months, pick the fast/slow MACD that made the most money over the trailing 3
years and trade it forward; roll. The stitched forward path is genuinely out-of-sample.
Judged against four references: buy-and-hold, fixed 12/26, the best-in-hindsight param
(oracle ceiling), and - the decisive one - the SAME adaptive scheme run on serially
shuffled surrogate returns. If re-tuning is real adaptation it clears that null; if it
is a perpetual-hope machine it matches noise-chasing and still loses.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u run_phase7_adaptive.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macd.data import load_panel, top_n_mask
from macd.indicator import long_state
from macd.backtest import portfolio
from macd.metrics import cagr, sharpe
from macd.surrogate import permute_within_columns
from macd.adaptive import walk_forward

HERE = Path(__file__).resolve().parent
COST = 10 / 1e4
MIN_OBS = 750
UNIVERSE_N = 350
LIQ_WINDOW = 252
TRAIN, STEP = 756, 126           # 3y trailing, re-tune every 6 months
FASTS = [5, 8, 10, 12, 15, 19, 24]
SLOWS = [20, 26, 32, 40, 50, 60, 80]
PARAMS = [(f, s) for f in FASTS for s in SLOWS if f < s]
N_NULL = 50
RNG = np.random.default_rng(20260909)
OUT = HERE / "phase7_adaptive_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def strat_by_param(prices, returns_masked) -> dict:
    return {
        (f, s): portfolio(long_state(prices, fast=f, slow=s), returns_masked,
                          cost=COST, concentrate=True)["strat"]
        for f, s in PARAMS
    }


def price_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    return (1 + returns.fillna(0.0)).cumprod().where(returns.notna())


def main() -> None:
    say("Loading panel and fixing the investable universe...")
    prices, returns, turnover = load_panel(min_obs=MIN_OBS)
    liq = turnover.rolling(LIQ_WINDOW, min_periods=LIQ_WINDOW // 2).mean().shift(1)
    eligible = top_n_mask(liq, UNIVERSE_N)
    ever = eligible.any()
    prices, returns, eligible = prices.loc[:, ever], returns.loc[:, ever], eligible.loc[:, ever]
    returns_masked = returns.where(eligible)

    books = strat_by_param(prices, returns_masked)
    adaptive, picks = walk_forward(books, train=TRAIN, step=STEP)
    span = adaptive.index
    bh = returns_masked.mean(axis=1).loc[span]
    fixed = books[(12, 26)].loc[span]
    oracle_key = max(PARAMS, key=lambda k: cagr(books[k].loc[span]))
    oracle = books[oracle_key].loc[span]

    say(f"universe: {int(ever.sum())} names; OOS span {span[0].date()} -> {span[-1].date()} "
        f"({len(span)} days), re-tune every {STEP}d on {TRAIN}d trailing\n")

    say("#" * 80)
    say("# WALK-FORWARD ADAPTIVE MACD vs references (concentrated, net 10bps)")
    say("#" * 80)

    def line(tag, r):
        say(f"  {tag:<34} CAGR {100*cagr(r):+6.2f}%  Sharpe {sharpe(r):+5.2f}  "
            f"excess {100*(cagr(r)-cagr(bh)):+6.2f}%")

    line("buy-and-hold (EW)", bh)
    line("fixed 12/26", fixed)
    line("WALK-FORWARD ADAPTIVE", adaptive)
    line(f"oracle (best-in-hindsight {oracle_key})", oracle)

    keyseq = [k for _, k in picks]
    distinct = len(set(keyseq))
    switches = sum(1 for a, b in zip(keyseq, keyseq[1:]) if a != b)
    say(f"\n  adaptation used {distinct} distinct param sets over {len(keyseq)} re-tunes, "
        f"{switches} switches")
    say(f"  most-picked: {max(set(keyseq), key=keyseq.count)}\n")

    say("#" * 80)
    say(f"# PERMUTATION NULL - the SAME adaptive scheme on shuffled returns ({N_NULL})")
    say("#" * 80)
    real_exc = cagr(adaptive) - cagr(bh)
    null = np.empty(N_NULL)
    for k in range(N_NULL):
        sr = permute_within_columns(returns, RNG)
        srm = sr.where(eligible)
        ps = price_from_returns(sr)
        sbooks = strat_by_param(ps, srm)
        sad, _ = walk_forward(sbooks, train=TRAIN, step=STEP)
        null[k] = cagr(sad) - cagr(srm.mean(axis=1).loc[sad.index])
    p = float((null >= real_exc).mean())
    say(f"real adaptive excess CAGR:  {100*real_exc:+.2f}%")
    say(f"null adaptive excess CAGR:  mean {100*null.mean():+.2f}%  sd {100*null.std():.2f}%  "
        f"95th pct {100*np.percentile(null, 95):+.2f}%")
    say(f"p-value (null >= real): {p:.3f}  -> "
        f"{'re-tuning is real adaptation' if p <= 0.05 else 'no better than chasing noise'}")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
