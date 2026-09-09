"""Phase 2 - parameter sweep, stability (plateau not peak), and a permutation null.

Sweeps the MACD fast/slow lengths (signal fixed at 9) over the investable universe,
net of costs, and asks three questions the README's anti-overfitting protocol
demands: (1) does any parameter set beat equal-weight buy-and-hold net; (2) is any
good region a contiguous plateau rather than an isolated peak; (3) does the best set
still look good on serially-shuffled surrogate returns (the permutation null)?

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u run_phase2_sweep.py
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
from macd.backtest import portfolio
from macd.metrics import cagr, sharpe
from macd.surrogate import permute_within_columns

HERE = Path(__file__).resolve().parent
COST = 10 / 1e4
MIN_OBS = 750
UNIVERSE_N = 350
LIQ_WINDOW = 252
SIGNAL = 9
FASTS = [5, 8, 10, 12, 15, 19, 24]
SLOWS = [20, 26, 32, 40, 50, 60, 80]
N_NULL = 200
RNG = np.random.default_rng(20260909)
OUT = HERE / "phase2_sweep_results.txt"
PNG = HERE / "phase2_sweep.png"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def price_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    return (1 + returns.fillna(0.0)).cumprod().where(returns.notna())


def net_excess(prices, returns_masked, bh, fast, slow) -> tuple[float, float]:
    pos = long_state(prices, fast=fast, slow=slow, signal=SIGNAL)
    strat = portfolio(pos, returns_masked, cost=COST, concentrate=True)["strat"]
    return cagr(strat) - cagr(bh), sharpe(strat)


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

    say("#" * 78)
    say("# SWEEP - net excess CAGR over equal-weight B&H (signal=9, 10bps)")
    say("#" * 78)
    grid = pd.DataFrame(index=FASTS, columns=SLOWS, dtype=float)
    sharpe_grid = pd.DataFrame(index=FASTS, columns=SLOWS, dtype=float)
    for f in FASTS:
        for s in SLOWS:
            if f >= s:
                continue
            exc, sh = net_excess(prices, returns_masked, bh, f, s)
            grid.loc[f, s] = 100 * exc
            sharpe_grid.loc[f, s] = sh
    say("excess CAGR vs B&H (%, positive = beats buy-and-hold):")
    say(grid.to_string(float_format=lambda x: f"{x:+.1f}"))
    say("")

    flat = grid.stack().sort_values(ascending=False)
    say("best 5 (fast, slow) by excess CAGR:")
    for (f, s), v in flat.head(5).items():
        say(f"  {f:>3}/{s:<3}  excess {v:+.2f}%  Sharpe {sharpe_grid.loc[f, s]:+.2f}")
    beat = int((grid > 0).sum().sum())
    total = int(grid.notna().sum().sum())
    say(f"\nparameter sets beating B&H net: {beat} of {total}")

    (bf, bs), best = flat.index[0], flat.iloc[0]
    say(f"\nSTABILITY around best {bf}/{bs} (excess {best:+.2f}%):")
    fi, si = FASTS.index(bf), SLOWS.index(bs)
    neigh = []
    for df_ in (-1, 0, 1):
        for ds in (-1, 0, 1):
            i, j = fi + df_, si + ds
            if 0 <= i < len(FASTS) and 0 <= j < len(SLOWS):
                v = grid.loc[FASTS[i], SLOWS[j]]
                if not np.isnan(v):
                    neigh.append(v)
    say(f"  neighbourhood excess CAGR: min {min(neigh):+.2f}%  max {max(neigh):+.2f}%  "
        f"mean {np.mean(neigh):+.2f}%  ({'plateau' if min(neigh) > 0 else 'isolated peak / not robust'})")
    say("")

    say("#" * 78)
    say(f"# PERMUTATION NULL at standard 12/26/9 ({N_NULL} surrogates)")
    say("#" * 78)
    real_exc, _ = net_excess(prices, returns_masked, bh, 12, 26)
    null = np.empty(N_NULL)
    for k in range(N_NULL):
        sr = permute_within_columns(returns, RNG)
        srm = sr.where(eligible)
        ps = price_from_returns(sr)
        strat = portfolio(long_state(ps, fast=12, slow=26, signal=SIGNAL), srm,
                          cost=COST, concentrate=True)["strat"]
        null[k] = cagr(strat) - cagr(srm.mean(axis=1))
    p = float((null >= real_exc).mean())
    say(f"real excess CAGR (12/26/9): {100*real_exc:+.2f}%")
    say(f"null excess CAGR: mean {100*null.mean():+.2f}%  sd {100*null.std():.2f}%  "
        f"95th pct {100*np.percentile(null, 95):+.2f}%")
    say(f"p-value (null >= real): {p:.3f}  -> "
        f"{'no evidence of a real timing edge' if p > 0.05 else 'edge beyond the null'}")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid.to_numpy(dtype=float), cmap="RdBu", aspect="auto",
                   vmin=-abs(np.nanmax(np.abs(grid.to_numpy(dtype=float)))),
                   vmax=abs(np.nanmax(np.abs(grid.to_numpy(dtype=float)))))
    ax.set_xticks(range(len(SLOWS)), SLOWS)
    ax.set_yticks(range(len(FASTS)), FASTS)
    ax.set_xlabel("slow EMA")
    ax.set_ylabel("fast EMA")
    ax.set_title("MACD net excess CAGR over B&H (%) - blue beats, red loses")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}, {PNG.name}")


if __name__ == "__main__":
    main()
