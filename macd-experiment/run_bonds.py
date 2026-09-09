"""MACD on bonds - where the programme's prior says trend-following should fare better.

Equities whipsaw a trend exit (V-shaped crashes); bonds trend (the sustained 2022
sell-off rewards a trend exit). Same rig as the equity phases (standard 12/26/9,
concentrated book vs equal-weight B&H, net of costs), plus a sweep, a permutation null,
and a 2022 spotlight. Thin, correlated cross-section (6 US bond ETFs) -- indicative.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u run_bonds.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macd.indicator import long_state, macd_cross_state
from macd.backtest import portfolio, extract_trades
from macd.metrics import cagr, sharpe, max_drawdown, hit_rate, mean_payoff
from macd.surrogate import permute_within_columns

HERE = Path(__file__).resolve().parent
BONDS = HERE.parent / "momentum-strategy" / "bond_closes.parquet"
COST = 10 / 1e4
FASTS = [5, 8, 10, 12, 15, 19, 24]
SLOWS = [20, 26, 32, 40, 50, 60, 80]
N_NULL = 500
RNG = np.random.default_rng(20260909)
OUT = HERE / "bonds_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def all_trades(positions, returns) -> np.ndarray:
    out: list[float] = []
    for c in positions.columns:
        p = positions[c] & returns[c].notna()
        if p.any():
            out.extend(extract_trades(p, returns[c].fillna(0.0)))
    return np.asarray(out, dtype=float)


def main() -> None:
    prices = pd.read_parquet(BONDS).sort_index()
    returns = prices.pct_change()
    bh = returns.mean(axis=1)
    say(f"Bonds: {prices.shape[1]} ETFs {list(prices.columns)}, "
        f"{prices.index.min().date()} -> {prices.index.max().date()}")
    say(f"buy-and-hold (EW): CAGR {100*cagr(bh):+.2f}%  Sharpe {sharpe(bh):.2f}  "
        f"maxDD {100*max_drawdown(bh):.0f}%\n")

    pos = long_state(prices, fast=12, slow=26, signal=9)
    gross = portfolio(pos, returns, cost=0.0, concentrate=True)
    net = portfolio(pos, returns, cost=COST, concentrate=True)
    tr = all_trades(pos, returns)

    say("#" * 76)
    say("# STANDARD MACD 12/26/9 on bonds (concentrated book)")
    say("#" * 76)
    for tag, r in [("buy-and-hold (EW)", bh), ("MACD gross", gross["strat"]),
                   ("MACD net 10bps", net["strat"])]:
        say(f"  {tag:<20} CAGR {100*cagr(r):+6.2f}%  Sharpe {sharpe(r):+5.2f}  "
            f"maxDD {100*max_drawdown(r):+6.1f}%  excess {100*(cagr(r)-cagr(bh)):+6.2f}%")
    say(f"  trades {len(tr):,}  hit-rate {100*hit_rate(tr):.1f}%  "
        f"mean payoff {100*mean_payoff(tr):+.3f}%\n")

    # 2022 spotlight - the sustained sell-off
    y = pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01")
    m = (bh.index >= y[0]) & (bh.index < y[1])
    say("2022 (sustained sell-off):")
    say(f"  B&H {100*cagr(bh[m]):+.2f}%   MACD net {100*cagr(net['strat'][m]):+.2f}%   "
        f"(trend-exit should help here)\n")

    say("#" * 76)
    say("# SWEEP - net excess CAGR over B&H (signal=9, concentrated, 10bps)")
    say("#" * 76)
    grid = pd.DataFrame(index=FASTS, columns=SLOWS, dtype=float)
    for f in FASTS:
        for s in SLOWS:
            if f < s:
                p = long_state(prices, fast=f, slow=s, signal=9)
                grid.loc[f, s] = 100 * (cagr(portfolio(p, returns, cost=COST,
                                        concentrate=True)["strat"]) - cagr(bh))
    say(grid.to_string(float_format=lambda x: f"{x:+.1f}"))
    beat = int((grid > 0).sum().sum())
    say(f"\nparameter sets beating B&H net: {beat} of {int(grid.notna().sum().sum())}")
    flat = grid.stack().sort_values(ascending=False)
    (bf, bs), best = flat.index[0], flat.iloc[0]
    say(f"best: {bf}/{bs} excess {best:+.2f}%\n")

    say("#" * 76)
    say(f"# PERMUTATION NULL at 12/26/9 ({N_NULL} surrogates)")
    say("#" * 76)
    real = cagr(net["strat"]) - cagr(bh)
    null = np.empty(N_NULL)
    for k in range(N_NULL):
        sr = permute_within_columns(returns, RNG)
        ps = (1 + sr.fillna(0.0)).cumprod().where(sr.notna())
        st = portfolio(long_state(ps, 12, 26, 9), sr, cost=COST, concentrate=True)["strat"]
        null[k] = cagr(st) - cagr(sr.mean(axis=1))
    p = float((null >= real).mean())
    say(f"real excess {100*real:+.2f}%   null mean {100*null.mean():+.2f}%  "
        f"sd {100*null.std():.2f}%  95th {100*np.percentile(null,95):+.2f}%")
    say(f"p-value (null >= real): {p:.3f}  -> "
        f"{'real edge beyond noise' if p <= 0.05 else 'no edge beyond noise'}")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
