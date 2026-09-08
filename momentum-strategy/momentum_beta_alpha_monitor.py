"""Beta-vs-alpha drawdown monitor: is a portfolio drawdown market-driven or strategy-driven?

A portfolio-NAV drawdown that is mostly MARKET (beta) is buying-cheap-beta territory -- OK to
release a pre-committed reserve tranche. A drawdown where the STRATEGY fell while the market did
NOT (an alpha/momentum-specific drawdown) is doubling-down-on-a-maybe-broken-edge territory --
freeze and investigate. This classifies every month, identifies the strategy's drawdown episodes,
labels each, and reports the residual (beta-adjusted) alpha drawdown, so the rule can be checked
against history before being trusted.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u momentum_beta_alpha_monitor.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from momentum_tradeability import build, cagr, sharpe, maxdd
from momentum_multifactor import build_factors
from momentum_stamp_duty import run
from vol_target_momentum import vol_target

HERE = Path(__file__).resolve().parent
STRAT_DD_TRIGGER = -0.15       # only classify meaningful strategy drawdowns
MKT_DD_BETA = -0.10            # market this far down => the drawdown is beta-driven
OUT = HERE / "momentum_beta_alpha_monitor_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True); _lines.append(s)


def dd(series_cum):
    return series_cum / series_cum.cummax() - 1


def main() -> None:
    say("Building recommended build + market series...")
    M, mret, liq, signal = build()
    quality, _v, _ = build_factors(M)
    lowvol = -mret.rolling(12).std().shift(1)
    P0, _ = run(M, mret, liq, [signal, quality, lowvol], 69, 0.0)
    r_tilt = P0["r"].dropna()
    vt, _w = vol_target(r_tilt, r_tilt.std() * np.sqrt(12), 1.0)
    idx = vt.index
    r_strat = vt
    r_mkt = P0["uni"].reindex(idx)                          # eligible-universe EW buy-and-hold

    beta = float(np.cov(r_strat, r_mkt)[0, 1] / np.var(r_mkt))
    say(f"{len(idx)} months; strat CAGR {100*cagr(r_strat):.1f}%/Sharpe {sharpe(r_strat):.2f}, "
        f"market CAGR {100*cagr(r_mkt):.1f}%; beta {beta:.2f}\n")

    cum_s = (1 + r_strat).cumprod()
    cum_m = (1 + r_mkt).cumprod()
    r_alpha = r_strat - beta * r_mkt                        # beta-neutralised strategy return
    cum_a = (1 + r_alpha).cumprod()
    dd_s, dd_m, dd_a = dd(cum_s), dd(cum_m), dd(cum_a)

    # classify each month in a meaningful strategy drawdown
    label = pd.Series("ok", index=idx)
    indd = dd_s < STRAT_DD_TRIGGER
    label[indd & (dd_m <= MKT_DD_BETA)] = "BETA (release tranche)"
    label[indd & (dd_m > MKT_DD_BETA)] = "ALPHA (freeze/investigate)"

    say("#" * 90)
    say("# THE RULE, applied monthly")
    say("#" * 90)
    say(f"  trigger: classify only when strategy drawdown < {100*STRAT_DD_TRIGGER:.0f}%")
    say(f"  BETA  (market also down < {100*MKT_DD_BETA:.0f}%): drawdown is market-driven -> release a reserve tranche")
    say(f"  ALPHA (market not down):                    strategy-specific -> freeze, investigate decay")
    n_beta = int((label == "BETA (release tranche)").sum())
    n_alpha = int((label == "ALPHA (freeze/investigate)").sum())
    say(f"\n  months in strat drawdown >{-100*STRAT_DD_TRIGGER:.0f}%: {int(indd.sum())}  "
        f"| classified BETA {n_beta}, ALPHA {n_alpha}\n")

    # drawdown episodes: contiguous runs where dd_s < trigger; report trough + concurrent market DD
    say("#" * 90)
    say("# STRATEGY DRAWDOWN EPISODES (>15%), each classified at its trough")
    say("#" * 90)
    say(f"{'trough date':>12} {'strat DD':>9} {'market DD':>10} {'alpha DD':>9}  classification")
    runs, cur = [], None
    for t in idx:
        if dd_s[t] < STRAT_DD_TRIGGER:
            cur = cur or []
            cur.append(t)
        elif cur:
            runs.append(cur); cur = None
    if cur:
        runs.append(cur)
    for run_months in runs:
        trough = min(run_months, key=lambda t: dd_s[t])
        sdd, mdd, add_ = dd_s[trough], dd_m[trough], dd_a[trough]
        cls = "BETA  -> release" if mdd <= MKT_DD_BETA else "ALPHA -> freeze"
        say(f"{trough.date().isoformat():>12} {100*sdd:>8.0f}% {100*mdd:>9.0f}% {100*add_:>8.0f}%  {cls}")

    say("\n(alpha DD = drawdown of the beta-neutralised strategy return; a deep alpha DD with a shallow")
    say(" market DD is the signature of a momentum-specific problem, not a cheap-beta opportunity.)")
    OUT.write_text("\n".join(_lines) + "\n")
    say(f"saved -> {OUT.name}")


if __name__ == "__main__":
    main()
