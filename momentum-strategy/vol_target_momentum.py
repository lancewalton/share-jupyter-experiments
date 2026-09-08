"""Vol-target the 12-1 momentum long-only tilt: does scaling exposure to a constant
volatility cut the ~-47% drawdown for a modest return give-up?

Take the net monthly tilt returns (tiered costs), estimate trailing realised vol
(causal, 12m), set exposure w = target_vol / trailing_vol (capped), scaled return =
w*tilt + (1-w)*cash(0). Compare raw vs vol-targeted (no leverage, and capped
leverage), across target vols. Small cost on the exposure changes.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u vol_target_momentum.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from momentum_tradeability import build, run, cagr, sharpe, maxdd

HERE = Path(__file__).resolve().parent
VOL_W = 12
SCALE_COST_BPS = 10.0     # cost on |change in exposure| each month
OUT = HERE / "vol_target_momentum_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def vol_target(r: pd.Series, target_ann: float, cap: float) -> tuple[pd.Series, pd.Series]:
    rv = r.rolling(VOL_W).std().shift(1) * np.sqrt(12)      # causal trailing vol
    w = (target_ann / rv).clip(upper=cap)
    w = w.reindex(r.index)
    scaled = w * r
    turn = w.diff().abs().fillna(0.0)
    scaled = scaled - turn * SCALE_COST_BPS / 1e4
    return scaled.dropna(), w


def line(name, r):
    say(f"  {name:28} CAGR {100*cagr(r):>+6.2f}%  vol {100*r.std()*np.sqrt(12):>5.1f}%  "
        f"Sharpe {sharpe(r):>5.2f}  maxDD {100*maxdd(r):>5.0f}%")


def main() -> None:
    say("Building panel + net tilt series (tiered costs, monthly)...")
    M, mret, liq, signal = build()
    P, _ = run(M, mret, liq, signal, 1, "tier", 0.0)
    r = P["lo"].dropna()
    base_vol = r.std() * np.sqrt(12)
    say(f"raw tilt: {len(r)} months, realised vol {100*base_vol:.1f}%\n")

    say("#" * 80)
    say("# VOL-TARGETING the 12-1 momentum long-only tilt")
    say("#" * 80)
    line("RAW tilt (no targeting)", r)
    say("")
    say("  --- target = raw tilt's own vol (exposure-neutral on average) ---")
    for cap, lab in ((1.0, "no leverage (cap 1.0)"), (1.5, "capped lev 1.5")):
        vt, w = vol_target(r, base_vol, cap)
        line(f"vol-target, {lab}", vt)
        say(f"  {'':28} avg exposure {w.reindex(vt.index).mean():.2f}, "
            f"2020 return {100*((1+vt.loc['2020']).prod()-1):+.0f}%, "
            f"2008-09 {100*((1+vt.loc['2008':'2009']).prod()-1):+.0f}%")
    say("")
    say("  --- fixed target vols (no leverage, cap 1.0) ---")
    for tv in (0.10, 0.12, 0.15):
        vt, w = vol_target(r, tv, 1.0)
        line(f"target {int(tv*100)}% vol", vt)
    say("")
    say("  --- fixed target vols (capped leverage 1.5) ---")
    for tv in (0.12, 0.15, 0.20):
        vt, w = vol_target(r, tv, 1.5)
        line(f"target {int(tv*100)}% vol, lev<=1.5", vt)

    say(f"\n  (raw tilt for reference: CAGR {100*cagr(r):.2f}%, Sharpe {sharpe(r):.2f}, maxDD {100*maxdd(r):.0f}%;")
    say(f"   eligible-universe B&H: CAGR {100*cagr(P['uni']):.2f}%, Sharpe {sharpe(P['uni']):.2f})")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
