"""Cross the two ideas: within COMPRESSION regimes, does relative line duration
predict PAYOFF (not just win)? And does the doubly-gated book (compression AND a
favourable duration) clear costs? Post-processing over the cached duration trades.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u duration_regime_cross.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from line_duration_study import DEFS, PRIMARY, add_features, collect
from regime_gate_ftse import (OOS_YEAR, add_costs, market_over_hold, metrics,
                              per_ticker_totals, year_block_boot)

HERE = Path(__file__).resolve().parent
VOL_THR = 1.0
OUT = HERE / "duration_regime_cross_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def sp(part: pd.DataFrame, f: str, y: str) -> tuple[float, float, int]:
    m = part[[f, y]].dropna()
    if len(m) < 30 or m[f].nunique() < 3:
        return np.nan, np.nan, len(m)
    r, p = stats.spearmanr(m[f], m[y])
    return r, p, len(m)


def stress(tr: pd.DataFrame, mkt_idx: pd.Series, bh: dict, name: str) -> None:
    tr = tr.copy().reset_index(drop=True)
    if len(tr) < 50:
        say(f"\n===== STRESS: {name} -- only {len(tr)} trades, skipped =====")
        return
    say(f"\n===== STRESS: {name}  (n={len(tr)}, names={tr.ticker.nunique()}, "
        f"years {tr.year.min()}-{tr.year.max()}) =====")
    for col, lab in [("gross", "gross"), ("net_10", "net(10bps+5%)")]:
        m, lo, hi = year_block_boot(tr, col)
        star = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
        say(f"  year-block bootstrap {lab:16s}: mean/tr {100*m:+.3f}%  "
            f"95% CI [{100*lo:+.3f}%, {100*hi:+.3f}%]{star}")
    sign = np.where(tr.direction.to_numpy() == "LONG", 1.0, -1.0)
    tr["alpha"] = tr.gross.to_numpy() - sign * market_over_hold(tr, mkt_idx)
    m, lo, hi = year_block_boot(tr, "alpha")
    star = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
    say(f"  beta-neutral alpha (gross)      : mean/tr {100*m:+.3f}%  "
        f"95% CI [{100*lo:+.3f}%, {100*hi:+.3f}%]{star}")
    say("  " + per_ticker_totals(tr, "net_10", bh))


def main() -> None:
    say("Loading cached duration trades...")
    tr, mkt_idx, bh = collect()
    tr = add_features(add_costs(tr)).sort_values("entry").reset_index(drop=True)
    comp = tr[tr.vol_state < VOL_THR].copy()
    say(f"{len(tr)} trades; {len(comp)} in compression (vol_state<{VOL_THR}).")
    say(f"compression base: gross {metrics(comp,'gross')}")
    say(f"compression base: net10 {metrics(comp,'net_10')}")

    mid = len(comp) // 2
    ins, oos = comp.iloc[:mid], comp.iloc[mid:]
    say(f"\nsplit within compression: IS {len(ins)} (to {ins.entry.iloc[-1].date()}), "
        f"OOS {len(oos)} (from {oos.entry.iloc[0].date()})")

    say("\n" + "#" * 74)
    say("# Within COMPRESSION: does relative duration predict PAYOFF (gross)?")
    say("#" * 74)
    say(f"{'feature':16} {'IS rho(gross)':>15} {'OOS rho(gross)':>16} {'OOS rho(win)':>14}")
    for f in [f"rel_{d}" for d in DEFS]:
        ri, pi, _ = sp(ins, f, "gross")
        ro, po, no = sp(oos, f, "gross")
        rw, _, _ = sp(oos, f, "win")
        say(f"{f:16} {ri:>+10.3f}(p{pi:4.2f}) {ro:>+11.3f}(p{po:4.2f}) {rw:>+14.3f}")

    # doubly-gated: compression AND favourable duration (bottom third of PRIMARY = shorter action line)
    bot_thr = tr[PRIMARY].quantile(1 / 3)
    double = comp[comp[PRIMARY] <= bot_thr]
    say("\n" + "#" * 74)
    say(f"# Doubly-gated book: compression AND action shorter-lived ({PRIMARY} bottom third)")
    say("#" * 74)
    say(f"  gross {metrics(double,'gross')}")
    say(f"  net10 {metrics(double,'net_10')}")
    for cut, lab in [("rel_touch_span", "touch-span"), (PRIMARY, "age-from-a")]:
        b = comp[comp[cut] <= comp[cut].quantile(1 / 3)]
        say(f"  [{lab} bottom third within compression] net10 {metrics(b,'net_10')}")

    stress(double, mkt_idx, bh, f"compression x {PRIMARY} bottom third")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
