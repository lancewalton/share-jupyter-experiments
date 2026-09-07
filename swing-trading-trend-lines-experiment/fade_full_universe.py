"""Out-of-sample test of the longer-hold compression fade on the FULL FTSE universe.

The hold=8d z-score compression fade (TH=2, MA/SD 20) was the best result on our
20-name set. Here we apply the SAME pre-registered parameters (no re-sweeping) to
every FTSE name we have (~120), and in particular to the ~100 names never used
before — the true out-of-sample set. Does the small net edge survive, or was it
the 20-name selection?

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u fade_full_universe.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import load_ftse
from backtest_run import UNIVERSE as ORIG20
from fade_hold_sweep import fade_book  # pre-registered: TH=2, MA/SD 20, compression gate
from regime_gate_ftse import (OOS_YEAR, market_over_hold, metrics,
                              per_ticker_totals, year_block_boot)

HERE = Path(__file__).resolve().parent
HOLD = 8                     # pre-registered from the 20-name sweep
PLATEAU = [5, 8, 15]
OUT = HERE / "fade_full_universe_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def market_and_bh(dfs: dict) -> tuple[pd.Series, dict]:
    closes = pd.DataFrame({t: d["close"] for t, d in dfs.items()}).sort_index()
    mkt_idx = (1 + closes.pct_change().mean(axis=1)).cumprod()
    bh = {t: float(d["close"].iloc[-1] / d["close"].iloc[0] - 1) for t, d in dfs.items()}
    return mkt_idx, bh


def report(dfs: dict, label: str) -> None:
    tr = fade_book(dfs, HOLD).reset_index(drop=True)
    mkt_idx, bh = market_and_bh(dfs)
    say(f"\n===== {label}  (names={len(dfs)}, hold={HOLD}d, n={len(tr)}) =====")
    if len(tr) < 50:
        say("  too few trades."); return
    say(f"  headline: gross {metrics(tr,'gross')}")
    say(f"            net10 {metrics(tr,'net_10')}")
    for col, lab in [("gross", "gross"), ("net_10", "net(10bps+5%)")]:
        m, lo, hi = year_block_boot(tr, col)
        star = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
        say(f"  year-block bootstrap {lab:16s}: mean/tr {100*m:+.3f}%  "
            f"95% CI [{100*lo:+.3f}%, {100*hi:+.3f}%]{star}")
    sign = np.where(tr.direction == "LONG", 1.0, -1.0)
    tr = tr.copy()
    tr["alpha"] = tr.gross.to_numpy() - sign * market_over_hold(tr, mkt_idx)
    m, lo, hi = year_block_boot(tr, "alpha")
    star = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
    say(f"  beta-neutral alpha (gross)      : mean/tr {100*m:+.3f}%  "
        f"95% CI [{100*lo:+.3f}%, {100*hi:+.3f}%]{star}")
    by_tk = tr.groupby("ticker").net_10.sum().sort_values()
    by_yr = tr.groupby("year").net_10.sum().sort_values()
    total = tr.net_10.sum()
    if abs(total) > 1e-9:
        say(f"  concentration (net): names +{(by_tk>0).sum()}/-{(by_tk<0).sum()}, "
            f"top-3 names {by_tk.tail(3).sum()/total*100:.0f}% of total; "
            f"years +{(by_yr>0).sum()}/-{(by_yr<0).sum()}, top-3 years {by_yr.tail(3).sum()/total*100:.0f}%")
    pre, post = tr[tr.year < OOS_YEAR], tr[tr.year >= OOS_YEAR]
    say(f"  era split @ {OOS_YEAR}: pre  net10 {metrics(pre,'net_10')}")
    say(f"  {'':15s}     post net10 {metrics(post,'net_10')}")
    say("  " + per_ticker_totals(tr, "net_10", bh))
    # hold plateau (robustness, not selection)
    plateau = "  ".join(f"{h}d {100*fade_book(dfs,h).net_10.mean():+.3f}%" for h in PLATEAU)
    say(f"  hold plateau net/tr: {plateau}")


def main() -> None:
    allnames = load_ftse.universe()
    say(f"FTSE universe: {len(allnames)} CSVs. Loading (repairing bad ticks)...")
    dfs, failed = {}, []
    for t in allnames:
        try:
            d = load_ftse.load(t)
            if len(d) > 150:
                dfs[t] = d
        except Exception as e:
            failed.append((t, str(e)[:40]))
    say(f"loaded {len(dfs)} usable; skipped {len(allnames)-len(dfs)} (short/failed).")

    held = {t: d for t, d in dfs.items() if t not in set(ORIG20)}
    orig = {t: d for t, d in dfs.items() if t in set(ORIG20)}

    report(held, "HELD-OUT names (never used before) -- TRUE OUT-OF-SAMPLE")
    report(dfs, "FULL universe")
    report(orig, "Original 20 (reference)")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
