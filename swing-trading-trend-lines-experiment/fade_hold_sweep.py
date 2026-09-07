"""Does a LONGER fixed hold let the mean-reversion fade amortise the fixed spread?

The spread is a fixed ~20 bps round-trip; financing is ~5%/yr (~1.37 bps/day). So a
longer hold helps only if the reversion keeps growing faster than financing accrues.
Test it on the strongest fade signal (z-score compression fade, the +0.41% gross
one): same entry, but hold a FIXED horizon H, swept from 1 to 40 days.

Entry (causal): z = (close - MA20)/SD20; when |z| >= TH and the day is in a
compression regime (vol_state = 20d/100d realised vol < 1), fade at the next open.
Exit at the close H trading days later. One position at a time per name.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u fade_hold_sweep.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import load_ftse
from backtest import CostModel, trade_return
from backtest_run import UNIVERSE
from regime_gate_ftse import (OOS_YEAR, market_over_hold, metrics,
                              per_ticker_totals, vol_state, year_block_boot)

HERE = Path(__file__).resolve().parent
LOOKBACK = 20
TH = 2.0
VOL_THR = 1.0
WARMUP = 100
HOLDS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]
NET = CostModel(10, 0.05)
OUT = HERE / "fade_hold_sweep_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def fade_book(dfs: dict, hold: int) -> pd.DataFrame:
    rows = []
    for t, df in dfs.items():
        close = df["close"].to_numpy()
        op = df["open"].to_numpy()
        idx = df.index
        z = ((df["close"] - df["close"].rolling(LOOKBACK).mean())
             / df["close"].rolling(LOOKBACK).std()).to_numpy()
        vs = vol_state(df).to_numpy()
        n = len(df)
        i = WARMUP
        while i < n - 1:
            if np.isfinite(z[i]) and np.isfinite(vs[i]) and vs[i] < VOL_THR and abs(z[i]) >= TH:
                direction = "LONG" if z[i] <= -TH else "SHORT"   # fade the stretch
                e = i + 1
                x = min(e + hold, n - 1)
                entry_px, exit_px = float(op[e]), float(close[x])
                days = max(1, (idx[x] - idx[e]).days)
                rows.append({
                    "ticker": t, "entry": idx[e], "exit": idx[x], "direction": direction,
                    "gross": trade_return(direction, entry_px, exit_px, days, CostModel(0.0, 0.0)),
                    "net_10": trade_return(direction, entry_px, exit_px, days, NET),
                    "year": idx[e].year,
                })
                i = x + 1
            else:
                i += 1
    return pd.DataFrame(rows)


def main() -> None:
    say(f"Loading data; z-score compression fade, TH={TH}, hold sweep {HOLDS}...")
    dfs = {t: load_ftse.load(t) for t in UNIVERSE}
    closes = pd.DataFrame({t: d["close"] for t, d in dfs.items()}).sort_index()
    mkt_idx = (1 + closes.pct_change().mean(axis=1)).cumprod()
    bh = {t: float(d["close"].iloc[-1] / d["close"].iloc[0] - 1) for t, d in dfs.items()}

    say("\n" + "#" * 68)
    say("# Hold-horizon sweep (net = gross - 20bps spread - 5%/yr financing)")
    say("#" * 68)
    say(f"{'hold':>5} {'n':>6} {'win%':>6} {'gross/tr%':>10} {'net/tr%':>9} {'PF':>5}")
    best = None
    books = {}
    for h in HOLDS:
        b = fade_book(dfs, h)
        books[h] = b
        net, gr = b.net_10, b.gross
        pfn, pfd = net[net > 0].sum(), -net[net < 0].sum()
        pf = pfn / pfd if pfd > 0 else float("inf")
        say(f"{h:>5} {len(b):>6} {100*(net>0).mean():>6.1f} {100*gr.mean():>10.3f} "
            f"{100*net.mean():>9.3f} {pf:>5.2f}")
        gm = net.mean()
        if best is None or gm > best[0]:
            best = (gm, h)

    say(f"\nBest net hold = {best[1]}d (net mean/tr {100*best[0]:+.3f}%)")
    stress = books[best[1]]
    tr = stress.copy().reset_index(drop=True)
    say(f"\n===== STRESS: fade hold={best[1]}d  (n={len(tr)}, names={tr.ticker.nunique()}, "
        f"years {tr.year.min()}-{tr.year.max()}) =====")
    for col, lab in [("gross", "gross"), ("net_10", "net(10bps+5%)")]:
        m, lo, hi = year_block_boot(tr, col)
        star = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
        say(f"  year-block bootstrap {lab:16s}: mean/tr {100*m:+.3f}%  "
            f"95% CI [{100*lo:+.3f}%, {100*hi:+.3f}%]{star}")
    sign = np.where(tr.direction == "LONG", 1.0, -1.0)
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
            f"years +{(by_yr>0).sum()}/-{(by_yr<0).sum()}, "
            f"top-3 years {by_yr.tail(3).sum()/total*100:.0f}%")
    pre, post = tr[tr.year < OOS_YEAR], tr[tr.year >= OOS_YEAR]
    say(f"  OOS split @ {OOS_YEAR}: pre  net10 {metrics(pre,'net_10')}")
    say(f"  {'':16s}      post net10 {metrics(post,'net_10')}")
    say("  " + per_ticker_totals(tr, "net_10", bh))

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
