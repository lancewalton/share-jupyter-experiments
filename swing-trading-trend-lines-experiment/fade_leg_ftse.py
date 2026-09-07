"""The fade leg: a daily MEAN-REVERSION entry on FTSE, gated to COMPRESSION.

The breakout (follow) leg lost robustly and the vol regime steered it the wrong
way - but compression was the *less-bad* regime, hinting that daily single-name
edge, if any, lives in reversion, not momentum. So test the opposite entry:

  - Signal (causal): z = (close - MA20) / SD20, evaluated at each close.
  - Entry (contrarian): z <= -TH -> LONG (oversold); z >= +TH -> SHORT. Fill at
    the NEXT day's open (no look-ahead).
  - Exit: revert to the mean (z crosses 0) at the close, OR a disaster stop at
    |z| >= STOP, OR a max hold of H days. Exit at that day's close.
  - Gate: keep trades entered in COMPRESSION (vol_state = 20d/100d realised vol
    < 1), the regime the breakout study flagged.

Pre-registered: MA/SD lookback 20, TH=2.0, STOP=3.5, H=10. TH sensitivity only.
Costs and the full stress harness (year-block bootstrap, beta-neutral alpha,
concentration, pre/post-2013 OOS, buy-and-hold) are reused from the breakout
study for an apples-to-apples comparison.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u fade_leg_ftse.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import load_ftse
from backtest import CostModel, trade_return
from backtest_run import UNIVERSE
from regime_gate_ftse import (COST_SWEEP, OOS_YEAR, VOL_THR, add_costs,
                              market_over_hold, metrics, per_ticker_totals,
                              vol_state, year_block_boot)

HERE = Path(__file__).resolve().parent
LOOKBACK = 20
TH = 2.0
TH_SENS = [1.5, 2.0, 2.5]
STOP = 3.5
MAX_HOLD = 10
WARMUP = 100                         # need 100 for vol_state
OUT = HERE / "fade_leg_ftse_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def fade_trades(df: pd.DataFrame, th: float) -> list[dict]:
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    idx = df.index
    ma = pd.Series(closes).rolling(LOOKBACK).mean().to_numpy()
    sd = pd.Series(closes).rolling(LOOKBACK).std().to_numpy()
    z = (closes - ma) / sd
    vs = vol_state(df)
    n = len(df)
    out, i = [], WARMUP
    while i < n - 1:
        if not np.isfinite(z[i]):
            i += 1
            continue
        if z[i] <= -th:
            direction = "LONG"
        elif z[i] >= th:
            direction = "SHORT"
        else:
            i += 1
            continue
        entry_idx = i + 1
        entry_px = float(opens[entry_idx])
        exit_idx = None
        for j in range(entry_idx, min(entry_idx + MAX_HOLD, n - 1) + 1):
            zj = z[j]
            reverted = (direction == "LONG" and zj >= 0) or (direction == "SHORT" and zj <= 0)
            stopped = (direction == "LONG" and zj <= -STOP) or (direction == "SHORT" and zj >= STOP)
            if reverted or stopped or j == min(entry_idx + MAX_HOLD, n - 1):
                exit_idx = j
                break
        exit_px = float(closes[exit_idx])
        days = max(1, (idx[exit_idx] - idx[entry_idx]).days)
        out.append({
            "ticker": None, "entry": idx[entry_idx], "exit": idx[exit_idx],
            "direction": direction, "entry_px": entry_px, "exit_px": exit_px,
            "days": days, "gross": trade_return(direction, entry_px, exit_px, days, CostModel(0.0, 0.0)),
            "vol_state": float(vs.get(idx[entry_idx], np.nan)), "year": idx[entry_idx].year,
        })
        i = exit_idx + 1
    return out


def build(th: float) -> tuple[pd.DataFrame, pd.Series, dict]:
    dfs = {t: load_ftse.load(t) for t in UNIVERSE}
    closes = pd.DataFrame({t: d["close"] for t, d in dfs.items()}).sort_index()
    mkt_idx = (1 + closes.pct_change().mean(axis=1)).cumprod()
    bh = {t: float(d["close"].iloc[-1] / d["close"].iloc[0] - 1) for t, d in dfs.items()}
    rows = []
    for t, df in dfs.items():
        for r in fade_trades(df, th):
            r["ticker"] = t
            rows.append(r)
    tr = pd.DataFrame(rows).sort_values("entry").reset_index(drop=True)
    return add_costs(tr), mkt_idx, bh


def stress_fade(tr: pd.DataFrame, mkt_idx: pd.Series, bh: dict, name: str) -> None:
    tr = tr.copy().reset_index(drop=True)
    if len(tr) == 0:
        say(f"\n===== STRESS: {name} -- no trades =====")
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
    by_tk = tr.groupby("ticker").net_10.sum().sort_values()
    by_yr = tr.groupby("year").net_10.sum().sort_values()
    total = tr.net_10.sum()
    if abs(total) > 1e-9:
        say(f"  concentration (net): names +{(by_tk>0).sum()}/-{(by_tk<0).sum()}, "
            f"top-3 names {by_tk.tail(3).sum()/total*100:.0f}% of total; "
            f"years +{(by_yr>0).sum()}/-{(by_yr<0).sum()}, "
            f"top-3 years {by_yr.tail(3).sum()/total*100:.0f}%")
    pre, post = tr[tr.year < OOS_YEAR], tr[tr.year >= OOS_YEAR]
    say(f"  OOS split @ {OOS_YEAR}: pre  gross {metrics(pre,'gross')}")
    say(f"  {'':16s}      pre  net10 {metrics(pre,'net_10')}")
    say(f"  {'':16s}      post gross {metrics(post,'gross')}")
    say(f"  {'':16s}      post net10 {metrics(post,'net_10')}")
    say("  " + per_ticker_totals(tr, "net_10", bh))


def main() -> None:
    say(f"Building the fade (mean-reversion) book, TH={TH}, gate=compression...")
    tr, mkt_idx, bh = build(TH)
    cov = tr.vol_state.notna()
    say(f"{len(tr)} fade trades; {cov.sum()} have a causal vol_state "
        f"({100*cov.mean():.0f}% coverage).")
    trc = tr[cov].copy()

    say("\n" + "#" * 68)
    say("# DISCRIMINATOR: does the fade work better in compression?")
    say("#" * 68)
    comp = trc[trc.vol_state < VOL_THR]
    exp = trc[trc.vol_state >= VOL_THR]
    for col, lab in [("gross", "gross"), ("net_10", "net(10bps+5%)")]:
        say(f"  {lab:16s} | compression(vs<{VOL_THR}) {metrics(comp,col)}")
        say(f"  {'':16s} | expansion(vs>={VOL_THR}) {metrics(exp,col)}")

    say("\n" + "#" * 68)
    say("# TH sensitivity (compression-gated fade book)")
    say("#" * 68)
    for th in TH_SENS:
        t2, _, _ = build(th)
        g = t2[(t2.vol_state.notna()) & (t2.vol_state < VOL_THR)]
        say(f"  TH={th}:  gross {metrics(g,'gross')}")
        say(f"  {'':9s}  net10 {metrics(g,'net_10')}")

    stress_fade(comp, mkt_idx, bh, f"FADE compression-gated (TH={TH}, vs<{VOL_THR})")
    stress_fade(trc, mkt_idx, bh, f"FADE all regimes (TH={TH}, for contrast)")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
