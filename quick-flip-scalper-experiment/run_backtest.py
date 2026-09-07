"""Run the quick-flip scalper across the S&P 500 5-minute data and report.

The simulation runs ONCE per name-day at threshold 0 (recording each day's
C1-range / ATR ratio). Liquidity threshold, transaction cost and trade side are
then applied as post-processing sweeps, so we never re-simulate.

    python run_backtest.py
"""
from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from scalper import Candle, Params, Side, Stage, simulate_day

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "candle-data" / "data"
NY = "America/New_York"
OPEN = time(9, 30)
CLOSE = time(16, 0)

THRESHOLDS = [0.25, 0.5, 0.75]
COSTS_BPS = [0.0, 5.0, 10.0]           # round-trip
NEEDED_BARS = Params().c1_bars + Params().box_bars   # first 18 bars must exist


def daily_atr(session_by_date: pd.core.groupby.DataFrameGroupBy) -> pd.Series:
    daily = session_by_date.agg(hi=("high", "max"), lo=("low", "min"),
                                cl=("close", "last"))
    prev = daily["cl"].shift(1)
    tr = np.maximum(daily.hi - daily.lo,
                    np.maximum((daily.hi - prev).abs(), (daily.lo - prev).abs()))
    return tr.rolling(14).mean().shift(1)   # ATR as of prior close


def day_candles(day: pd.DataFrame) -> list[Candle] | None:
    """First NEEDED_BARS contiguous 5-min bars from 09:30; None if incomplete."""
    day = day.sort_index()
    day = day[(day.index.time >= OPEN) & (day.index.time < CLOSE)]
    if len(day) < NEEDED_BARS:
        return None
    head = day.iloc[:NEEDED_BARS]
    # require the exact 5-min grid 09:30, 09:35, ... with no gaps
    expected = pd.date_range(head.index[0], periods=NEEDED_BARS, freq="5min")
    if head.index[0].time() != OPEN or not head.index.equals(expected):
        return None
    return [Candle(r.open, r.high, r.low, r.close) for r in head.itertuples()]


def run() -> pd.DataFrame:
    tickers = sorted(p.stem for p in DATA.glob("*.parquet") if not p.stem.startswith("_"))
    rows = []
    stage_counts = {s: 0 for s in Stage}
    incomplete = 0

    for t in tickers:
        df = pd.read_parquet(DATA / f"{t}.parquet").tz_convert(NY).sort_index()
        df["d"] = df.index.date
        atr = daily_atr(df.groupby("d"))
        for d, day in df.groupby("d"):
            candles = day_candles(day)
            if candles is None:
                incomplete += 1
                continue
            res = simulate_day(candles, atr.get(d), Params())
            stage_counts[res.stage] += 1
            if res.stage is Stage.TRADED:
                risk = abs(res.stop - res.entry_px) / res.entry_px
                rows.append({
                    "ticker": t, "date": d, "side": res.side.value,
                    "pattern": res.pattern.value, "liq_ratio": res.liq_ratio,
                    "gross_ret": res.gross_ret, "risk_frac": risk,
                    "exit": res.exit_reason.value, "mae": res.mae, "mfe": res.mfe,
                })
    trades = pd.DataFrame(rows)

    # ---- funnel ---------------------------------------------------------- #
    print("=== FUNNEL ===")
    print(f"  incomplete/opening-gap days skipped: {incomplete}")
    for s in Stage:
        print(f"  {s.value:10s} {stage_counts[s]:7d}")
    total_search = sum(stage_counts[s] for s in
                       (Stage.NO_SIGNAL, Stage.NO_FILL, Stage.TRADED))
    print(f"  -> days that reached signal search: {total_search}")

    trades.to_parquet(HERE / "trades.parquet", index=False)
    return trades


def metrics(tr: pd.DataFrame, cost_bps: float) -> dict:
    net = tr["gross_ret"] - cost_bps / 1e4
    r_net = net / tr["risk_frac"]
    wins = net > 0
    pf_num = net[net > 0].sum()
    pf_den = -net[net < 0].sum()
    return {
        "n": len(tr),
        "win%": round(100 * wins.mean(), 1) if len(tr) else np.nan,
        "avg_net_bps": round(1e4 * net.mean(), 1) if len(tr) else np.nan,
        "med_net_bps": round(1e4 * net.median(), 1) if len(tr) else np.nan,
        "avg_R": round(r_net.mean(), 3) if len(tr) else np.nan,
        "PF": round(pf_num / pf_den, 2) if pf_den > 0 else np.inf,
    }


def report(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("No trades.")
        return
    print("\n=== EXIT MIX (all trades) ===")
    print(trades["exit"].value_counts(normalize=True).mul(100).round(1).to_string())

    for side_label, sub_all in [("BOTH", trades),
                                ("SHORT", trades[trades.side == "short"]),
                                ("LONG", trades[trades.side == "long"])]:
        print(f"\n=== {side_label} ===")
        recs = []
        for thr in THRESHOLDS:
            sub = sub_all[sub_all.liq_ratio >= thr]
            for cost in COSTS_BPS:
                m = metrics(sub, cost)
                recs.append({"thr": thr, "cost_bps": cost, **m})
        print(pd.DataFrame(recs).to_string(index=False))


if __name__ == "__main__":
    trades = run()
    report(trades)
    print(f"\nSaved {len(trades)} trades to trades.parquet")
    sys.exit(0)
