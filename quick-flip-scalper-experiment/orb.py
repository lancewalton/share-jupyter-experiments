"""The opposite thesis: opening-range BREAKOUT (momentum), not the fade.

Same box as the quick-flip scalper, but we enter *with* the first break of the
box instead of fading it, stop at the far side of the box, and ride the move to
a horizon (box end, or end of day) rather than targeting reversion. This is the
structure the excursion diagnostics (adverse > favourable for the fade) point
at: let the continuation run, cut the small reversal.

    python orb.py

Reuses scalper.py primitives (Candle, Side, stop_entry_fill) and the same
conservative intrabar rule: a bar that straddles is assumed to hit the stop.
"""
from __future__ import annotations

from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from scalper import Candle, Side, stop_entry_fill

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "candle-data" / "data"
NY = "America/New_York"
OPEN = time(9, 30)
CLOSE = time(16, 0)
C1_BARS = 3
BOX_BARS = 15                 # 75-minute entry window after C1
THRESHOLDS = [0.0, 0.25, 0.5, 0.75]
COSTS_BPS = [0.0, 5.0, 10.0]


def preload() -> list[tuple[list[Candle], float]]:
    """Every usable name-day as (full contiguous session candles, prior ATR14)."""
    out = []
    tickers = sorted(p.stem for p in DATA.glob("*.parquet") if not p.stem.startswith("_"))
    for t in tickers:
        df = pd.read_parquet(DATA / f"{t}.parquet").tz_convert(NY).sort_index()
        df["d"] = df.index.date
        daily = df.groupby("d").agg(hi=("high", "max"), lo=("low", "min"), cl=("close", "last"))
        prev = daily["cl"].shift(1)
        tr = np.maximum(daily.hi - daily.lo,
                        np.maximum((daily.hi - prev).abs(), (daily.lo - prev).abs()))
        atr = tr.rolling(14).mean().shift(1)
        for d, day in df.groupby("d"):
            a = atr.get(d)
            if a is None or not np.isfinite(a) or a <= 0:
                continue
            day = day[(day.index.time >= OPEN) & (day.index.time < CLOSE)].sort_index()
            if len(day) < C1_BARS + 2 or day.index[0].time() != OPEN:
                continue
            grid = pd.date_range(day.index[0], periods=len(day), freq="5min")
            n = 0
            while n < len(day) and day.index[n] == grid[n]:
                n += 1
            if n < C1_BARS + 2:
                continue
            head = day.iloc[:n]
            out.append((t, d, [Candle(r.open, r.high, r.low, r.close) for r in head.itertuples()],
                        float(a)))
    return out


def simulate_orb(candles: list[Candle], atr: float, exit_mode: str) -> dict | None:
    c1 = candles[:C1_BARS]
    c1_hi = max(c.high for c in c1)
    c1_lo = min(c.low for c in c1)
    liq_ratio = (c1_hi - c1_lo) / atr
    box_start = C1_BARS
    box_end = min(len(candles) - 1, C1_BARS + BOX_BARS - 1)
    exit_end = box_end if exit_mode == "box_end" else len(candles) - 1

    # first break of the box, entered in the breakout direction
    entry = None
    for i in range(box_start, box_end + 1):
        bar = candles[i]
        up, dn = bar.high >= c1_hi, bar.low <= c1_lo
        if up and dn:
            side = Side.LONG if bar.close >= bar.open else Side.SHORT
        elif up:
            side = Side.LONG
        elif dn:
            side = Side.SHORT
        else:
            continue
        level = c1_hi if side is Side.LONG else c1_lo
        fill = stop_entry_fill(bar, side, level)
        if fill is not None:
            entry = (i, side, fill)
            break
    if entry is None:
        return {"traded": False, "liq_ratio": liq_ratio}

    i, side, entry_px = entry
    stop = c1_lo if side is Side.LONG else c1_hi
    risk = abs(stop - entry_px) / entry_px
    if risk <= 0:
        return {"traded": False, "liq_ratio": liq_ratio}

    mae = mfe = 0.0
    exit_px, reason = candles[exit_end].close, "time"
    for k in range(i, exit_end + 1):
        bar = candles[k]
        if side is Side.LONG:
            adv, fav = bar.low, bar.high
            hit = bar.low <= stop
        else:
            adv, fav = bar.high, bar.low
            hit = bar.high >= stop
        signed = (lambda px: (px - entry_px) / entry_px) if side is Side.LONG \
            else (lambda px: (entry_px - px) / entry_px)
        mae = max(mae, -signed(adv))
        mfe = max(mfe, signed(fav))
        if hit:
            exit_px, reason = stop, "stop"
            break

    gross = ((exit_px - entry_px) / entry_px) if side is Side.LONG else ((entry_px - exit_px) / entry_px)
    return {"traded": True, "liq_ratio": liq_ratio, "side": side.value,
            "gross_ret": gross, "risk_frac": risk, "exit": reason,
            "mae": mae, "mfe": mfe}


def metrics(tr: pd.DataFrame, cost_bps: float) -> dict:
    net = tr.gross_ret - cost_bps / 1e4
    r = net / tr.risk_frac
    pf_num, pf_den = net[net > 0].sum(), -net[net < 0].sum()
    return {"n": len(tr), "win%": round(100 * (net > 0).mean(), 1),
            "avg_bps": round(1e4 * net.mean(), 1), "avg_R": round(r.mean(), 3),
            "PF": round(pf_num / pf_den, 2) if pf_den > 0 else np.inf}


def report(days, exit_mode: str):
    rows = []
    for t, d, c, a in days:
        r = simulate_orb(c, a, exit_mode)
        if r and r["traded"]:
            r["ticker"], r["date"] = t, d
            rows.append(r)
    tr = pd.DataFrame(rows)
    print(f"\n########## EXIT = {exit_mode}  ({len(tr)} trades) ##########")
    print("exit mix:", tr.exit.value_counts(normalize=True).mul(100).round(1).to_dict())
    print(f"MFE median {(tr.mfe/tr.risk_frac).median():.2f}R  "
          f"MAE median {(tr.mae/tr.risk_frac).median():.2f}R")
    for label, sub_all in [("BOTH", tr), ("LONG", tr[tr.side == "long"]),
                           ("SHORT", tr[tr.side == "short"])]:
        recs = []
        for thr in THRESHOLDS:
            sub = sub_all[sub_all.liq_ratio >= thr]
            if len(sub) < 50:
                continue
            for cost in COSTS_BPS:
                recs.append({"thr": thr, "cost": cost, **metrics(sub, cost)})
        print(f"\n--- {label} ---")
        print(pd.DataFrame(recs).to_string(index=False))
    tr.to_parquet(HERE / f"orb_trades_{exit_mode}.parquet", index=False)


def main():
    print("Preloading full sessions...", flush=True)
    days = preload()
    print(f"{len(days)} name-days in memory.")
    for mode in ("box_end", "eod"):
        report(days, mode)


if __name__ == "__main__":
    main()
