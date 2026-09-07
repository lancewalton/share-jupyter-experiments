"""Parameter-robustness sweep for the quick-flip scalper.

Question: is the negative result robust, or does some reasonable corner of the
parameter space flip it positive net of cost? We preload every name-day once,
then run each Params combo over the in-memory data and report the best net
expectancy achievable (over the liquidity-threshold choices) for each combo.

    python sweep.py
"""
from __future__ import annotations

import itertools
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from scalper import Candle, Params, Stage, simulate_day

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "candle-data" / "data"
NY = "America/New_York"
OPEN = time(9, 30)
CLOSE = time(16, 0)
MAX_BARS = 40          # 09:30-12:50; covers the longest box we test
THRESHOLDS = [0.25, 0.5, 0.75]


def preload() -> list[tuple[list[Candle], float]]:
    """Every usable name-day as (contiguous morning candles, prior-day ATR14)."""
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
            if len(day) < 5 or day.index[0].time() != OPEN:
                continue
            # maximal contiguous 5-min run from the open
            grid = pd.date_range(day.index[0], periods=min(len(day), MAX_BARS), freq="5min")
            n = 0
            while n < len(grid) and n < len(day) and day.index[n] == grid[n]:
                n += 1
            if n < 5:
                continue
            head = day.iloc[:n]
            out.append(([Candle(r.open, r.high, r.low, r.close) for r in head.itertuples()],
                        float(a)))
    return out


def run_combo(days, p: Params) -> pd.DataFrame:
    rows = []
    for candles, atr in days:
        r = simulate_day(candles, atr, p)
        if r.stage is Stage.TRADED:
            risk = abs(r.stop - r.entry_px) / r.entry_px
            if risk > 0:
                rows.append((r.side.value, r.liq_ratio, r.gross_ret, risk))
    return pd.DataFrame(rows, columns=["side", "liq_ratio", "gross_ret", "risk_frac"])


def best_net(tr: pd.DataFrame, cost_bps: float) -> dict:
    """Best avg net-R over the liquidity thresholds (and the threshold that won)."""
    best = {"avg_R": -np.inf}
    for thr in THRESHOLDS:
        sub = tr[tr.liq_ratio >= thr]
        if len(sub) < 50:
            continue
        net = sub.gross_ret - cost_bps / 1e4
        r = (net / sub.risk_frac).mean()
        if r > best["avg_R"]:
            pf_num = net[net > 0].sum()
            pf_den = -net[net < 0].sum()
            best = {"avg_R": r, "thr": thr, "n": len(sub),
                    "win%": round(100 * (net > 0).mean(), 1),
                    "PF": round(pf_num / pf_den, 2) if pf_den > 0 else np.inf}
    return best


WICK_PRESETS = {
    "default": dict(wick_body_mult=2.0, wick_min_frac=0.5, opp_wick_max_frac=0.10),
    "loose":   dict(wick_body_mult=1.5, wick_min_frac=0.4, opp_wick_max_frac=0.15),
    "strict":  dict(wick_body_mult=3.0, wick_min_frac=0.6, opp_wick_max_frac=0.05),
}
TRAILS = [0.5, 0.7]
BOXES = [15, 24]          # 75 vs 120 minutes
STOP_MULTS = [1.0, 1.5, 2.0]


def main() -> None:
    print("Preloading name-days...", flush=True)
    days = preload()
    print(f"{len(days)} usable name-days in memory.\n", flush=True)

    recs = []
    combos = list(itertools.product(WICK_PRESETS, TRAILS, BOXES, STOP_MULTS))
    for k, (wick, trail, box, sm) in enumerate(combos, 1):
        p = Params(trail_frac=trail, box_bars=box, stop_mult=sm, **WICK_PRESETS[wick])
        tr = run_combo(days, p)
        b0 = best_net(tr, 0.0)
        b5 = best_net(tr, 5.0)
        recs.append({
            "wick": wick, "trail": trail, "box": box, "stop_mult": sm,
            "n_trades": len(tr),
            "bestR_0bps": round(b0["avg_R"], 3), "bestR_5bps": round(b5["avg_R"], 3),
            "PF_5bps": b5.get("PF"), "win%_5bps": b5.get("win%"), "thr*_5bps": b5.get("thr"),
        })
        print(f"[{k}/{len(combos)}] {wick:7s} trail={trail} box={box} sm={sm} "
              f"n={len(tr):5d} bestR@0={b0['avg_R']:+.3f} bestR@5bps={b5['avg_R']:+.3f}",
              flush=True)

    res = pd.DataFrame(recs)
    res.to_csv(HERE / "sweep_results.csv", index=False)
    print("\n=== TOP 10 by net-of-5bps expectancy (best over thresholds) ===")
    print(res.sort_values("bestR_5bps", ascending=False).head(10).to_string(index=False))
    print("\n=== TOP 10 by gross (0 bps) expectancy ===")
    print(res.sort_values("bestR_0bps", ascending=False).head(10).to_string(index=False))
    pos5 = (res.bestR_5bps > 0).sum()
    pos0 = (res.bestR_0bps > 0).sum()
    print(f"\ncombos positive net of 5bps: {pos5}/{len(res)}   "
          f"| positive even at 0bps: {pos0}/{len(res)}")
    print("Saved sweep_results.csv")


if __name__ == "__main__":
    main()
