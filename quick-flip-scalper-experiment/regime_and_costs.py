"""Two follow-ups on the opening-range scalper, both as post-processing over the
already-cached trade frames (no re-simulation):

  (3) FUTURES-COST re-run - the headline verdict charged 5 bps round-trip, right
      for trading 500 cash equities. Someone trading the *index future* (ES)
      pays ~0.5-1 bp. Does the faint gross edge survive at futures costs?

  (1) VOLATILITY-REGIME gate - use the one thing the programme showed is
      forecastable (volatility persistence) to decide *fade vs follow* per day.
      Pre-registered, causal regime = yesterday's true range / its trailing
      ATR14 (`vol_state`, known before today's open). Hypothesis: expansion
      regimes trend (take the breakout), compression regimes revert (take the
      fade). Any positive book is then run through the SAME stress tests that
      killed the original breakout (day-block bootstrap, concentration, beta).

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u regime_and_costs.py

One pass over the 5-min universe builds both the causal regime table and the
market-return (beta) map; everything else is arithmetic on cached trades.
"""
from __future__ import annotations

from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "candle-data" / "data"
NY = "America/New_York"
OPEN = time(9, 30)
CLOSE = time(16, 0)
C1_BARS = 3
BOX_BARS = 15

COSTS_BPS = [0.0, 0.5, 1.0, 2.0, 5.0]     # round-trip; 0.5-1 = futures, 5 = cash equities
THRESHOLDS = [0.0, 0.5, 0.75]
VOL_THR = 1.0                              # pre-registered expansion/compression split
VOL_THR_SENS = [0.8, 1.0, 1.2]            # robustness only
N_BOOT = 5000
RNG = np.random.default_rng(20260907)


# --------------------------------------------------------------------------- #
# one pass: causal regime signal + market morning-return (beta) map
# --------------------------------------------------------------------------- #
def preload_universe() -> tuple[pd.DataFrame, dict]:
    vs_rows, mkt_rows = [], []
    tickers = sorted(p.stem for p in DATA.glob("*.parquet") if not p.stem.startswith("_"))
    for t in tickers:
        df = pd.read_parquet(DATA / f"{t}.parquet").tz_convert(NY).sort_index()
        df["d"] = df.index.date
        daily = df.groupby("d").agg(hi=("high", "max"), lo=("low", "min"), cl=("close", "last"))
        prev = daily.cl.shift(1)
        tr = np.maximum(daily.hi - daily.lo,
                        np.maximum((daily.hi - prev).abs(), (daily.lo - prev).abs()))
        atr = tr.rolling(14).mean()
        vs = (tr / atr).shift(1)               # day d value = TR_{d-1}/ATR_{d-1} (causal)
        for d, v in vs.items():
            if np.isfinite(v):
                vs_rows.append((t, d, float(v)))
        # morning market return: C1 close (bar index 2) -> box end, per name-day
        for d, day in df.groupby("d"):
            day = day[(day.index.time >= OPEN) & (day.index.time < CLOSE)].sort_index()
            if len(day) < C1_BARS + 2:
                continue
            be = min(len(day) - 1, C1_BARS + BOX_BARS - 1)
            start = day.iloc[C1_BARS - 1].close
            if start > 0:
                mkt_rows.append((d, day.iloc[be].close / start - 1.0))
    vs_df = pd.DataFrame(vs_rows, columns=["ticker", "date", "vol_state"])
    mkt = pd.DataFrame(mkt_rows, columns=["date", "r"]).groupby("date")["r"].mean().to_dict()
    return vs_df, mkt


# --------------------------------------------------------------------------- #
def metrics(tr: pd.DataFrame, cost_bps: float) -> dict:
    if len(tr) == 0:
        return {"n": 0, "avg_R": float("nan")}
    net = tr.gross_ret - cost_bps / 1e4
    r = net / tr.risk_frac
    pf_num, pf_den = net[net > 0].sum(), -net[net < 0].sum()
    return {"n": len(tr), "win%": round(100 * (net > 0).mean(), 1),
            "avg_bps": round(1e4 * net.mean(), 1), "avg_R": round(r.mean(), 3),
            "PF": round(pf_num / pf_den, 2) if pf_den > 0 else np.inf}


def sweep_costs(tr: pd.DataFrame, label: str) -> None:
    recs = [{"thr": thr, "cost_bps": c, **metrics(tr[tr.liq_ratio >= thr], c)}
            for thr in THRESHOLDS for c in COSTS_BPS]
    print(f"\n--- {label} (n={len(tr)}) ---", flush=True)
    print(pd.DataFrame(recs).to_string(index=False), flush=True)


def day_block_boot(tr: pd.DataFrame, value: pd.Series) -> tuple[float, float, float]:
    by_date = {d: value[tr.date == d].to_numpy() for d in tr.date.unique()}
    dates = list(by_date)
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = RNG.choice(len(dates), size=len(dates), replace=True)
        means[i] = np.concatenate([by_date[dates[j]] for j in pick]).mean()
    return float(value.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stress_book(tr: pd.DataFrame, cost_bps: float, mkt: dict, name: str) -> None:
    cost = cost_bps / 1e4
    tr = tr.copy().reset_index(drop=True)
    tr["net_R"] = (tr.gross_ret - cost) / tr.risk_frac
    print(f"\n===== STRESS: {name}  (cost {cost_bps} bps, n={len(tr)}, "
          f"tickers={tr.ticker.nunique()}, dates={tr.date.nunique()}) =====", flush=True)

    m, lo, hi = day_block_boot(tr, tr.net_R)
    star = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
    print(f"  day-block bootstrap: mean {m:+.4f}R  95% CI [{lo:+.4f}, {hi:+.4f}]{star}")

    by_dt = tr.groupby("date").net_R.sum().sort_values()
    total = tr.net_R.sum()
    if abs(total) > 1e-9:
        top3 = by_dt.tail(3).index
        rest = tr[~tr.date.isin(top3)]
        print(f"  concentration: net-pos days {(by_dt>0).sum()} vs neg {(by_dt<0).sum()}; "
              f"top-3 days = {by_dt.tail(3).sum()/total*100:.0f}% of total; "
              f"ex-top3 mean {rest.net_R.mean():+.4f}R")

    sign = np.where(tr.side.to_numpy() == "long", 1.0, -1.0)
    mret = tr.date.map(mkt).to_numpy()
    tr["alpha_R"] = (tr.gross_ret.to_numpy() - sign * mret - cost) / tr.risk_frac.to_numpy()
    am, alo, ahi = day_block_boot(tr, tr.alpha_R)
    astar = "  <-- CI excludes 0" if (alo > 0 or ahi < 0) else ""
    print(f"  beta-neutral alpha:  mean {am:+.4f}R  95% CI [{alo:+.4f}, {ahi:+.4f}]{astar}")


# --------------------------------------------------------------------------- #
def main():
    fade = pd.read_parquet(HERE / "trades.parquet")
    follow = pd.read_parquet(HERE / "orb_trades_box_end.parquet")
    if "traded" in follow:
        follow = follow[follow.traded].copy()

    print("#" * 72)
    print("# (3) FUTURES-COST RE-RUN")
    print("#" * 72, flush=True)
    sweep_costs(fade, "FADE  (quick-flip)")
    sweep_costs(follow, "FOLLOW (breakout, box-end)")

    print("\nBuilding causal regime + beta map (one universe pass)...", flush=True)
    vs, mkt = preload_universe()

    print("\n" + "#" * 72)
    print("# (1) VOLATILITY-REGIME GATE  (vol_state = yesterday TR / trailing ATR14)")
    print("#" * 72, flush=True)
    fade = fade.merge(vs, on=["ticker", "date"], how="left")
    follow = follow.merge(vs, on=["ticker", "date"], how="left")
    fade_r = fade.dropna(subset=["vol_state"])
    follow_r = follow.dropna(subset=["vol_state"])
    print(f"\nregime coverage: fade {len(fade_r)}/{len(fade)}, "
          f"follow {len(follow_r)}/{len(follow)} trades have a causal vol_state", flush=True)

    print("\n--- discriminator check: mean net_R by regime ---", flush=True)
    for nm, book in [("FOLLOW", follow_r), ("FADE", fade_r)]:
        hi_ = book[book.vol_state >= VOL_THR]
        lo_ = book[book.vol_state < VOL_THR]
        for cost in (1.0, 5.0):
            print(f"  {nm:6s} cost {cost:>3.0f}bps | "
                  f"expansion(vs>={VOL_THR}) n={len(hi_):5d} {metrics(hi_,cost)['avg_R']:+.3f}R | "
                  f"compression(vs<{VOL_THR}) n={len(lo_):5d} {metrics(lo_,cost)['avg_R']:+.3f}R")

    print("\n--- GATED book: follow if vol_state>=thr, else fade ---", flush=True)
    for thr in VOL_THR_SENS:
        g = pd.concat([follow_r[follow_r.vol_state >= thr],
                       fade_r[fade_r.vol_state < thr]], ignore_index=True)
        recs = [{"vol_thr": thr, "cost_bps": c, **metrics(g, c)} for c in COSTS_BPS]
        print(pd.DataFrame(recs).to_string(index=False))

    gated = pd.concat([follow_r[follow_r.vol_state >= VOL_THR],
                       fade_r[fade_r.vol_state < VOL_THR]], ignore_index=True)
    ran_any = False
    for cost in (0.5, 1.0, 5.0):
        if metrics(gated, cost)["avg_R"] > 0:
            stress_book(gated, cost, mkt, f"gated (vol_thr={VOL_THR})")
            ran_any = True
    if not ran_any:
        print("\ngated book is non-positive at every cost tested -> no stress run needed.", flush=True)


if __name__ == "__main__":
    main()
