"""Does a causal VOLATILITY-REGIME gate rescue the daily FTSE trend-line breakout?

The intraday scalper showed "follow breakouts in expansion regimes" is
directionally right but not significant on one 60-day window. Here we test the
SAME mechanism on ~27 years of daily FTSE (20 liquid names) - many regimes,
which is exactly what the intraday test lacked.

Method (pre-registered, honest):
  - Re-run the exact breakout (k=0.5, min_span=0) to get every trade with its
    direction and hold, gross return, and net (10 bps/side + 5%/yr financing).
  - Regime signal, causal: vol_state = 20-day realised vol / 100-day realised
    vol, shifted so day d uses only data through d-1. >=1 = expansion.
  - Gate: keep breakout trades in expansion; report the expansion-vs-compression
    discriminator.
  - Stress any positive book with the tests that killed the intraday breakout:
    year-block bootstrap CI, concentration by name & year, beta-neutral alpha
    (subtract the equal-weight market move over the hold, in the trade's
    direction), a pre/post-2013 out-of-sample split, and the buy-and-hold
    benchmark.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u regime_gate_ftse.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import load_ftse
from backtest import CostModel, simulate, trade_return
from backtest_run import UNIVERSE

HERE = Path(__file__).resolve().parent
VOL_THR = 1.0
VOL_THR_SENS = [0.8, 1.0, 1.2]
OOS_YEAR = 2013
COST_SWEEP = [CostModel(0, 0.0), CostModel(10, 0.05), CostModel(20, 0.05)]
REALISTIC = CostModel(10, 0.05)
N_BOOT = 5000
RNG = np.random.default_rng(20260907)
OUT = HERE / "regime_gate_ftse_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


# --------------------------------------------------------------------------- #
def vol_state(df: pd.DataFrame) -> pd.Series:
    """Causal 20d/100d realised-vol ratio, indexed by date (known at entry)."""
    r = np.log(df["close"]).diff()
    rv20 = r.rolling(20).std()
    rv100 = r.rolling(100).std()
    return (rv20 / rv100).shift(1)


TRADE_CACHE = HERE / "regime_trades.parquet"


def build_trades() -> tuple[pd.DataFrame, pd.Series, dict]:
    dfs = {t: load_ftse.load(t) for t in UNIVERSE}
    # equal-weight market proxy: mean daily simple return across the universe
    closes = pd.DataFrame({t: d["close"] for t, d in dfs.items()}).sort_index()
    mkt_idx = (1 + closes.pct_change().mean(axis=1)).cumprod()
    bh = {t: float(d["close"].iloc[-1] / d["close"].iloc[0] - 1) for t, d in dfs.items()}

    if TRADE_CACHE.exists():
        return pd.read_parquet(TRADE_CACHE), mkt_idx, bh

    rows = []
    for t, df in dfs.items():
        vs = vol_state(df)
        for tr in simulate(df, k=0.5, min_span=0, costs=CostModel(0.0, 0.0)):
            rows.append({
                "ticker": t, "entry": tr.entry_date, "exit": tr.exit_date,
                "direction": tr.direction, "entry_px": tr.entry_price,
                "exit_px": tr.exit_price, "days": tr.days, "gross": tr.ret,
                "vol_state": float(vs.get(tr.entry_date, np.nan)),
                "year": tr.entry_date.year,
            })
    trades = pd.DataFrame(rows).sort_values("entry").reset_index(drop=True)
    trades.to_parquet(TRADE_CACHE)
    return trades, mkt_idx, bh


def add_costs(tr: pd.DataFrame) -> pd.DataFrame:
    for cm in COST_SWEEP:
        col = f"net_{int(cm.spread_bps_per_side)}"
        tr[col] = [trade_return(d, e, x, dd, cm)
                   for d, e, x, dd in zip(tr.direction, tr.entry_px, tr.exit_px, tr.days)]
    return tr


def market_over_hold(tr: pd.DataFrame, mkt_idx: pd.Series) -> np.ndarray:
    lookup = mkt_idx.ffill().to_dict()   # dict map tolerates duplicate entry dates
    e = tr.entry.map(lookup).to_numpy(dtype=float)
    x = tr.exit.map(lookup).to_numpy(dtype=float)
    return x / e - 1.0


def metrics(tr: pd.DataFrame, col: str) -> str:
    if len(tr) == 0:
        return "n=0"
    v = tr[col]
    pf_num, pf_den = v[v > 0].sum(), -v[v < 0].sum()
    pf = pf_num / pf_den if pf_den > 0 else np.inf
    return (f"n={len(tr):5d}  win={100*(v>0).mean():4.1f}%  "
            f"mean/tr={100*v.mean():+6.3f}%  PF={pf:4.2f}")


def year_block_boot(tr: pd.DataFrame, col: str) -> tuple[float, float, float]:
    by_year = {y: tr[tr.year == y][col].to_numpy() for y in tr.year.unique()}
    years = [y for y in by_year if len(by_year[y])]
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = RNG.choice(len(years), size=len(years), replace=True)
        means[i] = np.concatenate([by_year[years[j]] for j in pick]).mean()
    return float(tr[col].mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_ticker_totals(tr: pd.DataFrame, col: str, bh: dict) -> str:
    tot = tr.groupby("ticker")[col].apply(lambda s: np.prod(1 + s.to_numpy()) - 1)
    prof = (tot > 0).mean()
    med_bh = np.median([bh[t] for t in tot.index])
    return (f"per-ticker compound (net): median {100*tot.median():+.1f}%  "
            f"%profitable {100*prof:.0f}%  |  median buy&hold {100*med_bh:+.1f}%")


def stress(tr: pd.DataFrame, mkt_idx: pd.Series, bh: dict, name: str) -> None:
    tr = tr.copy().reset_index(drop=True)
    say(f"\n===== STRESS: {name}  (n={len(tr)}, names={tr.ticker.nunique()}, "
        f"years {tr.year.min()}-{tr.year.max()}) =====")

    for col, lab in [("gross", "gross"), ("net_10", "net(10bps+5%)")]:
        m, lo, hi = year_block_boot(tr, col)
        star = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
        say(f"  year-block bootstrap {lab:16s}: mean/tr {100*m:+.3f}%  "
            f"95% CI [{100*lo:+.3f}%, {100*hi:+.3f}%]{star}")

    # beta-neutral alpha (gross minus market move over the hold in trade direction)
    sign = np.where(tr.direction.to_numpy() == "LONG", 1.0, -1.0)
    tr["alpha"] = tr.gross.to_numpy() - sign * market_over_hold(tr, mkt_idx)
    m, lo, hi = year_block_boot(tr, "alpha")
    star = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
    say(f"  beta-neutral alpha (gross)      : mean/tr {100*m:+.3f}%  "
        f"95% CI [{100*lo:+.3f}%, {100*hi:+.3f}%]{star}")

    # concentration
    by_tk = tr.groupby("ticker").net_10.sum().sort_values()
    by_yr = tr.groupby("year").net_10.sum().sort_values()
    total = tr.net_10.sum()
    if abs(total) > 1e-9:
        say(f"  concentration (net): names +{(by_tk>0).sum()}/-{(by_tk<0).sum()}, "
            f"top-3 names {by_tk.tail(3).sum()/total*100:.0f}% of total; "
            f"years +{(by_yr>0).sum()}/-{(by_yr<0).sum()}, "
            f"top-3 years {by_yr.tail(3).sum()/total*100:.0f}%")

    # out-of-sample split
    pre, post = tr[tr.year < OOS_YEAR], tr[tr.year >= OOS_YEAR]
    say(f"  OOS split @ {OOS_YEAR}: pre  gross {metrics(pre,'gross')}")
    say(f"  {'':16s}      pre  net10 {metrics(pre,'net_10')}")
    say(f"  {'':16s}      post gross {metrics(post,'gross')}")
    say(f"  {'':16s}      post net10 {metrics(post,'net_10')}")
    say("  " + per_ticker_totals(tr, "net_10", bh))


# --------------------------------------------------------------------------- #
def main() -> None:
    say("Re-running the breakout to capture direction + hold (20 names)...")
    trades, mkt_idx, bh = build_trades()
    trades = add_costs(trades)
    cov = trades.vol_state.notna()
    say(f"{len(trades)} trades; {cov.sum()} have a causal vol_state "
        f"({100*cov.mean():.0f}% coverage).")
    tr = trades[cov].copy()

    say("\n" + "#" * 68)
    say("# DISCRIMINATOR: expansion vs compression (does the regime steer?)")
    say("#" * 68)
    hi_ = tr[tr.vol_state >= VOL_THR]
    lo_ = tr[tr.vol_state < VOL_THR]
    for col, lab in [("gross", "gross"), ("net_10", "net(10bps+5%)")]:
        say(f"  {lab:16s} | expansion(vs>={VOL_THR}) {metrics(hi_,col)}")
        say(f"  {'':16s} | compression(vs<{VOL_THR}) {metrics(lo_,col)}")

    say("\n" + "#" * 68)
    say("# GATED BOOK: keep breakout trades in expansion (sensitivity on thr)")
    say("#" * 68)
    for thr in VOL_THR_SENS:
        g = tr[tr.vol_state >= thr]
        say(f"  vol_thr={thr}:  gross {metrics(g,'gross')}")
        say(f"  {'':12s}  net10 {metrics(g,'net_10')}")
        say(f"  {'':12s}  net20 {metrics(g,'net_20')}")

    # stress the pre-registered gated book, and the full (ungated) book for contrast
    stress(tr[tr.vol_state >= VOL_THR], mkt_idx, bh, f"GATED expansion (vol_thr={VOL_THR})")
    stress(tr, mkt_idx, bh, "FULL breakout book (ungated, for contrast)")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
