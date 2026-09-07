"""Breakout EXTENT + VOLUME + line-duration alignment: do they predict continuation?

Lance's hypothesis: a strong breakout (large extent), on high volume, in the same
direction as the longer trend line, is more likely to CONTINUE (bigger payoff)
rather than revert. The real prize is payoff, which no feature has predicted yet.

Features (all causal, at the breakout):
  break_extent  - how far the signal bar cleared the action ray, in ATR units (new).
  rel_volume    - breakout-bar volume / trailing 20-bar median.
  aligned       - is the trade in the same direction as the LONGER of the two lines?
  rel_age_from_a- relative duration of action vs safety line (from the last study).

Guarded IS/OOS. We test each vs payoff (gross) and win, an IS-standardised combined
score, and a triple-gate book (strong + high-volume + aligned) through the stress
harness.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u continuation_study.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import load_ftse
from backtest import CostModel, simulate
from backtest_run import UNIVERSE
from line_duration_study import add_features
from regime_gate_ftse import (OOS_YEAR, add_costs, market_over_hold, metrics,
                              per_ticker_totals, vol_state, year_block_boot)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "continuation_trades.parquet"
OUT = HERE / "continuation_study_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def collect() -> tuple[pd.DataFrame, pd.Series, dict]:
    dfs = {t: load_ftse.load(t) for t in UNIVERSE}
    closes = pd.DataFrame({t: d["close"] for t, d in dfs.items()}).sort_index()
    mkt_idx = (1 + closes.pct_change().mean(axis=1)).cumprod()
    bh = {t: float(d["close"].iloc[-1] / d["close"].iloc[0] - 1) for t, d in dfs.items()}
    if CACHE.exists():
        return pd.read_parquet(CACHE), mkt_idx, bh
    rows = []
    for t, df in dfs.items():
        vs = vol_state(df)
        for tr in simulate(df, k=0.5, min_span=0, costs=CostModel(0.0, 0.0)):
            rows.append({
                "ticker": t, "entry": tr.entry_date, "exit": tr.exit_date,
                "direction": tr.direction, "entry_px": tr.entry_price,
                "exit_px": tr.exit_price, "days": tr.days, "gross": tr.ret,
                "year": tr.entry_date.year, "signal_bar": tr.signal_bar,
                "action_a": tr.action_a, "action_b": tr.action_b,
                "safety_a": tr.safety_a, "safety_b": tr.safety_b,
                "action_touches": tr.action_touches, "action_touch_first": tr.action_touch_first,
                "action_touch_last": tr.action_touch_last, "safety_touches": tr.safety_touches,
                "safety_touch_first": tr.safety_touch_first, "safety_touch_last": tr.safety_touch_last,
                "action_slope": tr.action_slope, "safety_slope": tr.safety_slope,
                "break_extent": tr.break_extent, "rel_volume": tr.rel_volume,
                "vol_state": float(vs.get(tr.entry_date, np.nan)),
            })
    tr = pd.DataFrame(rows).sort_values("entry").reset_index(drop=True)
    tr.to_parquet(CACHE)
    return tr, mkt_idx, bh


def sp(part: pd.DataFrame, f: str, y: str) -> tuple[float, float, int]:
    m = part[[f, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(m) < 30 or m[f].nunique() < 3:
        return np.nan, np.nan, len(m)
    r, p = stats.spearmanr(m[f], m[y])
    return r, p, len(m)


def corr_row(ins, oos, f: str) -> None:
    ri, pi, _ = sp(ins, f, "gross")
    ro, po, no = sp(oos, f, "gross")
    rw, _, _ = sp(oos, f, "win")
    say(f"{f:14} {ri:>+9.3f}(p{pi:4.2f}) {ro:>+9.3f}(p{po:4.2f}) {rw:>+12.3f}   n_OOS={no}")


def deciles(oos: pd.DataFrame, feat: str) -> None:
    o = oos[[feat, "gross", "net_10", "win"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(o) < 200:
        say(f"  (too few rows for {feat})"); return
    o["dec"] = pd.qcut(o[feat], 10, labels=False, duplicates="drop")
    say(f"\nOOS payoff by {feat} decile (10 = strongest):")
    say(f"{'dec':>4} {'n':>5} {'win%':>6} {'gross/tr%':>10} {'net/tr%':>9} {'med':>8}")
    for d, g in o.groupby("dec"):
        say(f"{int(d)+1:>4} {len(g):>5} {100*g.win.mean():>6.1f} "
            f"{100*g.gross.mean():>10.3f} {100*g.net_10.mean():>9.3f} {g[feat].median():>8.2f}")


def stress(tr: pd.DataFrame, mkt_idx, bh, name: str) -> None:
    tr = tr.copy().reset_index(drop=True)
    if len(tr) < 50:
        say(f"\n===== STRESS: {name} -- only {len(tr)} trades, skipped ====="); return
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
    say("Collecting continuation-instrumented trades (20 names)...")
    tr, mkt_idx, bh = collect()
    tr = add_features(add_costs(tr))
    # aligned: trade in the same direction as the LONGER (by age_from_a) line's slope
    longer_action = (tr.age_from_a_A >= tr.age_from_a_S).to_numpy()
    longer_slope = np.where(longer_action, tr.action_slope.to_numpy(), tr.safety_slope.to_numpy())
    tsign = np.where(tr.direction.to_numpy() == "LONG", 1.0, -1.0)
    tr["aligned"] = ((tsign * np.sign(longer_slope)) > 0).astype(float)
    tr["align_slope"] = tsign * longer_slope
    tr = tr.sort_values("entry").reset_index(drop=True)

    mid = len(tr) // 2
    ins, oos = tr.iloc[:mid], tr.iloc[mid:]
    say(f"{len(tr)} trades. break_extent coverage {100*tr.break_extent.notna().mean():.0f}%, "
        f"rel_volume {100*tr.rel_volume.notna().mean():.0f}%.  aligned rate {100*tr.aligned.mean():.0f}%.")
    say(f"OOS base: win {100*oos.win.mean():.1f}%  gross/tr {100*oos.gross.mean():+.3f}%")

    say("\n" + "#" * 72)
    say("# Spearman vs PAYOFF (gross) and win, IS -> OOS")
    say("#" * 72)
    say(f"{'feature':14} {'IS rho(gross)':>15} {'OOS rho(gross)':>15} {'OOS rho(win)':>13}")
    for f in ["break_extent", "rel_volume", "align_slope", "aligned", "rel_age_from_a"]:
        corr_row(ins, oos, f)

    for f in ["break_extent", "rel_volume"]:
        deciles(oos, f)

    # IS-standardised combined score (no look-ahead): strong + high-volume (+ aligned tilt)
    say("\n" + "#" * 72)
    say("# Combined score = z(break_extent) + z(rel_volume), IS-standardised")
    say("#" * 72)
    feats = ["break_extent", "rel_volume"]
    mu = ins[feats].mean(); sd = ins[feats].std() + 1e-9
    for part in (ins, oos):
        z = ((part[feats] - mu) / sd)
        part.loc[:, "score"] = z["break_extent"] + z["rel_volume"]
    deciles(oos, "score")
    ro, po, _ = sp(oos, "score", "gross")
    say(f"\nOOS Spearman(score, gross) = {ro:+.3f} (p{po:4.2f})")

    # triple gate: strong break AND high volume AND aligned (full sample, then stress)
    say("\n" + "#" * 72)
    say("# TRIPLE GATE: break_extent top third AND rel_volume top third AND aligned")
    say("#" * 72)
    be_thr = tr.break_extent.quantile(2/3)
    rv_thr = tr.rel_volume.quantile(2/3)
    gate = tr[(tr.break_extent >= be_thr) & (tr.rel_volume >= rv_thr) & (tr.aligned == 1)]
    say(f"  gross {metrics(gate,'gross')}")
    say(f"  net10 {metrics(gate,'net_10')}")
    # also without the alignment condition, for contrast
    g2 = tr[(tr.break_extent >= be_thr) & (tr.rel_volume >= rv_thr)]
    say(f"  [strong+high-vol only, no alignment] gross {metrics(g2,'gross')}  |  net10 {metrics(g2,'net_10')}")
    stress(gate, mkt_idx, bh, "triple gate (strong + high-vol + aligned)")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
