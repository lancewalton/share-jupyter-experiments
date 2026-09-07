"""12-1 cross-sectional momentum on FTSE, full rigour: monthly rebalance, turnover
costs (spread-bet), long-only top-quintile tilt AND market-neutral long-short,
OOS split, rolling read (is the 2024-26 revival real?), beta/alpha, drawdown, and a
year-block bootstrap CI.

Signal at month-end t: return from t-12 to t-1 (skip the most recent month). Hold
the next month. Quintiles of the ~120-name universe.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u momentum_backtest.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import load_ftse

HERE = Path(__file__).resolve().parent
SPREAD_BPS = [0.0, 10.0, 20.0]     # per side
FRAC = 0.2
SPLIT = pd.Timestamp("2013-01-01")
N_BOOT = 5000
RNG = np.random.default_rng(20260907)
OUT = HERE / "momentum_backtest_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def build_portfolios(M: pd.DataFrame):
    """Monthly returns of long-only top quintile, long-short (Q5-Q1), universe, and
    the one-way turnover of each leg, all aligned to holding months."""
    mret = M.pct_change()
    sig = M.shift(1) / M.shift(12) - 1        # 12-1: skip most recent month
    months = M.index
    lo_ret, ls_ret, uni_ret = [], [], []
    lo_turn, ls_turn, idx = [], [], []
    w_long_prev = pd.Series(0.0, index=M.columns)
    w_short_prev = pd.Series(0.0, index=M.columns)
    for i in range(1, len(months)):
        t = months[i - 1]                      # formation month-end
        s = sig.loc[t].dropna()
        r = mret.loc[months[i]].reindex(M.columns)   # holding-month return
        if len(s) < 20 or r.notna().sum() < 20:
            continue
        k = max(1, int(len(s) * FRAC))
        top = s.nlargest(k).index
        bot = s.nsmallest(k).index
        w_long = pd.Series(0.0, index=M.columns); w_long[top] = 1.0 / k
        w_short = pd.Series(0.0, index=M.columns); w_short[bot] = 1.0 / k
        lo_ret.append(float(r[top].mean()))
        ls_ret.append(float(r[top].mean() - r[bot].mean()))
        uni_ret.append(float(r.mean()))
        lo_turn.append(float((w_long - w_long_prev).abs().sum()))
        ls_turn.append(float((w_long - w_long_prev).abs().sum() + (w_short - w_short_prev).abs().sum()))
        w_long_prev, w_short_prev = w_long, w_short
        idx.append(months[i])
    return pd.DataFrame({"lo": lo_ret, "ls": ls_ret, "uni": uni_ret,
                         "lo_turn": lo_turn, "ls_turn": ls_turn}, index=pd.DatetimeIndex(idx))


def metrics(r: pd.Series, uni: pd.Series | None = None) -> dict:
    r = r.dropna()
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    sh = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0.0
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    out = {"CAGR": ann, "vol": vol, "Sharpe": sh, "maxDD": mdd}
    if uni is not None:
        u = uni.reindex(r.index)
        b = np.cov(r, u)[0, 1] / np.var(u)
        out["beta"] = b
        out["alpha"] = r.mean() * 12 - b * u.mean() * 12
    return out


def yb_boot(r: pd.Series) -> tuple[float, float, float]:
    by = {y: r[r.index.year == y].to_numpy() for y in r.index.year.unique()}
    ys = [y for y in by if len(by[y])]
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = RNG.choice(len(ys), len(ys), replace=True)
        means[i] = np.concatenate([by[ys[j]] for j in pick]).mean() * 12
    return float(r.mean() * 12), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def net(P: pd.DataFrame, leg: str, turn: str, spread: float) -> pd.Series:
    return P[leg] - P[turn] * spread / 1e4


def main() -> None:
    names = load_ftse.universe()
    say(f"Loading {len(names)} FTSE names (monthly)...")
    closes = {}
    for t in names:
        try:
            closes[t] = load_ftse.load(t)["close"]
        except Exception:
            pass
    daily = pd.DataFrame(closes).sort_index()
    M = daily.resample("ME").last()
    P = build_portfolios(M)
    say(f"{len(closes)} names, {len(P)} holding months {P.index.min().date()}->{P.index.max().date()}.\n")

    uni = P["uni"]
    say("=== Universe equal-weight (monthly) ===")
    m = metrics(uni)
    say(f"  CAGR {100*m['CAGR']:+.2f}%  vol {100*m['vol']:.1f}%  Sharpe {m['Sharpe']:.2f}  maxDD {100*m['maxDD']:.0f}%")

    say("\n=== 12-1 momentum, net of costs (turnover-based) ===")
    say(f"{'book':>10} {'cost':>5} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7} {'beta':>6} {'alpha':>7} {'turn/yr':>8}")
    for spread in SPREAD_BPS:
        for leg, turn, lab in [("lo", "lo_turn", "long-only"), ("ls", "ls_turn", "long-short")]:
            r = net(P, leg, turn, spread)
            mm = metrics(r, uni if leg == "lo" else None)
            turnyr = P[turn].mean() * 12
            beta = mm.get("beta", 0.0); alpha = mm.get("alpha", float("nan"))
            say(f"{lab:>10} {spread:>4.0f}b {100*mm['CAGR']:>+6.2f}% {mm['Sharpe']:>7.2f} "
                f"{100*mm['maxDD']:>6.0f}% {beta:>6.2f} {100*alpha:>+6.2f}% {turnyr:>7.1f}x")

    # focus on realistic 10bps
    say("\n=== At 10 bps/side ===")
    lo10, ls10 = net(P, "lo", "lo_turn", 10), net(P, "ls", "ls_turn", 10)
    for lab, r, u in [("long-only tilt", lo10, uni), ("long-short (neutral)", ls10, None)]:
        pre, post = r[r.index < SPLIT], r[r.index >= SPLIT]
        rc = r[r.index >= pd.Timestamp("2023-01-01")]
        mm, mpre, mpost = metrics(r, u), metrics(pre), metrics(post)
        b, lo_ci, hi_ci = yb_boot(r)
        star = "  CI>0" if lo_ci > 0 else ("  CI<0" if hi_ci < 0 else "  CI spans 0")
        say(f"\n  {lab}: full CAGR {100*mm['CAGR']:+.2f}% Sharpe {mm['Sharpe']:.2f}")
        say(f"    OOS: pre-2013 CAGR {100*mpre['CAGR']:+.2f}% Sh {mpre['Sharpe']:.2f} | "
            f"post-2013 {100*mpost['CAGR']:+.2f}% Sh {mpost['Sharpe']:.2f}")
        say(f"    2023-26: CAGR {100*metrics(rc)['CAGR']:+.2f}%  Sharpe {metrics(rc)['Sharpe']:.2f} (n={len(rc)}m)")
        say(f"    year-block bootstrap ann mean {100*b:+.2f}% 95% CI [{100*lo_ci:+.2f}%, {100*hi_ci:+.2f}%]{star}")

    # rolling 3y (36m) annualised return of the market-neutral book, year-ends
    say("\n=== Long-short (10bps) rolling 36m annualised return ===")
    roll = (1 + ls10).rolling(36).apply(lambda x: x.prod() ** (12 / 36) - 1, raw=True)
    say(f"{'year':>6} {'roll36_ann':>11}")
    for yr in range(2005, 2027):
        sub = ls10.index[ls10.index.year == yr]
        if len(sub) and np.isfinite(roll.get(sub[-1], np.nan)):
            say(f"{yr:>6} {100*roll[sub[-1]]:>+10.2f}%")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
