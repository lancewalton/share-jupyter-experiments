"""Breadth test: does the channel-SELECTION result replicate on US large caps?
Same method as FTSE (hold names currently in a qualifying rising channel, causal),
full period + OOS split @2013 + rolling edge and its correlation with market vol.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u us_selector.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from channel_entry_sweep import precompute_rich

HERE = Path(__file__).resolve().parent
BARS_YR = 252
L = 250
G_MIN, R2_MIN = 0.10, 0.80
SPREAD = 10 / 1e4
W = 756
SPLIT = pd.Timestamp("2013-01-01")
W_MINS = [0.0, 0.15, 0.25]
OUT = HERE / "us_selector_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def cagr(d: np.ndarray) -> float:
    if len(d) == 0:
        return float("nan")
    tot = np.prod(1 + d) - 1
    return (1 + tot) ** (1 / (len(d) / BARS_YR)) - 1 if tot > -1 else float("nan")


def sharpe(d: np.ndarray) -> float:
    return float(np.mean(d) / np.std(d) * np.sqrt(BARS_YR)) if np.std(d) > 0 else 0.0


def sel_daily(qual, RETS):
    held = qual.shift(1).fillna(False)
    n = held.sum(axis=1)
    mret = RETS.where(held).mean(axis=1).fillna(0.0)
    ch = held.astype(int).diff().abs().sum(axis=1)
    cost = (SPREAD * ch / n.replace(0, np.nan)).fillna(0.0)
    return (mret.to_numpy() - cost.to_numpy()).astype("float64"), n


def main() -> None:
    closes_df = pd.read_parquet(HERE / "us_closes.parquet").sort_index()
    dates = closes_df.index
    say(f"US: {closes_df.shape[1]} names, {dates.min().date()}->{dates.max().date()}. Precomputing L={L}...")
    ANN, RELW, R2 = {}, {}, {}
    for t in closes_df.columns:
        s = closes_df[t].dropna()
        a = precompute_rich(pd.DataFrame({"close": s}), L)
        if a is None:
            continue
        _, _, ann, relw, r2 = a
        ANN[t] = pd.Series(ann, index=s.index)
        RELW[t] = pd.Series(relw, index=s.index)
        R2[t] = pd.Series(r2, index=s.index)
    ANN = pd.DataFrame(ANN).reindex(dates); RELW = pd.DataFrame(RELW).reindex(dates); R2 = pd.DataFrame(R2).reindex(dates)
    RETS = closes_df.reindex(dates).pct_change()
    base = (ANN >= G_MIN) & (R2 >= R2_MIN)
    uni = RETS.mean(axis=1).fillna(0.0).to_numpy().astype("float64")
    pre, post = dates < SPLIT, dates >= SPLIT
    say(f"Universe EW B&H: CAGR {100*cagr(uni):+.2f}%  Sharpe {sharpe(uni):.2f}  "
        f"| pre {100*cagr(uni[pre]):+.2f}%  post {100*cagr(uni[post]):+.2f}%\n")

    say("#" * 84)
    say("# US CHANNEL-SELECTION (hold names in qualifying rising channel), L=250")
    say("#" * 84)
    say(f"{'w_min':>6} {'avgHeld':>8} {'CAGR':>7} {'Sharpe':>7} {'pre13':>7} {'post13':>7}  vsUniv")
    rep = None
    for w in W_MINS:
        qual = (base & (RELW >= w)).fillna(False)
        d, n = sel_daily(qual, RETS)
        beat = "WINS" if cagr(d) > cagr(uni) else "loses"
        say(f"{w:>6.2f} {n.mean():>8.0f} {100*cagr(d):>6.2f}% {sharpe(d):>7.2f} "
            f"{100*cagr(d[pre]):>6.2f}% {100*cagr(d[post]):>6.2f}%  {beat}")
        if w == 0.0:
            rep = (qual, d)

    # rolling edge + vol correlation for w_min=0
    qual, d = rep
    sel = pd.Series(d, index=dates)
    uni_s = pd.Series(uni, index=dates)
    def roll_ann(x):
        return np.expm1(np.log1p(x.astype("float64")).rolling(W).sum() * (BARS_YR / W))
    edge = roll_ann(sel) - roll_ann(uni_s)
    vol = uni_s.rolling(W).std() * np.sqrt(BARS_YR)
    m = pd.DataFrame({"edge": edge, "vol": vol}).dropna()
    say(f"\nrolling 3y edge (w_min=0): corr(edge, mkt vol) = {m.edge.corr(m.vol):+.2f}")
    prem = m[m.index < SPLIT]; postm = m[m.index >= SPLIT]
    say(f"  mean edge pre-2013 {100*prem.edge.mean():+.1f}%  post-2013 {100*postm.edge.mean():+.1f}%")
    say(f"{'year':>6} {'edge':>7}")
    for yr in range(2003, 2027):
        sub = dates[dates.year == yr]
        if len(sub) and np.isfinite(edge.get(sub[-1], np.nan)):
            say(f"{yr:>6} {100*edge[sub[-1]]:>+6.1f}%")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
