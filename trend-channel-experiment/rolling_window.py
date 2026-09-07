"""Rolling-window view of the channel-SELECTOR edge, to see whether the decay was a
sharp break or a gradual fade, and whether the edge tracks market volatility (the
QE/vol-suppression hypothesis).

Trailing 3-year (756 trading day) annualised return of the selection book and of
equal-weight buy-and-hold, their difference (edge), and trailing market realised
vol. Reports year-end snapshots and corr(edge, vol).

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u rolling_window.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import load_ftse
from channel_entry_sweep import precompute_rich

HERE = Path(__file__).resolve().parent
BARS_YR = 252
W = 756
L = 250
G_MIN, R2_MIN, W_MIN = 0.10, 0.80, 0.0
SPREAD = 10 / 1e4
OUT = HERE / "rolling_window_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def roll_ann(daily: pd.Series) -> pd.Series:
    lg = np.log1p(daily.astype("float64"))
    return np.expm1(lg.rolling(W).sum() * (BARS_YR / W))


def main() -> None:
    names = load_ftse.universe()
    say(f"Loading {len(names)} FTSE names; precomputing L={L}...")
    closes, ANN, RELW, R2 = {}, {}, {}, {}
    for t in names:
        try:
            df = load_ftse.load(t)
        except Exception:
            continue
        a = precompute_rich(df, L)
        if a is None:
            continue
        closes[t] = df["close"]
        _, _, ann, relw, r2 = a
        ANN[t] = pd.Series(ann, index=df.index)
        RELW[t] = pd.Series(relw, index=df.index)
        R2[t] = pd.Series(r2, index=df.index)
    dates = pd.DatetimeIndex(sorted(set().union(*[set(closes[t].index) for t in closes])))
    RETS = pd.DataFrame(closes).reindex(dates).pct_change()
    ANN = pd.DataFrame(ANN).reindex(dates); RELW = pd.DataFrame(RELW).reindex(dates); R2 = pd.DataFrame(R2).reindex(dates)
    qual = ((ANN >= G_MIN) & (R2 >= R2_MIN) & (RELW >= W_MIN)).fillna(False)
    held = qual.shift(1).fillna(False)
    n = held.sum(axis=1)
    mret = RETS.where(held).mean(axis=1).fillna(0.0)
    changes = held.astype(int).diff().abs().sum(axis=1)
    cost = (SPREAD * changes / n.replace(0, np.nan)).fillna(0.0)
    sel = pd.Series((mret.to_numpy() - cost.to_numpy()).astype("float64"), index=dates)
    uni = RETS.mean(axis=1).fillna(0.0).astype("float64")

    sel_a, uni_a = roll_ann(sel), roll_ann(uni)
    edge = sel_a - uni_a
    vol = uni.rolling(W).std() * np.sqrt(BARS_YR)

    say(f"\nRolling 3y (756d) — channel-selector (L={L}, w_min={W_MIN}) vs EW buy-and-hold\n")
    say(f"{'year-end':>9} {'sel_ann':>8} {'bh_ann':>8} {'edge':>7} {'mkt_vol':>8} {'n_held':>7}")
    for yr in range(2003, 2027):
        sub = dates[(dates.year == yr)]
        if len(sub) == 0:
            continue
        d = sub[-1]
        if not np.isfinite(edge.get(d, np.nan)):
            continue
        say(f"{d.date()!s:>9} {100*sel_a[d]:>7.1f}% {100*uni_a[d]:>7.1f}% {100*edge[d]:>+6.1f}% "
            f"{100*vol[d]:>7.1f}% {n[d]:>7.0f}")

    m = pd.DataFrame({"edge": edge, "vol": vol}).dropna()
    say(f"\ncorr(rolling edge, rolling market vol) = {m.edge.corr(m.vol):+.2f}  (n={len(m)})")
    pre = m[m.index < pd.Timestamp('2013-01-01')]
    post = m[m.index >= pd.Timestamp('2013-01-01')]
    say(f"  mean edge pre-2013 {100*pre.edge.mean():+.1f}% (mean vol {100*pre.vol.mean():.1f}%)  |  "
        f"post-2013 {100*post.edge.mean():+.1f}% (mean vol {100*post.vol.mean():.1f}%)")
    say("  positive corr => the edge is bigger when market volatility is higher (vol-suppression story).")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
