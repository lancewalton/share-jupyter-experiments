"""Cross-strategy REGIME DASHBOARD.

A common trend-regime indicator (market efficiency ratio: |net move| / sum|daily
moves| over a trailing window -- high = smooth trend, low = choppy/whipsaw) plotted
against the rolling 3y edge (vs equal-weight buy-and-hold) of four implementable
FTSE strategies:
  - channel selector   (trend / momentum, from this experiment)
  - 12-1 momentum      (long top-quintile trailing 12m-skip-1m return)
  - low-vol tilt       (long bottom-quintile trailing 120d vol)
  - short-term reversal (long bottom-quintile trailing 21d return -- the "fade")

Reports corr(edge, regime) for each (which ideas are trend-regime-linked) and the
current 2025-26 reading (whose regime has turned). Saves a chart. Edges are gross
(pre-cost) for clean regime comparison; costs shift levels, not co-movement.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u regime_dashboard.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import load_ftse
from channel_entry_sweep import precompute_rich

HERE = Path(__file__).resolve().parent
BARS_YR = 252
W = 756           # 3y rolling edge
ER_W = 120        # efficiency-ratio window
L = 250
G_MIN, R2_MIN = 0.10, 0.80
OUT = HERE / "regime_dashboard_results.txt"
CHART = HERE / "charts" / "regime_dashboard.png"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def roll_ann(x: pd.Series) -> pd.Series:
    return np.expm1(np.log1p(x.astype("float64")).rolling(W).sum() * (BARS_YR / W))


def monthly_quintile(signal: pd.DataFrame, RETS: pd.DataFrame, pick: str, frac=0.2) -> pd.Series:
    """Long-only equal-weight portfolio, rebalanced monthly on the prior month-end
    signal; returns daily returns (causal)."""
    ym = signal.index.to_period("M")
    months = ym.unique()
    member = pd.DataFrame(False, index=signal.index, columns=signal.columns)
    for i in range(1, len(months)):
        last_prev = signal.index[ym == months[i - 1]][-1]
        sig = signal.loc[last_prev].dropna()
        if len(sig) < 10:
            continue
        k = max(1, int(len(sig) * frac))
        chosen = (sig.nlargest(k) if pick == "top" else sig.nsmallest(k)).index
        member.loc[ym == months[i], chosen] = True
    return RETS.where(member.shift(1).fillna(False)).mean(axis=1).fillna(0.0)


def efficiency_ratio(level: pd.Series, w: int) -> pd.Series:
    move = level.diff().abs().rolling(w).sum()
    net = (level - level.shift(w)).abs()
    return (net / move).clip(0, 1)


def main() -> None:
    names = load_ftse.universe()
    say(f"Loading {len(names)} FTSE names; precomputing channels (L={L})...")
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
        ANN[t] = pd.Series(ann, index=df.index); RELW[t] = pd.Series(relw, index=df.index); R2[t] = pd.Series(r2, index=df.index)
    dates = pd.DatetimeIndex(sorted(set().union(*[set(closes[t].index) for t in closes])))
    RETS = pd.DataFrame(closes).reindex(dates).pct_change()
    ANN = pd.DataFrame(ANN).reindex(dates); RELW = pd.DataFrame(RELW).reindex(dates); R2 = pd.DataFrame(R2).reindex(dates)
    uni = RETS.mean(axis=1).fillna(0.0)

    # --- strategies (gross daily returns) ---
    qual = ((ANN >= G_MIN) & (R2 >= R2_MIN)).fillna(False)
    channel = RETS.where(qual.shift(1).fillna(False)).mean(axis=1).fillna(0.0)
    closes_df = pd.DataFrame(closes).reindex(dates)
    mom = monthly_quintile(closes_df.shift(21) / closes_df.shift(252) - 1, RETS, "top")
    lowvol = monthly_quintile(-RETS.rolling(120).std(), RETS, "top")   # top of -vol = lowest vol
    reversal = monthly_quintile(closes_df / closes_df.shift(21) - 1, RETS, "bottom")

    strat = {"channel": channel, "momentum": mom, "low_vol": lowvol, "reversal": reversal}
    edges = {k: roll_ann(v) - roll_ann(uni) for k, v in strat.items()}

    idx = (1 + uni).cumprod()
    er = efficiency_ratio(idx, ER_W)
    er_roll = er.rolling(W).mean()   # smooth to match the 3y edge horizon

    say("\n" + "#" * 80)
    say(f"# REGIME DASHBOARD — corr(rolling 3y edge, trend regime [efficiency ratio])")
    say("#" * 80)
    for k, e in edges.items():
        m = pd.DataFrame({"e": e, "r": er_roll}).dropna()
        say(f"  {k:10} corr(edge, trend-regime) = {m.e.corr(m.r):+.2f}   "
            f"full-sample mean edge {100*e.dropna().mean():+.1f}%/yr")

    say("\n  positive corr => the strategy does better in smooth-trend regimes (high ER);")
    say("  negative corr => it does better in choppy/whipsaw regimes (low ER).\n")

    say(f"{'year':>6} {'regime':>7} " + " ".join(f"{k[:8]:>9}" for k in strat))
    for yr in range(2004, 2027):
        sub = dates[dates.year == yr]
        if not len(sub):
            continue
        d = sub[-1]
        if not np.isfinite(er_roll.get(d, np.nan)):
            continue
        row = f"{yr:>6} {er_roll[d]:>7.2f} "
        row += " ".join(f"{100*edges[k].get(d, np.nan):>+8.1f}%" for k in strat)
        say(row)

    # current reading
    d = dates[-1]
    say(f"\nLatest ({d.date()}): trend-regime (ER) = {er_roll[d]:.2f} "
        f"[sample median {er_roll.median():.2f}]")
    for k in strat:
        e = edges[k][d]
        say(f"  {k:10} current 3y edge {100*e:+.1f}%/yr  "
            f"({'favourable regime' if edges[k].corr(er_roll) > 0 else 'adverse regime'} rising)"
            if np.isfinite(e) else f"  {k:10} n/a")

    # --- chart ---
    CHART.parent.mkdir(exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 2]})
    ax0.plot(er_roll.index, er_roll, color="#333", lw=1.3)
    ax0.axhline(er_roll.median(), color="#888", ls=":", lw=0.8)
    ax0.fill_between(er_roll.index, er_roll, er_roll.median(),
                     where=(er_roll > er_roll.median()), color="#4d79c0", alpha=0.25, label="trending")
    ax0.fill_between(er_roll.index, er_roll, er_roll.median(),
                     where=(er_roll <= er_roll.median()), color="#c0504d", alpha=0.25, label="choppy/whipsaw")
    ax0.set_ylabel("trend regime\n(efficiency ratio, 3y avg)"); ax0.legend(fontsize=8, loc="upper right")
    ax0.set_title("Regime dashboard — trend regime vs rolling 3y edge of each strategy (FTSE, gross)")
    colours = {"channel": "#0c757f", "momentum": "#a8631a", "low_vol": "#4d79c0", "reversal": "#c0504d"}
    for k, e in edges.items():
        ax1.plot(e.index, 100 * e, label=f"{k} (corr {edges[k].corr(er_roll):+.2f})", lw=1.3, color=colours[k])
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_ylabel("rolling 3y edge vs buy&hold (%/yr)"); ax1.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(CHART, dpi=130, bbox_inches="tight")
    plt.close(fig)
    say(f"\nchart -> {CHART.relative_to(HERE)}")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"saved -> {OUT.name}")


if __name__ == "__main__":
    main()
