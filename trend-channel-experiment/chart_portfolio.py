"""Two charts for the recommended build (mom+quality+low-vol, vol-targeted, net spread):

  1. Portfolio equity curve from £100, over the full history (vs the market), with a drawdown panel.
  2. Intra-month daily paths of ALL held shares over ALL months, each normalised to its price at the
     start of its holding month -- faint individual paths + median/percentile bands ("typical month").

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u chart_portfolio.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from momentum_tradeability import build, tier_spread, cagr, sharpe, maxdd, N as UNIV_N, BT_START, DATA, OHLCV
from momentum_multifactor import build_factors, zscore
from vol_target_momentum import vol_target

HERE = Path(__file__).resolve().parent
CH = HERE / "charts"
SIG, LIM, INK = "#0c757f", "#a8631a", "#14181b"
FRAC = 0.2


def load_daily_adj():
    df = pd.read_parquet(OHLCV, columns=["date", "code", "adjusted_close"])
    df = df[df["date"] >= "2000-06-01"]
    return df.pivot(index="date", columns="code", values="adjusted_close").sort_index()


def held_sets(M, mret, liq, signal, quality, lowvol):
    """For each holding month, the top-quintile composite names selected the prior month-end."""
    months = M.index
    out = {}
    for i in range(1, len(months)):
        t, th = months[i - 1], months[i]
        if th < pd.Timestamp(BT_START):
            continue
        elig = liq.loc[t].dropna().nlargest(UNIV_N).index
        z = pd.concat([zscore(signal.loc[t, elig]), zscore(quality.loc[t, elig]),
                       zscore(lowvol.loc[t, elig])], axis=1).mean(axis=1, skipna=True)
        z = z[zscore(signal.loc[t, elig]).notna()].dropna()
        if len(z) < 30:
            continue
        k = max(1, int(len(z) * FRAC))
        out[th] = list(z.nlargest(k).index)
    return out


def strat_series(M, mret, liq, signal, quality, lowvol):
    from momentum_stamp_duty import run
    P, _ = run(M, mret, liq, [signal, quality, lowvol], 69, 0.0)
    r = P["r"].dropna()
    vt, _w = vol_target(r, r.std() * np.sqrt(12), 1.0)
    return vt, P["uni"].reindex(vt.index)


def chart_equity(vt, uni):
    eq = 100 * (1 + vt).cumprod()
    mk = 100 * (1 + uni).cumprod()
    dd = eq / eq.cummax() - 1
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
    a.plot(eq.index, eq.values, color=SIG, lw=1.8, label=f"Strategy  (£100 → £{eq.iloc[-1]:,.0f}, {100*cagr(vt):.1f}%/yr, Sharpe {sharpe(vt):.2f})")
    a.plot(mk.index, mk.values, color=INK, lw=1.2, alpha=.55, label=f"Market EW buy-and-hold  (£100 → £{mk.iloc[-1]:,.0f}, {100*cagr(uni):.1f}%/yr)")
    a.set_yscale("log"); a.set_ylabel("Portfolio value (£, log)")
    a.set_title("Recommended build: £100 grown over the full history (net of spread)", fontweight="bold")
    a.legend(loc="upper left", fontsize=9, frameon=False); a.grid(True, which="both", alpha=.15)
    b.fill_between(dd.index, 100 * dd.values, 0, color=LIM, alpha=.5)
    b.set_ylabel("Drawdown (%)"); b.set_ylim(top=2); b.grid(True, alpha=.15)
    b.annotate(f"worst {100*dd.min():.0f}%", xy=(dd.idxmin(), 100 * dd.min()),
               fontsize=9, color=LIM, ha="center", va="top")
    fig.tight_layout(); fig.savefig(CH / "portfolio_equity_curve.png", dpi=130); plt.close(fig)


def chart_month(adj, holds):
    paths, maxlen = [], 0
    for th, names in holds.items():
        sub = adj[(adj.index.year == th.year) & (adj.index.month == th.month)]
        if len(sub) < 2:
            continue
        block = sub[names] if set(names) <= set(sub.columns) else sub.reindex(columns=names)
        for c in names:
            s = block[c].dropna()
            if len(s) < 2:
                continue
            v = 100 * s.values / s.values[0]
            paths.append(v); maxlen = max(maxlen, len(v))
    P = np.full((len(paths), maxlen), np.nan)
    for i, v in enumerate(paths):
        P[i, :len(v)] = v
    days = np.arange(maxlen)
    med = np.nanmedian(P, axis=0)
    p10, p25, p75, p90 = (np.nanpercentile(P, q, axis=0) for q in (10, 25, 75, 90))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    rng = np.random.default_rng(0)
    for i in rng.choice(len(paths), size=min(1500, len(paths)), replace=False):
        v = P[i]; ax.plot(days, v, color=SIG, lw=.4, alpha=.02)
    ax.fill_between(days, p10, p90, color=SIG, alpha=.12, label="10–90th percentile")
    ax.fill_between(days, p25, p75, color=SIG, alpha=.22, label="25–75th percentile")
    ax.plot(days, med, color=LIM, lw=2.4, label="median held share")
    ax.axhline(100, color=INK, lw=.8, alpha=.5)
    ax.set_xlim(0, maxlen - 1); ax.set_ylim(88, 112)
    ax.set_xlabel("Trading day within the holding month (0 = first day)")
    ax.set_ylabel("Price, normalised to start of month (=100)")
    ax.set_title(f"Intra-month behaviour of all held shares, all months\n"
                 f"({len(paths):,} share-months; held names normalised to their month-open price)",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, frameon=False); ax.grid(True, alpha=.15)
    fig.text(.5, -.01, f"median share ends the month at {med[np.isfinite(med)][-1]:.2f} "
             f"(≈ {med[np.isfinite(med)][-1]-100:+.2f}% over the month)", ha="center", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(CH / "held_shares_monthly_paths.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def drawdown_episodes(eq):
    """Peak -> trough -> recovery (new high) episodes, monthly series."""
    peak = eq.cummax()
    dd = eq / peak - 1
    eps, in_dd = [], False
    peak_dt = eq.index[0]
    for i, (dt, d) in enumerate(dd.items()):
        if d >= -1e-9:
            if in_dd:
                eps.append(dict(peak=peak_dt, trough=tr_dt, depth=tr_d, recovery=dt))
                in_dd = False
            peak_dt = dt
        else:
            if not in_dd:
                in_dd = True; tr_d, tr_dt = d, dt
            elif d < tr_d:
                tr_d, tr_dt = d, dt
    if in_dd:
        eps.append(dict(peak=peak_dt, trough=tr_dt, depth=tr_d, recovery=None))
    loc = {dt: i for i, dt in enumerate(eq.index)}
    for e in eps:
        e["to_trough"] = loc[e["trough"]] - loc[e["peak"]]
        e["underwater"] = (loc[e["recovery"]] - loc[e["peak"]]) if e["recovery"] else (len(eq) - 1 - loc[e["peak"]])
    return sorted(eps, key=lambda e: e["depth"])


def chart_recovery(eq):
    eps = drawdown_episodes(eq)[:8]
    dd = eq / eq.cummax() - 1
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(dd.index, 100 * dd.values, 0, color=SIG, alpha=.10)
    for e in eps:
        end = e["recovery"] or eq.index[-1]
        y = 100 * e["depth"]
        ax.plot([e["peak"], end], [y, y], color=LIM, lw=2, solid_capstyle="round", alpha=.9)
        ax.plot([e["trough"]], [y], "o", color=LIM, ms=6)
        rec = f"{e['underwater']} mo" + ("" if e["recovery"] else " (ongoing)")
        ax.annotate(f"{y:.0f}%, back in {rec}", xy=(end, y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.axhline(0, color=INK, lw=.8, alpha=.5)
    ax.set_ylabel("Drawdown (%)"); ax.set_ylim(top=6)
    ax.set_title("Every drawdown recovered: the 8 worst, with months peak → new high",
                 fontweight="bold")
    ax.grid(True, alpha=.15)
    fig.tight_layout(); fig.savefig(CH / "drawdown_recovery.png", dpi=130); plt.close(fig)
    print("\nWorst drawdowns (recommended build, £100 equity curve):", flush=True)
    print(f"  {'peak':>10} {'trough':>10} {'depth':>7} {'to trough':>10} {'to new high':>12}", flush=True)
    for e in eps:
        rec = f"{e['underwater']} mo" if e["recovery"] else f"{e['underwater']}+ mo (ongoing)"
        print(f"  {e['peak'].date().isoformat():>10} {e['trough'].date().isoformat():>10} "
              f"{100*e['depth']:>6.0f}% {e['to_trough']:>8} mo {rec:>12}", flush=True)


def main() -> None:
    print("Building...", flush=True)
    M, mret, liq, signal = build()
    quality, _v, _ = build_factors(M)
    lowvol = -mret.rolling(12).std().shift(1)
    vt, uni = strat_series(M, mret, liq, signal, quality, lowvol)
    print("Equity curve...", flush=True)
    chart_equity(vt, uni)
    chart_recovery(100 * (1 + vt).cumprod())
    print("Reconstructing held sets + daily paths...", flush=True)
    holds = held_sets(M, mret, liq, signal, quality, lowvol)
    adj = load_daily_adj()
    chart_month(adj, holds)
    print(f"saved -> {CH/'portfolio_equity_curve.png'}\n         {CH/'held_shares_monthly_paths.png'}", flush=True)


if __name__ == "__main__":
    main()
