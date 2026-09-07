"""Phase 2: harvest the variance risk premium naively.

Short a 1-month variance swap every month (non-overlapping, constant vega), pay
realised. Measure the risk-adjusted return AND the crashes -- the naive seller's
Sharpe is flattered until a vol spike arrives.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.returns import log_returns
from vrp.data import panel
from vrp.premium import realised_vol
from vrp.strategy import stats, varswap_pnl

H = 21


def decisions(df):
    """Monthly non-overlapping decision points; return (pnl, dates, vix, rv)."""
    rets = np.concatenate([[np.nan], log_returns(df["spx"].to_numpy())])
    vixv = df["vix"].to_numpy()
    idx = np.arange(22, len(df) - H - 1, H)
    vix = vixv[idx]
    rv = np.array([realised_vol(rets[i + 1:i + 1 + H]) for i in idx])
    return varswap_pnl(vix, rv), df.index.to_numpy()[idx], vix, rv, idx


def main():
    df = panel()
    pnl, dates, vix, rv, _ = decisions(df)
    s = stats(pnl)
    print(f"naive short-variance harvest  ({s['n']} months, "
          f"{str(dates[0])[:7]}..{str(dates[-1])[:7]}):")
    print(f"  annualised Sharpe   {s['sharpe']:+.2f}")
    print(f"  mean monthly P&L    {s['mean']:+.2f} vol-pts/vega")
    print(f"  worst month         {s['worst']:+.1f}  (the tail)")
    print(f"  max drawdown        {s['mdd']:+.1f}")
    print(f"  % profitable months {100*np.mean(pnl > 0):.0f}%")

    print("\n  sub-period Sharpe (thirds):", end=" ")
    for part in np.array_split(np.arange(len(pnl)), 3):
        print(f"{stats(pnl[part])['sharpe']:+.2f}", end="  ")
    print()
    order = np.argsort(pnl)[:4]
    print("  worst months:", ", ".join(f"{str(dates[i])[:7]} ({pnl[i]:+.0f})" for i in order))

    _plot(dates, pnl)
    print("\nsaved phase2_harvest.png")


def _plot(dates, pnl):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    a1.plot(dates, np.cumsum(pnl), lw=1.4)
    a1.set_title("Cumulative P&L (per vega) — small gains, sharp cliffs")
    a1.set_ylabel("cumulative vol-points")
    a2.bar(dates, pnl, width=25, color=np.where(pnl > 0, "tab:green", "tab:red"))
    a2.set_title("Monthly P&L — the premium is many small wins, few huge losses")
    a2.axhline(0, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig("phase2_harvest.png", dpi=110)


if __name__ == "__main__":
    main()
