"""Phase 3: can risk-management tame the carry crash?

Carry pays a Sharpe ~0.36 but with a −27% drawdown (2008/2020 risk-off unwinds).
Unlike the variance premium's from-calm blowup, carry crashes build up over weeks
of rising volatility — so a vol signal has a chance. We test, versus naive carry:

  * vol-targeted -- scale inversely to carry's OWN trailing 12-month volatility
    (the Barroso–Santa-Clara idea);
  * VIX-scaled   -- scale inversely to the VIX level (cut when fear is high).

For a fair comparison every book is scaled to the same 10% annual volatility, so
Sharpe and max-drawdown are read at matched risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fx.data import _level, build_total_returns

N, TGT = 4, 0.10


def _norm(r, tgt=TGT):
    """Scale a return series to `tgt` annual vol (for matched-risk comparison)."""
    v = np.nanstd(r) * np.sqrt(12)
    return r * (tgt / v) if v > 0 else r


def _stats(r):
    m = np.isfinite(r); x = r[m]; eq = np.cumsum(x)
    return (x.mean() / x.std() * np.sqrt(12), (eq - np.maximum.accumulate(eq)).min(),
            x.min())


def main():
    total, diff = build_total_returns()
    T = total.to_numpy(); D = diff.to_numpy(); dates = total.index; n, p = T.shape
    vix = _level("VIXCLS").resample("ME").last().reindex(dates).to_numpy()

    carry = np.full(n, np.nan)
    for t in range(1, n):
        o = np.argsort(D[t - 1]); w = np.zeros(p); w[o[-N:]] = 1 / N; w[o[:N]] = -1 / N
        carry[t] = w @ T[t]

    # causal scalers — sized on info known at the START of month t (i.e. t-1)
    tvol = pd.Series(carry).rolling(12).std().shift(1).to_numpy() * np.sqrt(12)
    s_vt = np.clip(0.10 / tvol, 0, 3.0)
    vix_lag = pd.Series(vix).shift(1).to_numpy()                    # prior-month VIX
    med_vix = pd.Series(vix).expanding().median().shift(1).to_numpy()
    s_vx = np.clip(med_vix / vix_lag, 0, 3.0)

    books = {"naive carry": carry,
             "vol-targeted": s_vt * carry,
             "VIX-scaled": s_vx * carry}
    print(f"carry risk-management · {n} months · {dates[0].date()}..{dates[-1].date()}"
          "  (all scaled to 10% vol)\n")
    print(f"{'strategy':16s} {'Sharpe':>7s} {'maxDD':>7s} {'worst mo':>9s}  {'avg exposure':>12s}")
    plot = {}
    for name, r in books.items():
        rn = _norm(r); plot[name] = rn
        sh, dd, wm = _stats(rn)
        expo = np.nanmean(s_vt if name == "vol-targeted" else s_vx if name == "VIX-scaled" else np.ones(n))
        print(f"{name:16s} {sh:+7.2f} {dd*100:+6.0f}% {wm*100:+8.1f}%  {expo:12.2f}")

    _plot(dates, plot)
    print("\nsaved phase3_carry_riskmgmt.png")


def _plot(dates, plot):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    cols = {"naive carry": "#14181b", "vol-targeted": "#0c757f", "VIX-scaled": "#a8631a"}
    for name, r in plot.items():
        s = np.where(np.isfinite(r), r, 0.0)
        ax.plot(dates, np.cumsum(s), lw=1.7, color=cols[name],
                ls="--" if name == "naive carry" else "-",
                label=f"{name}  (Sharpe {np.nanmean(r)/np.nanstd(r)*np.sqrt(12):+.2f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("cumulative return (all at 10% vol)")
    ax.set_title("Risk-managing the carry crash (2002–2026)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig("phase3_carry_riskmgmt.png", dpi=110)


if __name__ == "__main__":
    main()
