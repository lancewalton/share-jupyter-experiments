"""Phase 5: the real tradeable test -- short VIX futures, net of costs.

Replace the variance-swap proxy with the actual short-VIX-futures return
(carry - MTM, from the term structure), 2007-2026, and re-ask whether regime
gating helps -- now with roll, term structure, and transaction costs in.

Gates (all causal, from prior-day data):
  * VIX-gate      -- short only when VIX is below its expanding median;
  * contango-gate -- short only when the curve is in contango (VIX3M > VIX), the
                     classic short-vol danger filter (backwardation = stress);
  * combined      -- both.
Costs: a daily roll cost + a switching cost when the gate flips.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vrp.data import term_structure
from vrp.futures import equity_stats, short_vix_return

C_ROLL, C_SWITCH = 0.0001, 0.0005   # ~2.5%/yr roll cost; 5 bps per full switch


def _strategy(sr, p, c_roll=C_ROLL, c_switch=C_SWITCH):
    p = np.asarray(p, float)
    dp = np.abs(np.diff(p, prepend=p[0]))
    r = p * sr - c_roll * p - c_switch * dp
    return r


def main():
    df = term_structure()
    vix, vix3m = df["vix"].to_numpy(), df["vix3m"].to_numpy()
    dates = df.index.to_numpy()
    n = len(vix)
    sr = short_vix_return(vix, vix3m)

    med = np.array([np.median(vix[:max(t, 1)]) for t in range(n)])   # expanding, causal
    lag = lambda a: np.concatenate([[np.nan], a[:-1]])               # prior-day value
    vix_l, v3_l, med_l = lag(vix), lag(vix3m), med
    gates = {
        "naive":          np.ones(n),
        "VIX-gate":       (vix_l < med_l).astype(float),
        "contango-gate":  (v3_l > vix_l).astype(float),
        "combined":       ((v3_l > vix_l) & (vix_l < med_l)).astype(float),
    }
    for g in gates.values():
        g[0] = 0.0

    print(f"short VIX futures, net of costs  ({n} days, "
          f"{str(dates[0])[:10]}..{str(dates[-1])[:10]})\n")
    print(f"{'strategy':14s} {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s} "
          f"{'worstDay':>9s} {'%short':>7s}")
    series = {}
    for name, p in gates.items():
        r = _strategy(sr, p)
        series[name] = r
        s = equity_stats(r[1:])
        print(f"{name:14s} {s['sharpe']:+7.2f} {s['cagr']*100:+6.0f}% "
              f"{s['mdd']*100:+6.0f}% {s['worst']*100:+8.0f}% {100*np.mean(p):6.0f}%")

    # the blow-up windows
    print("\nnaive short-vol in the famous blow-ups (cumulative return):")
    for lo, hi, tag in [("2008-09-01", "2008-12-01", "GFC 2008"),
                        ("2018-02-01", "2018-02-15", "volmageddon Feb-2018"),
                        ("2020-02-20", "2020-03-20", "COVID 2020")]:
        m = (dates >= np.datetime64(lo)) & (dates <= np.datetime64(hi))
        for name in ("naive", "contango-gate"):
            cum = np.prod(1 + series[name][m][np.isfinite(series[name][m])]) - 1
            print(f"  {tag:22s} {name:14s} {cum*100:+7.0f}%")

    _plot(dates, series)
    print("\nsaved phase5_futures.png")


def _plot(dates, series):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for name in ("naive", "VIX-gate", "contango-gate", "combined"):
        r = np.nan_to_num(series[name])
        a1.plot(dates, np.cumprod(1 + r), lw=1.3, label=name)
    a1.set_yscale("log"); a1.set_title("Growth of $1 short VIX futures (net of costs)")
    a1.legend(fontsize=8); a1.set_ylabel("equity (log)")
    for name in ("naive", "contango-gate"):
        r = np.nan_to_num(series[name]); eq = np.cumprod(1 + r)
        a2.plot(dates, eq / np.maximum.accumulate(eq) - 1, lw=1.0, label=name)
    a2.set_title("Drawdown — gating tames the blow-ups"); a2.legend(fontsize=8)
    a2.set_ylabel("drawdown")
    fig.tight_layout(); fig.savefig("phase5_futures.png", dpi=110)


if __name__ == "__main__":
    main()
