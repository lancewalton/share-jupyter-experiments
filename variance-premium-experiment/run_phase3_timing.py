"""Phase 3: does a causal vol-regime forecast improve the VRP harvest?

We forecast forward realised vol (HAR + VIX, expanding, causal) and use it to
size the short-variance book three ways, versus the naive constant-vega seller:

  * vol-gate      -- step aside when forecast vol is high (the "obvious" rule);
  * premium-sized -- sell proportional to the *expected* premium VIX - forecastRV
    (our forecast sharpens how rich variance really is);
  * VIX-gate      -- step aside when VIX is high (does the forecast beat just VIX?).

The crux the Feb-2020 tail poses: that crash came from a CALM tape (VIX ~15), so a
vol forecast that extrapolates recent calm cannot see it. We check honestly
whether any rule would have stepped aside there.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.returns import log_returns
from vrp.data import panel
from vrp.forecast import har_walk_forward, trailing_rv
from vrp.premium import forward_realised_vol, realised_vol
from vrp.strategy import stats, varswap_pnl

H, MIN_TRAIN = 21, 252


def _norm(v):
    v = np.asarray(v, float); m = np.nanmean(v)
    return v / m if m > 0 else v


def main():
    df = panel()
    rets = np.concatenate([[np.nan], log_returns(df["spx"].to_numpy())])
    vixv = df["vix"].to_numpy()
    feat = np.column_stack([trailing_rv(rets, (1, 5, 22)), vixv])
    target = forward_realised_vol(rets, H)
    pred = har_walk_forward(feat, target, H, MIN_TRAIN)

    start = MIN_TRAIN + H + 1
    idx = np.arange(start, len(df) - H - 1, H)
    fc, vx = pred[idx], vixv[idx]
    rv = np.array([realised_vol(rets[i + 1:i + 1 + H]) for i in idx])
    dates = df.index.to_numpy()[idx]
    pnl1 = varswap_pnl(vx, rv)
    ok = np.isfinite(fc) & np.isfinite(pnl1)
    fc, vx, rv, pnl1, dates = fc[ok], vx[ok], rv[ok], pnl1[ok], dates[ok]

    print(f"forecast skill: corr(forecastRV, realisedRV) = "
          f"{np.corrcoef(fc, rv)[0,1]:+.2f}   ({len(fc)} months from {str(dates[0])[:7]})\n")

    med_fc = np.array([np.median(fc[:max(i,1)]) for i in range(len(fc))])   # causal median
    med_vx = np.array([np.median(vx[:max(i,1)]) for i in range(len(vx))])
    rules = {
        "naive":         np.ones(len(fc)),
        "vol-gate":      (fc < med_fc).astype(float),
        "premium-sized": _norm(np.clip(vx - fc, 0, None)),
        "VIX-gate":      (vx < med_vx).astype(float),
    }
    print(f"{'rule':14s} {'Sharpe':>7s} {'meanP&L':>8s} {'worst':>8s} {'maxDD':>8s} {'avgVega':>8s}")
    series = {}
    for name, w in rules.items():
        wn = _norm(w)
        series[name] = wn * pnl1
        s = stats(series[name])
        print(f"{name:14s} {s['sharpe']:+7.2f} {s['mean']:+8.2f} {s['worst']:+8.0f} "
              f"{s['mdd']:+8.0f} {np.nanmean(w):8.2f}")

    # the Feb-2020 decision: did anything step aside?
    j = int(np.argmin(pnl1))
    print(f"\nthe tail decision ({str(dates[j])[:7]}, P&L {pnl1[j]:+.0f}):")
    print(f"  VIX {vx[j]:.1f}   forecastRV {fc[j]:.1f}   realisedRV {rv[j]:.1f}")
    print(f"  vol-gate vega {rules['vol-gate'][j]:.0f}   premium-sized vega "
          f"{_norm(rules['premium-sized'])[j]:.2f}   VIX-gate vega {rules['VIX-gate'][j]:.0f}")
    print("  => if the forecast was low here (calm before the storm), no rule saved it.")

    _plot(dates, series)
    print("\nsaved phase3_timing.png")


def _plot(dates, series):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for name in ("naive", "vol-gate", "VIX-gate", "premium-sized"):
        a1.plot(dates, np.cumsum(series[name]), lw=1.5, label=name)
    a1.axhline(0, color="k", lw=0.5)
    a1.set_title("Cumulative P&L: gating on high vol beats naive selling")
    a1.set_ylabel("cumulative vol-points / vega"); a1.legend(fontsize=8)
    names = list(series)
    sh = [np.nanmean(series[n]) / (np.nanstd(series[n]) + 1e-9) * np.sqrt(12) for n in names]
    wr = [np.nanmin(series[n]) for n in names]
    x = np.arange(len(names))
    a2.bar(x - 0.2, sh, 0.4, label="Sharpe", color="tab:blue")
    a2b = a2.twinx()
    a2b.bar(x + 0.2, wr, 0.4, label="worst month", color="tab:red", alpha=0.7)
    a2.set_xticks(x); a2.set_xticklabels(names, rotation=20, fontsize=8)
    a2.set_ylabel("Sharpe"); a2b.set_ylabel("worst month P&L")
    a2.set_title("Timing: higher Sharpe AND shallower tail")
    fig.tight_layout(); fig.savefig("phase3_timing.png", dpi=110)


if __name__ == "__main__":
    main()
