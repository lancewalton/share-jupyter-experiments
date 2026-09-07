"""Phase 6: does a long-horizon contrarian factor rotation beat buy-and-hold?

The idea: buy the factor money has fled, hold for years until it flows back, then
rotate. It works only if asset classes MEAN-REVERT over multi-year horizons.

  1. Reversion test -- pooled rank-IC of trailing L-year return vs forward H-year
     return. Negative = reversion (contrarian premise holds); positive = momentum
     (buy-the-loser is exactly wrong).
  2. Backtest -- an annually-rebalanced contrarian book (overweight the worst
     multi-year performers) vs equal-weight buy-hold, momentum, and equity-only.

Return proxies are imperfect (log price for equity/commodity/FX, -Δyield for
bonds; no dividends/carry/roll), so read the STRATEGY comparison as directional;
the reversion sign is the robust part.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pca.data import build_returns

ASSETS = ["US equity", "UST 2Y", "UST 10Y", "UST 30Y", "WTI oil",
          "Nat gas", "EUR/USD", "USD/JPY", "GBP/USD"]


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    ra, rb = np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def _sharpe(r):
    r = np.asarray(r); return r.mean() / (r.std() + 1e-12) * np.sqrt(12)


def main():
    R = build_returns()[ASSETS]
    M = R.resample("ME").sum()                      # monthly log returns
    dates = M.index; X = M.to_numpy(); n, p = X.shape
    print(f"{p} assets · {n} months · {dates[0].date()}..{dates[-1].date()}\n")

    # --- 1. reversion test: trailing L vs forward H (pooled, thinned) ---
    def roll_sum(a, w):
        return np.array([a[t - w:t].sum() if t - w >= 0 else np.nan for t in range(len(a))])
    print("long-horizon reversion — pooled rank-IC(trailing, forward); <0 = contrarian works:")
    print(f"{'':8s}" + "".join(f"  fwd {h}y" for h in (1, 2, 3)))
    for L in (1, 2, 3):
        row = f"trail {L}y"
        for H in (1, 2, 3):
            tr, fw = [], []
            for j in range(p):
                a = X[:, j]
                t = roll_sum(a, 12 * L)
                f = np.array([a[i:i + 12 * H].sum() if i + 12 * H <= n else np.nan
                              for i in range(n)])
                sub = np.arange(12 * L, n - 12 * H, 12 * H)   # non-overlapping
                tr += list(t[sub]); fw += list(f[sub])
            row += f"  {_spearman(tr, fw):+7.2f}"
        print(f"  {row}")

    # --- 2. backtest: annually-rebalanced books ---
    L_form, RB = 24, 12                             # 2y formation, annual rebalance
    def run(weight_fn):
        w = np.ones(p) / p; ret = np.full(n, np.nan)
        for t in range(n):
            if t >= L_form and (t - L_form) % RB == 0:
                trail = X[t - L_form:t].sum(0)
                w = weight_fn(trail)
            if t > 0:
                ret[t] = w @ X[t]
        return ret
    def contrarian(tr):
        r = np.argsort(np.argsort(tr)); wgt = (p - 1 - r).astype(float); return wgt / wgt.sum()
    def momentum(tr):
        r = np.argsort(np.argsort(tr)); wgt = r.astype(float) + 1; return wgt / wgt.sum()

    books = {"contrarian (buy losers)": run(contrarian),
             "momentum (buy winners)": run(momentum),
             "equal-weight buy-hold": run(lambda tr: np.ones(p) / p),
             "equity only": np.r_[np.nan, X[1:, 0]]}
    print("\nbacktest (2y formation, annual rebalance):")
    print(f"{'strategy':26s} {'ann.ret':>8s} {'ann.vol':>8s} {'Sharpe':>7s}")
    for name, r in books.items():
        rr = r[np.isfinite(r)]
        print(f"{name:26s} {rr.mean()*12*100:+7.1f}% {rr.std()*np.sqrt(12)*100:7.1f}% "
              f"{_sharpe(rr):+7.2f}")

    _plot(dates, books)
    print("\nsaved phase6_rotation.png")


def _plot(dates, books):
    fig, ax = plt.subplots(figsize=(11, 5.4))
    cols = {"contrarian (buy losers)": "#0c757f", "momentum (buy winners)": "#a8631a",
            "equal-weight buy-hold": "#4a8ca8", "equity only": "#14181b"}
    for name, r in books.items():
        eq = np.exp(np.nancumsum(np.where(np.isfinite(r), r, 0.0)))
        ax.plot(dates, eq, lw=1.7, label=name, color=cols[name],
                ls="--" if name == "equity only" else "-")
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.set_title("Long-horizon contrarian rotation vs buy-and-hold")
    ax.legend(fontsize=9); ax.axhline(1, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig("phase6_rotation.png", dpi=110)


if __name__ == "__main__":
    main()
