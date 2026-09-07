"""Phase 8: contrarian rotation between REGIONS — the promising case?

Regional equity leadership runs in decade-long cycles (US in the '90s, emerging in
the 2000s, US again since 2010) — the cyclical, reverting behaviour of asset
CLASSES (where contrarian worked, Phase 6), not the trending of sectors (where it
failed, Phase 7). So regions are the natural place the idea should pay. Ken French
regional total returns (USD), 1990-2026: North America, Europe, Japan, Asia
Pacific, Emerging.

  1. reversion: cross-sectional rank-IC(trailing L-year, forward H-year); <0 = the
     beaten-down region wins next.
  2. backtest: contrarian (hold the worst 2 of 5) vs equal-weight, momentum, and
     buy-hold US — does rotating into the out-of-favour region beat holding the US?
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pca.french import load_regions

COST = 0.0005


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    ra, rb = np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    R, rf = load_regions()
    S = R.to_numpy(); rfv = rf.to_numpy(); dates = R.index; n, p = S.shape
    us = R.columns.get_loc("North America")
    print(f"{p} regions · {n} months · {dates[0].date()}..{dates[-1].date()}\n  "
          + ", ".join(R.columns) + "\n")

    print("cross-sectional reversion — rank-IC(trailing, forward); <0 = contrarian wins:")
    print(f"{'':9s}" + "".join(f"  fwd {h}y" for h in (1, 3, 5)))
    for L in (1, 3, 5):
        row = f"trail {L}y"
        for H in (1, 3, 5):
            ics = []
            for t in range(12 * L, n - 12 * H, 12 * H):
                tr = np.array([np.log1p(S[t - 12 * L:t, s]).sum() for s in range(p)])
                fw = np.array([np.log1p(S[t:t + 12 * H, s]).sum() for s in range(p)])
                ics.append(_spearman(tr, fw))
            row += f"  {np.mean(ics):+7.2f}"
        print(f"  {row}")

    RB, N = 12, 2
    def run(pick, L_form):
        w = np.ones(p) / p; ret = np.full(n, np.nan); to = 0.0
        for t in range(n):
            if t >= L_form and (t - L_form) % RB == 0:
                trail = np.array([np.log1p(S[t - L_form:t, s]).sum() for s in range(p)])
                nw = np.zeros(p); nw[pick(trail)] = 1.0 / N; to = np.abs(nw - w).sum(); w = nw
            if t > 0:
                ret[t] = w @ S[t] - (COST * to if t >= L_form and (t - L_form) % RB == 0 else 0)
        return ret
    worst = lambda tr: np.argsort(tr)[:N]
    best = lambda tr: np.argsort(tr)[-N:]
    eqw = np.array([S[t].mean() if t > 0 else np.nan for t in range(n)])
    usbh = np.r_[np.nan, S[1:, us]]

    def sh(r):
        m = np.isfinite(r); ex = (r - rfv)[m]; return ex.mean() / ex.std() * np.sqrt(12)
    print("\nSharpe by formation (hold worst/best 2 of 5):")
    print(f"{'formation':11s} {'contrarian':>11s} {'momentum':>10s} {'equal-wt':>9s} {'US':>6s}")
    for L in (36, 60, 120):
        print(f"  {L//12}y{'':8s} {sh(run(worst, L)):+11.2f} {sh(run(best, L)):+10.2f} "
              f"{sh(eqw):+9.2f} {sh(usbh):+6.2f}")

    books = {"contrarian (worst 2, 5y)": run(worst, 60), "momentum (best 2, 5y)": run(best, 60),
             "equal-weight regions": eqw, "buy-hold US": usbh}
    print("\ndetail (5y formation):")
    print(f"{'strategy':26s} {'ann.ret':>8s} {'vol':>6s} {'Sharpe':>7s} {'end $1':>8s}")
    for name, r in books.items():
        m = np.isfinite(r)
        print(f"{name:26s} {r[m].mean()*12*100:+7.1f}% {r[m].std()*np.sqrt(12)*100:5.1f}% "
              f"{sh(r):+7.2f} {1+np.expm1(np.log1p(r[m]).sum()):8.0f}x")

    _plot(dates, books)
    print("\nsaved phase8_regional.png")


def _plot(dates, books):
    fig, ax = plt.subplots(figsize=(11, 5.4))
    cols = {"contrarian (worst 2, 5y)": "#0c757f", "momentum (best 2, 5y)": "#a8631a",
            "equal-weight regions": "#4a8ca8", "buy-hold US": "#14181b"}
    for name, r in books.items():
        eq = np.exp(np.nancumsum(np.log1p(np.where(np.isfinite(r), r, 0.0))))
        ax.plot(dates, eq, lw=1.7, label=name, color=cols[name],
                ls="--" if name == "buy-hold US" else "-")
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log, total return USD)")
    ax.set_title("Contrarian regional rotation vs buy-and-hold US (1990–2026)")
    ax.legend(fontsize=9); ax.axhline(1, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig("phase8_regional.png", dpi=110)


if __name__ == "__main__":
    main()
