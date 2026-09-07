"""Phase 2: is the market rotating between segments over time?

Two descriptive maps over 1999-2026:

  * ROTATION -- fix the segment definitions (full-sample loadings) and track how
    much of the market's day-to-day activity each segment carries in a rolling
    window. If the composition shifts, the market is trading a different theme.
  * INTEGRATION -- the absorption ratio: the share of variance the single top
    principal component takes in a rolling window. It rises toward 1 when
    everything co-moves (crises), the classic systemic-fragility signal.

Descriptive (full-sample loadings), so it maps history; the causal/predictive
version is Phase 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pca.data import build_returns
from pca.factors import pca, standardize

WIN = 126        # ~6-month rolling window
SEG = {0: "Rates", 1: "Dollar", 2: "Risk-off", 3: "Energy", 4: "Oil–gas"}
CRISES = {"2008-09-15": "Lehman", "2011-08-01": "Euro/US downgrade",
          "2020-03-01": "COVID", "2022-02-24": "Ukraine/hikes"}


def absorption_ratio(Z, win=WIN, step=5):
    """Rolling top-eigenvalue share of the correlation matrix (systemic integration)."""
    n = len(Z)
    idx = np.arange(win, n, step)
    ar = np.empty(len(idx))
    for k, t in enumerate(idx):
        C = np.corrcoef(Z[t - win:t], rowvar=False)
        w = np.linalg.eigvalsh(C)
        ar[k] = w[-1] / w.sum()
    return idx, ar


def main():
    R = build_returns()
    dates = R.index
    Z, _, _ = standardize(R.to_numpy())
    var, loadings, scores = pca(Z)          # fixed full-sample segments

    # rolling activity (variance) of each fixed segment -> composition shares
    S = pd.DataFrame(scores, index=dates)
    act = S.pow(2).rolling(WIN).mean()       # activity of each PC
    share = act.div(act.sum(axis=1), axis=0) # fraction of total activity per PC

    idx, ar = absorption_ratio(Z)

    print(f"panel {R.shape[1]} instruments x {len(R)} days, {dates[0].date()}..{dates[-1].date()}\n")
    print("segment activation share — full-sample vs crisis windows:")
    full = share.mean()
    for i, name in SEG.items():
        print(f"  {name:9s} avg {full[i]*100:4.0f}%")
    print(f"\nabsorption ratio (top-PC share): median {np.median(ar):.2f}  "
          f"calm~{np.percentile(ar,20):.2f}  crisis~{np.percentile(ar,95):.2f}")
    # how much does the risk-off segment swell in crises?
    ro = share[2]
    for d in ("2020-03-20", "2008-11-15"):
        v = ro.asof(pd.Timestamp(d))
        print(f"  Risk-off share around {d[:7]}: {v*100:.0f}%  (vs {full[2]*100:.0f}% avg)")

    _plot(dates, share, idx, ar)
    print("\nsaved phase2_rotation.png")


def _plot(dates, share, idx, ar):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1], "hspace": 0.28})
    cols = ["#0c757f", "#4a8ca8", "#a8631a", "#6b8e23", "#b07a3c"]
    m = share[0].notna().to_numpy()                 # drop the rolling warm-up
    x = dates[m]
    top = [share[i].to_numpy()[m] for i in SEG]
    other = np.clip(1 - sum(top), 0, None)
    a1.stackplot(x, *top, other, labels=list(SEG.values()) + ["Other"],
                 colors=cols + ["#c9d2d4"], alpha=.92)
    a1.set_ylim(0, 1); a1.set_xlim(x[0], x[-1]); a1.margins(x=0)
    a1.set_ylabel("share of market activity")
    a1.set_title("Segment rotation — which theme the cross-asset market is trading")
    a1.legend(loc="lower center", ncol=6, fontsize=8, frameon=False,
              bbox_to_anchor=(0.5, 1.02))
    a2.plot(dates[idx], ar, color="#0c757f", lw=1.2)
    a2.set_ylabel("absorption ratio\n(top-PC share)")
    a2.set_title("Systemic integration — one factor dominates in crises")
    a2.margins(x=0)
    for d, name in CRISES.items():
        t = pd.Timestamp(d)
        for ax in (a1, a2):
            ax.axvline(t, color="#14181b", lw=0.8, ls=":")
        a1.text(t, 0.015, name, rotation=90, fontsize=7.5, color="#14181b",
                va="bottom", ha="right")
    fig.savefig("phase2_rotation.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
