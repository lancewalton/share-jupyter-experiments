"""Phase 1: derive the cross-asset segments (static PCA, 1999-2026).

Decompose the standardised cross-asset return panel into its principal components
-- the market's segments -- and read off what each one is from its loadings.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pca.data import asset_classes, build_returns
from pca.factors import pca, standardize


def main():
    R = build_returns()
    labels = list(R.columns)
    cls = asset_classes()
    Z, _, _ = standardize(R.to_numpy())
    var, loadings, scores = pca(Z)

    print(f"panel: {len(labels)} instruments x {len(R)} days  "
          f"{R.index[0].date()}..{R.index[-1].date()}\n")
    print("variance explained:  " + "  ".join(
        f"PC{i+1} {v*100:.0f}%" for i, v in enumerate(var[:6])))
    print(f"  first 3 PCs capture {var[:3].sum()*100:.0f}% of cross-asset variance\n")

    for i in range(4):
        order = np.argsort(-np.abs(loadings[i]))
        top = "  ".join(f"{labels[j]} {loadings[i,j]:+.2f}" for j in order[:5])
        print(f"PC{i+1} ({var[i]*100:.0f}%): {top}")

    _plot(labels, cls, var, loadings)
    print("\nsaved phase1_segments.png")


def _plot(labels, cls, var, loadings):
    k = 5
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                 gridspec_kw={"width_ratios": [1, 1.5]})
    a1.bar(range(1, len(var) + 1), var * 100, color="tab:teal" if False else "#0c757f")
    a1.set_xlabel("component"); a1.set_ylabel("variance explained (%)")
    a1.set_title("Scree — how many segments?")
    L = loadings[:k]
    im = a2.imshow(L.T, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    a2.set_xticks(range(k)); a2.set_xticklabels([f"PC{i+1}\n{var[i]*100:.0f}%" for i in range(k)])
    a2.set_yticks(range(len(labels))); a2.set_yticklabels(labels, fontsize=9)
    for i in range(k):
        for j in range(len(labels)):
            a2.text(i, j, f"{L[i,j]:+.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(L[i, j]) > 0.4 else "#333")
    a2.set_title("Segment loadings — what each PC is")
    fig.colorbar(im, ax=a2, fraction=0.046, label="loading")
    fig.tight_layout(); fig.savefig("phase1_segments.png", dpi=110)


if __name__ == "__main__":
    main()
