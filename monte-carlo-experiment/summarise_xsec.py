"""Summarise cross-asset OOS results (reads xsec_results.csv; works on partial)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = ("/private/tmp/claude-501/-Users-lance-Projects-share-jupyter-experiments-"
      "monte-carlo-experiment/ecdd1cf6-87c2-423e-ad16-60786d5aca36/scratchpad")
RESULTS_CSV = f"{SP}/xsec_results.csv"


def _load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    # wide: one row per (source,ticker), columns per model/metric
    return df.pivot_table(index=["source", "ticker"], columns="model",
                          values=["cov90", "cov99", "crps"])


def _summary(label: str, w: pd.DataFrame) -> None:
    n = len(w)
    e90, f090, f190 = (w[("cov90", m)].mean() for m in ("ewma", "fhs0", "fhs1"))
    e99, f099, f199 = (w[("cov99", m)].mean() for m in ("ewma", "fhs0", "fhs1"))
    ratio = np.median(w[("crps", "fhs0")] / w[("crps", "ewma")])
    tail = (np.abs(w[("cov99", "fhs0")] - 0.99) < np.abs(w[("cov99", "ewma")] - 0.99)).mean()
    crps = (w[("crps", "fhs0")] < w[("crps", "ewma")]).mean()
    print(f"=== {label}  (n={n}) ===")
    print(f"  mean cov90:  ewma {e90:.3f}  fhs0 {f090:.3f}  fhs1 {f190:.3f}   (nom 0.900)")
    print(f"  mean cov99:  ewma {e99:.3f}  fhs0 {f099:.3f}  fhs1 {f199:.3f}   (nom 0.990)")
    print(f"  FHS(g0) vs EWMA: tail-closer {tail:.0%}  CRPS-better {crps:.0%}  "
          f"median CRPS ratio {ratio:.3f}\n")


def main() -> None:
    if not os.path.exists(RESULTS_CSV):
        print("no results yet"); return
    w = _load()
    _summary("ALL", w)
    for src in ["yfinance", "ukinvest", "fx"]:
        sub = w[w.index.get_level_values("source") == src]
        if len(sub):
            _summary(src, sub)

    srcs = [s for s in ["yfinance", "ukinvest", "fx"]
            if (w.index.get_level_values("source") == s).any()]
    fig, axes = plt.subplots(1, len(srcs), figsize=(4.7 * len(srcs), 4.6), sharey=True)
    if len(srcs) == 1:
        axes = [axes]
    for ax, src in zip(axes, srcs):
        sub = w[w.index.get_level_values("source") == src]
        data = [sub[("cov99", m)].to_numpy() for m in ("ewma", "fhs0", "fhs1")]
        ax.boxplot(data, tick_labels=["ewma", "fhs0", "fhs1"], showmeans=True)
        ax.axhline(0.99, color="tab:green", lw=0.8, ls="--")
        ax.set_title(f"{src}  (n={len(sub)})")
    axes[0].set_ylabel("99% coverage (h10) across series")
    fig.suptitle("Cross-asset tail calibration: FHS vs EWMA (nominal 0.99 dashed)")
    fig.tight_layout()
    fig.savefig("xsec_all.png", dpi=110)
    print("saved xsec_all.png")


if __name__ == "__main__":
    main()
