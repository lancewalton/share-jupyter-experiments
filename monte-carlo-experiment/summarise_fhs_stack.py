"""Summarise the full FHS-EWMA vs FHS-stack OOS results (reads checkpoint CSV)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

SP = ("/private/tmp/claude-501/-Users-lance-Projects-share-jupyter-experiments-"
      "monte-carlo-experiment/ecdd1cf6-87c2-423e-ad16-60786d5aca36/scratchpad")
RESULTS = sys.argv[1] if len(sys.argv) > 1 else f"{SP}/fhs_stack_full.csv"
LEVELS = [0.50, 0.75, 0.90, 0.95, 0.99]
COLS = ["cov50", "cov75", "cov90", "cov95", "cov99"]


def _summary(label, w):
    n = len(w)
    print(f"=== {label}  (n={n}) ===")
    print(f"  mean coverage@h10   {'ewma':>18}   {'stack':>8}   nominal")
    for col, p in zip(COLS, LEVELS):
        e, s = w[(col, "fhs_ewma")].mean(), w[(col, "fhs_stack")].mean()
        print(f"    {col:<6}{'':>10}{e:>10.3f}   {s:>8.3f}   {p:.2f}")
    e99h1 = w[("cov99_h1", "fhs_ewma")].mean(); s99h1 = w[("cov99_h1", "fhs_stack")].mean()
    print(f"    cov99_h1{'':>8}{e99h1:>10.3f}   {s99h1:>8.3f}   0.99")
    # win rates (stack better than ewma)
    tail_closer = (np.abs(w[("cov99", "fhs_stack")] - 0.99)
                   < np.abs(w[("cov99", "fhs_ewma")] - 0.99)).mean()
    crps_better = (w[("crps10", "fhs_stack")] < w[("crps10", "fhs_ewma")]).mean()
    body_closer = (np.abs(w[("cov50", "fhs_stack")] - 0.50)
                   < np.abs(w[("cov50", "fhs_ewma")] - 0.50)).mean()
    ratio = np.median(w[("crps10", "fhs_stack")] / w[("crps10", "fhs_ewma")])
    print(f"  stack beats ewma:  99%-tail {tail_closer:.0%}   50%-body {body_closer:.0%}   "
          f"CRPS {crps_better:.0%}   median CRPS ratio {ratio:.3f}\n")


def main():
    df = pd.read_csv(RESULTS)
    w = df.pivot_table(index=["source", "ticker"], columns="model",
                       values=COLS + ["cov99_h1", "crps10"])
    _summary("ALL", w)
    for src in ["yfinance", "ukinvest", "fx"]:
        sub = w[w.index.get_level_values("source") == src]
        if len(sub):
            _summary(src, sub)


if __name__ == "__main__":
    main()
