"""Cross-sectional out-of-sample: does FHS's edge hold across many equities?

For each stock we run the fixed models walk-forward and record tail/body
coverage and CRPS at h=10. Then we summarise across the cross-section: mean
calibration and the rate at which FHS beats the EWMA benchmark. Individual
stocks are a harder test than the index (fatter, more idiosyncratic tails).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.data import load_close
from mc.returns import log_returns
from mc.forecasters import gaussian_ewma, fhs_ewma
from mc.walkforward import evaluate

T, MIN_HISTORY, STRIDE, NSIM = 10, 1000, 10, 6000
DATA = "/Users/lance/Projects/shares/data/yfinance"
MAX_STOCKS = 45
MIN_ROWS = 2500

MODELS = {
    "ewma": lambda: gaussian_ewma(lam=0.94, min_obs=MIN_HISTORY),
    "fhs0": lambda: fhs_ewma(lam=0.94, gamma=0.0, n_sims=NSIM, seed=0),
    "fhs1": lambda: fhs_ewma(lam=0.94, gamma=1.0, n_sims=NSIM, seed=0),
}


GLITCH = 0.6  # a >~80% one-day move is a split/corporate-action artefact, not a return


def _load_returns(path):
    try:
        r = log_returns(load_close(path).to_numpy())
    except Exception:
        return None
    if len(r) < MIN_ROWS or np.abs(r).max() > GLITCH:
        return None
    return r


def main() -> None:
    files = sorted(f for f in glob.glob(os.path.join(DATA, "*.csv")))
    rows = {}  # ticker -> {model -> (cov90, cov99, crps)}
    used = []
    for path in files:
        if len(used) >= MAX_STOCKS:
            break
        r = _load_returns(path)
        if r is None:
            continue
        tk = os.path.splitext(os.path.basename(path))[0]
        try:
            res = {m: evaluate(r, f(), T=T, min_history=MIN_HISTORY, stride=STRIDE)
                   for m, f in MODELS.items()}
        except Exception:
            continue
        rows[tk] = {m: (v.coverage[0.90][9], v.coverage[0.99][9], v.mean_crps[9])
                    for m, v in res.items()}
        used.append(tk)

    print(f"stocks used: {len(used)}  (stride={STRIDE}, n_sims={NSIM})")
    print(", ".join(used) + "\n")

    def col(model, k):
        return np.array([rows[t][model][k] for t in used])

    print(f"{'metric':<22}{'ewma':>9}{'fhs0':>9}{'fhs1':>9}   nominal")
    print(f"{'mean cov90 (h10)':<22}{col('ewma',0).mean():>9.3f}"
          f"{col('fhs0',0).mean():>9.3f}{col('fhs1',0).mean():>9.3f}    0.900")
    print(f"{'mean cov99 (h10)':<22}{col('ewma',1).mean():>9.3f}"
          f"{col('fhs0',1).mean():>9.3f}{col('fhs1',1).mean():>9.3f}    0.990")
    # CRPS as a scale-free per-stock ratio vs EWMA (robust to price level).
    ratio0 = np.median(col('fhs0', 2) / col('ewma', 2))
    ratio1 = np.median(col('fhs1', 2) / col('ewma', 2))
    print(f"{'median CRPS ratio/ewma':<22}{1.0:>9.3f}{ratio0:>9.3f}{ratio1:>9.3f}"
          "   (<1 better)")

    # Win rates: FHS(g=0) vs EWMA.
    tail_closer = np.abs(col('fhs0', 1) - 0.99) < np.abs(col('ewma', 1) - 0.99)
    body_closer = np.abs(col('fhs0', 0) - 0.90) < np.abs(col('ewma', 0) - 0.90)
    crps_better = col('fhs0', 2) < col('ewma', 2)
    n = len(used)
    print(f"\nFHS(g=0) beats EWMA in:")
    print(f"  99% tail calibration : {tail_closer.sum():>3}/{n}  ({tail_closer.mean():.0%})")
    print(f"  90% body calibration : {body_closer.sum():>3}/{n}  ({body_closer.mean():.0%})")
    print(f"  CRPS (lower)         : {crps_better.sum():>3}/{n}  ({crps_better.mean():.0%})")

    _plot(col, used)
    print("\nsaved xsec.png")


def _plot(col, used) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    # tail coverage per stock: EWMA vs FHS(g=0)
    a1.scatter(col('ewma', 1), col('fhs0', 1), s=18, alpha=0.7)
    lim = [min(col('ewma', 1).min(), col('fhs0', 1).min()) - 0.005, 1.0]
    a1.plot(lim, lim, "k:", lw=0.8)
    a1.axhline(0.99, color="tab:green", lw=0.6, ls="--")
    a1.axvline(0.99, color="tab:green", lw=0.6, ls="--")
    a1.set_xlabel("EWMA 99% coverage"); a1.set_ylabel("FHS(g=0) 99% coverage")
    a1.set_title("per-stock tail coverage\n(above diagonal = FHS wider/closer)")
    a1.set_xlim(lim); a1.set_ylim(lim)

    labels = ["ewma", "fhs0", "fhs1"]
    data = [col(m, 1) for m in labels]
    a2.boxplot(data, tick_labels=labels, showmeans=True)
    a2.axhline(0.99, color="tab:green", lw=0.8, ls="--", label="nominal 0.99")
    a2.set_ylabel("99% coverage (h10) across stocks")
    a2.set_title("tail-coverage distribution")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("xsec.png", dpi=110)


if __name__ == "__main__":
    main()
