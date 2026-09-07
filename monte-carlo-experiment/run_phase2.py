"""Phase 2: age-weighted and block bootstraps vs the EWMA benchmark.

Tests whether recency weighting closes the Phase-1 body over-coverage and the
stationary block bootstrap closes the 99% tail gap, plus their combination.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.data import close_array
from mc.returns import log_returns
from mc.forecasters import (
    gaussian_ewma,
    bootstrap_iid,
    bootstrap_age_weighted,
    bootstrap_stationary,
)
from mc.walkforward import evaluate, NOMINAL_INTERVALS

T = 10
MIN_HISTORY = 500
STRIDE = 5

forecasters = {
    "gaussian_ewma(0.94)": gaussian_ewma(lam=0.94, min_obs=MIN_HISTORY),
    "bootstrap_iid(500)": bootstrap_iid(N=500, n_sims=10_000, seed=0),
    "age_wt(hl=20)": bootstrap_age_weighted(N=500, half_life=20, n_sims=10_000, seed=0),
    "block(L=10)": bootstrap_stationary(N=500, mean_block=10, n_sims=10_000, seed=0),
    "age_wt+block(hl=20,L=10)": bootstrap_stationary(
        N=500, mean_block=10, half_life=20, n_sims=10_000, seed=0
    ),
}


def main() -> None:
    r = log_returns(close_array())
    print(f"log returns: {len(r)}   T={T}  min_history={MIN_HISTORY}  stride={STRIDE}\n")

    results = {}
    for name, f in forecasters.items():
        res = evaluate(r, f, T=T, min_history=MIN_HISTORY, stride=STRIDE)
        results[name] = res
        print(f"=== {name}   (n_origins={res.n_origins}) ===")
        print("  coverage @ horizon 1 / 5 / 10  (nominal -> empirical):")
        for p in NOMINAL_INTERVALS:
            v = res.coverage[p]
            print(f"    {p:>4.0%}:  {v[0]:.3f}  {v[4]:.3f}  {v[9]:.3f}")
        print(f"  mean CRPS  h1={res.mean_crps[0]:.5f}  "
              f"h5={res.mean_crps[4]:.5f}  h10={res.mean_crps[9]:.5f}\n")

    _plot(results)
    print("saved phase2_pit.png")


def _plot(results: dict) -> None:
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 7))
    for j, (name, res) in enumerate(results.items()):
        for h, colour in ((1, "tab:blue"), (10, "tab:red")):
            pit = res.pit[:, h - 1]
            ax = axes[0, j]
            ax.hist(pit, bins=20, range=(0, 1), density=True, histtype="step",
                    color=colour, label=f"h={h}")
            ax.axhline(1.0, color="k", lw=0.6, ls=":")
            ax.set_title(f"PIT\n{name}", fontsize=8)
            ax.set_xlabel("PIT"); ax.set_ylim(0, 2.2)
            if j == 0:
                ax.set_ylabel("density")
            ax.legend(fontsize=7)
        noms = list(NOMINAL_INTERVALS)
        axr = axes[1, j]
        for h, colour in ((1, "tab:blue"), (10, "tab:red")):
            emp = [res.coverage[p][h - 1] for p in noms]
            axr.plot(noms, emp, "o-", color=colour, label=f"h={h}")
        axr.plot([0, 1], [0, 1], "k:", lw=0.8)
        axr.set_xlim(0.4, 1.0); axr.set_ylim(0.4, 1.02)
        axr.set_title("reliability", fontsize=8)
        axr.set_xlabel("nominal")
        if j == 0:
            axr.set_ylabel("empirical")
        axr.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("phase2_pit.png", dpi=110)


if __name__ == "__main__":
    main()
