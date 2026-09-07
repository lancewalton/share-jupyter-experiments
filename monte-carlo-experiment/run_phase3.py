"""Phase 3: is the backward-simulation distrust score a useful trust signal?

Gate: does per-origin backward distrust predict forward miscalibration of the
regime-blind IID(500) forecaster -- and does it beat the trivial control
"recent vol / window vol"? Uses only past data for the score, only future data
for the outcome (no leakage).
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.data import close_array
from mc.returns import log_returns, realised_cumulative
from mc.forecasters import bootstrap_iid
from mc.backward import backward_distrust, forward_miscalibration
from mc.scoring import crps_from_quantiles, interval_hit
from mc.walkforward import DENSE_LEVELS

T = 10
B = 10
N = 500
MIN_HISTORY = N
STRIDE = 2


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _idx(level: float) -> int:
    return int(np.argmin(np.abs(DENSE_LEVELS - level)))


def main() -> None:
    r = log_returns(close_array())
    f = bootstrap_iid(N=N, n_sims=10_000, seed=0)
    lev = DENSE_LEVELS

    first, last = MIN_HISTORY - 1, len(r) - T - 1
    origins = range(max(first, B - 1), last + 1, STRIDE)

    distrust, miscal, crps, volratio = [], [], [], []
    hit90, hit99 = [], []  # forward coverage at horizon 10
    i_lo90, i_hi90 = _idx(0.05), _idx(0.95)
    i_lo99, i_hi99 = _idx(0.005), _idx(0.995)

    for t in origins:
        hist = r[: t + 1]
        q = f(hist, max(T, B), lev)
        y = realised_cumulative(r, t, T)
        distrust.append(backward_distrust(q, r, t, B, lev))
        miscal.append(forward_miscalibration(q, y, lev))
        crps.append(float(crps_from_quantiles(q, y, lev).mean()))
        window = hist[-N:]
        volratio.append(float(np.std(window[-B:]) / np.std(window)))
        hit90.append(bool(interval_hit(q[i_lo90, T - 1], q[i_hi90, T - 1], y[T - 1])))
        hit99.append(bool(interval_hit(q[i_lo99, T - 1], q[i_hi99, T - 1], y[T - 1])))

    distrust = np.array(distrust); miscal = np.array(miscal)
    crps = np.array(crps); volratio = np.array(volratio)
    hit90 = np.array(hit90); hit99 = np.array(hit99)
    n = len(distrust)
    print(f"origins: {n}   (overlapping; treat significance qualitatively)\n")

    print("Spearman correlations (forward outcome vs signal):")
    print(f"  distrust  vs forward-miscalibration : {_spearman(distrust, miscal):+.3f}")
    print(f"  vol-ratio vs forward-miscalibration : {_spearman(volratio, miscal):+.3f}")
    print(f"  distrust  vs forward-CRPS           : {_spearman(distrust, crps):+.3f}")
    print(f"  vol-ratio vs forward-CRPS           : {_spearman(volratio, crps):+.3f}")
    print(f"  distrust  vs vol-ratio (redundancy) : {_spearman(distrust, volratio):+.3f}\n")

    # Tercile stratification by distrust.
    order = np.argsort(distrust)
    terciles = np.array_split(order, 3)
    labels = ["low distrust", "mid distrust", "high distrust"]
    print("Forward outcome by distrust tercile:")
    print(f"  {'tercile':<14} {'CRPS':>8} {'cov90':>7} {'cov99':>7} {'miscal':>7}")
    for lab, ix in zip(labels, terciles):
        print(f"  {lab:<14} {crps[ix].mean():>8.5f} {hit90[ix].mean():>7.3f} "
              f"{hit99[ix].mean():>7.3f} {miscal[ix].mean():>7.3f}")

    _plot(distrust, miscal, crps, terciles, labels, hit99)
    print("\nsaved phase3.png")


def _plot(distrust, miscal, crps, terciles, labels, hit99) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].hexbin(distrust, miscal, gridsize=30, cmap="viridis", mincnt=1)
    axes[0].set_xlabel("backward distrust (past only)")
    axes[0].set_ylabel("forward miscalibration")
    axes[0].set_title("distrust vs forward miscalibration")

    means = [crps[ix].mean() for ix in terciles]
    axes[1].bar(labels, means, color=["tab:green", "tab:orange", "tab:red"])
    axes[1].set_ylabel("mean forward CRPS")
    axes[1].set_title("forward CRPS by distrust tercile")
    axes[1].tick_params(axis="x", labelrotation=15)

    breach = [1 - hit99[ix].mean() for ix in terciles]
    axes[2].axhline(0.01, color="k", ls=":", label="nominal 1%")
    axes[2].bar(labels, breach, color=["tab:green", "tab:orange", "tab:red"])
    axes[2].set_ylabel("99% breach rate (h=10)")
    axes[2].set_title("tail breaches by distrust tercile")
    axes[2].tick_params(axis="x", labelrotation=15)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("phase3.png", dpi=110)


if __name__ == "__main__":
    main()
