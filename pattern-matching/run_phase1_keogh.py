"""Phase 1: the Keogh null -- are the linear residual shapes market-specific?

Phase 0 showed the uncentred SVD extracts a drift ramp (comp 1) then sinusoids
(comp 2+). Keogh & Lin (2005) warn those sinusoids may be artefacts of drift +
diffusion, identical for any series. We test it directly: build, for every real
series, two matched surrogates --

  * iid     -- Gaussian random walk with the same drift and volatility;
  * shuffle -- the real returns permuted (same distribution, no time structure);

pool each source's windows, take its SVD components, and measure how close the
REAL components are to the surrogate ones. If they coincide, the linear route's
'patterns' carry no market information -- confirming we must go non-linear.

Similarity is reported two ways: per-component |cosine| (sign-invariant), and
the alignment of the whole top-k subspace (mean cosine of principal angles),
which is robust to rotation among near-degenerate components.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matching.represent import vol_scaled_windows
from matching.universe import iid_surrogate, load_universe, shuffle_surrogate

SCALES = [20, 80, 260]
REP = 80
P, STRIDE, LOOKBACK, K = 32, 5, 60, 6
RNG = np.random.default_rng(0)


def _pool(series_list, L, transform=None):
    Ws = []
    for _, logP, _ in series_list:
        p = transform(logP, RNG) if transform else logP
        W, _, _, drift = vol_scaled_windows(p, L=L, P=P, stride=STRIDE, lookback=LOOKBACK)
        ok = np.isfinite(W).all(axis=1)
        if ok.any():
            Ws.append(W[ok])
    return np.vstack(Ws)


def _components(W, k=K):
    _, _, Vt = np.linalg.svd(W, full_matrices=False)
    sh = Vt[:k].copy()
    for i in range(k):
        if sh[i, -1] < sh[i, 0]:
            sh[i] *= -1
    return sh


def _subspace_alignment(A, B):
    """Mean cosine of principal angles between row-spaces of A and B (in [0,1])."""
    s = np.linalg.svd(A @ B.T, compute_uv=False)
    return float(s.mean())


def main():
    uni = load_universe()
    print(f"universe: {len(uni)} stocks\n")

    print("top-6 subspace alignment (1.0 = identical shape-space):")
    for L in SCALES:
        real = _components(_pool(uni, L))
        iidc = _components(_pool(uni, L, iid_surrogate))
        shuf = _components(_pool(uni, L, shuffle_surrogate))
        print(f"  L={L:3d}   real-vs-iid {_subspace_alignment(real, iidc):.3f}"
              f"   real-vs-shuffle {_subspace_alignment(real, shuf):.3f}")

    # detailed per-component comparison at the representative scale
    real = _components(_pool(uni, REP))
    iidc = _components(_pool(uni, REP, iid_surrogate))
    shuf = _components(_pool(uni, REP, shuffle_surrogate))
    print(f"\nper-component |cosine| at L={REP}:")
    print("  comp   real-vs-iid   real-vs-shuffle")
    for i in range(K):
        ci = abs(float(real[i] @ iidc[i]))
        cs = abs(float(real[i] @ shuf[i]))
        print(f"   c{i+1}      {ci:.3f}         {cs:.3f}")

    _plot(real, iidc, shuf)
    print("\nsaved phase1_keogh.png")


def _plot(real, iidc, shuf):
    xg = np.linspace(0, 1, P)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for i, ax in enumerate(axes.ravel()):
        ax.plot(xg, real[i], lw=2.2, label="real")
        ax.plot(xg, iidc[i] * np.sign(iidc[i] @ real[i]), "--", label="iid RW")
        ax.plot(xg, shuf[i] * np.sign(shuf[i] @ real[i]), ":", lw=2, label="shuffled")
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_title(f"Component {i+1}")
        ax.set_xlabel("normalised time through window")
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Real vs surrogate SVD components at L={REP} "
                 "(overlap = shapes are generic, not market structure)", y=1.0)
    fig.tight_layout(); fig.savefig("phase1_keogh.png", dpi=110)


if __name__ == "__main__":
    main()
