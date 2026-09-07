"""Phase 1 gate: is unsupervised subsequence clustering meaningful on our data?

Keogh & Lin (2005) showed sliding-window subsequence cluster centres are roughly
data-independent (they approach sinusoids). We test that directly: cluster the
z-normalised price-shape windows of FTSE, a matched RANDOM WALK, and SHUFFLED
FTSE returns, and check whether the centroid *sets* coincide and how sinusoidal
they are. If they coincide -> unsupervised discovery is meaningless here -> pivot
to the supervised (forward-label) framing.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from mc.data import close_array
from mc.returns import log_returns
from patterns.windows import log_price, extract_windows

L, K, STRIDE = 30, 8, 1
RNG = np.random.default_rng(0)


def _centroids(logP):
    W, _, _ = extract_windows(logP, L=L, stride=STRIDE, znorm=True)
    km = KMeans(n_clusters=K, n_init=5, random_state=0).fit(W)
    C = km.cluster_centers_
    # order clusters by size for stable display
    counts = np.bincount(km.labels_, minlength=K)
    return C[np.argsort(-counts)], np.sort(counts)[::-1], W.shape[0]


def _match_similarity(A, B):
    """Mean best-match correlation of each centroid in A to some centroid in B."""
    sims = []
    for a in A:
        sims.append(max(np.corrcoef(a, b)[0, 1] for b in B))
    return float(np.mean(sims))


def _sinusoidality(C):
    """Mean R^2 of centroids fit by the first 3 sine/cosine harmonics."""
    t = np.linspace(0, 1, C.shape[1])
    basis = [np.ones_like(t)]
    for k in range(1, 4):
        basis += [np.sin(2 * np.pi * k * t), np.cos(2 * np.pi * k * t)]
    Bm = np.column_stack(basis)
    r2 = []
    for c in C:
        beta, *_ = np.linalg.lstsq(Bm, c, rcond=None)
        pred = Bm @ beta
        ss_res = ((c - pred) ** 2).sum(); ss_tot = ((c - c.mean()) ** 2).sum()
        r2.append(1 - ss_res / ss_tot)
    return float(np.mean(r2))


def main():
    r = log_returns(close_array())
    logP_ftse = log_price(np.exp(np.concatenate([[0.0], np.cumsum(r)])))  # rebuild path

    # matched random walk: gaussian returns, FTSE daily vol
    rw = RNG.normal(0, r.std(), size=len(r))
    logP_rw = np.concatenate([[0.0], np.cumsum(rw)])

    # shuffled FTSE returns: same distribution, temporal structure destroyed
    sh = RNG.permutation(r)
    logP_sh = np.concatenate([[0.0], np.cumsum(sh)])

    C_ftse, n_ftse, m = _centroids(logP_ftse)
    C_rw, _, _ = _centroids(logP_rw)
    C_sh, _, _ = _centroids(logP_sh)

    print(f"L={L}, K={K}, windows≈{m}\n")
    print("Cross-source centroid similarity (mean best-match correlation):")
    print(f"  FTSE vs random walk    : {_match_similarity(C_ftse, C_rw):.3f}")
    print(f"  FTSE vs shuffled FTSE  : {_match_similarity(C_ftse, C_sh):.3f}")
    print(f"  random walk vs shuffled: {_match_similarity(C_rw, C_sh):.3f}\n")
    print("Sinusoidality of centroids (mean R^2 vs 3 harmonics; ~1 => just waves):")
    print(f"  FTSE       : {_sinusoidality(C_ftse):.3f}")
    print(f"  random walk: {_sinusoidality(C_rw):.3f}")
    print(f"  shuffled   : {_sinusoidality(C_sh):.3f}")

    _plot(C_ftse, C_rw, C_sh)
    print("\nsaved phase1_meaningfulness.png")


def _plot(C_ftse, C_rw, C_sh):
    fig, axes = plt.subplots(3, K, figsize=(2 * K, 6), sharex=True, sharey=True)
    for row, (C, name) in enumerate([(C_ftse, "FTSE"), (C_rw, "random walk"),
                                     (C_sh, "shuffled FTSE")]):
        for j in range(K):
            axes[row, j].plot(C[j], color="tab:blue")
            axes[row, j].set_yticks([])
            if j == 0:
                axes[row, j].set_ylabel(name, fontsize=9)
            if row == 0:
                axes[row, j].set_title(f"cluster {j+1}", fontsize=8)
    fig.suptitle(f"Unsupervised window-cluster centroids (L={L}, K={K}) — "
                 "same shapes regardless of source => meaningless")
    fig.tight_layout()
    fig.savefig("phase1_meaningfulness.png", dpi=110)


if __name__ == "__main__":
    main()
