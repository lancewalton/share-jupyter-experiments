"""Phase 1b: diagnosing the L=260 Keogh anomaly.

At L=260 the real-vs-surrogate top-6 subspace alignment fell to ~0.51 (it is
1.000 at L=20/80). Two explanations: (i) genuine long-horizon structure that
diffusion surrogates lack, or (ii) small-sample / heavy-overlap instability
(L=260 with stride 5 -> adjacent windows share 255/260 points, so few effectively
independent windows, and the low-variance components 4-6 are noisy).

Three tests separate them:
  A. alignment vs number of components (k=2..6): if the top-2 still align but the
     tail drags it down, the misalignment lives in noisy low-variance modes.
  B. real split-half self-alignment: pool two DISJOINT halves of the universe and
     align their own subspaces. If real-vs-real is also ~0.5 at L=260, the
     subspace is simply unstable -> instability, not structure. If real-vs-real
     stays high while real-vs-surrogate is low, the structure is genuine.
  C. stride sensitivity: reducing overlap (larger stride) should not matter if
     structure is genuine.
"""
from __future__ import annotations

import numpy as np

from matching.represent import vol_scaled_windows
from matching.universe import iid_surrogate, load_universe, shuffle_surrogate

P, LOOKBACK = 32, 60
RNG = np.random.default_rng(0)


def pool(series, L, stride, transform=None):
    Ws = []
    for _, logP, _ in series:
        p = transform(logP, RNG) if transform else logP
        W, _, _, _ = vol_scaled_windows(p, L=L, P=P, stride=stride, lookback=LOOKBACK)
        ok = np.isfinite(W).all(axis=1)
        if ok.any():
            Ws.append(W[ok])
    return np.vstack(Ws)


def comps(W, k):
    _, _, Vt = np.linalg.svd(W, full_matrices=False)
    sh = Vt[:k].copy()
    for i in range(k):
        if sh[i, -1] < sh[i, 0]:
            sh[i] *= -1
    return sh


def align(A, B):
    return float(np.linalg.svd(A @ B.T, compute_uv=False).mean())


def main():
    uni = load_universe()
    print(f"universe: {len(uni)} stocks\n")

    # --- Test A: alignment vs k, at L=80 (control) and L=260 ---
    print("A. real-vs-surrogate alignment by #components:")
    print("     L    k    vs-iid   vs-shuffle")
    for L in (80, 260):
        real = pool(uni, L, 5)
        iidW = pool(uni, L, 5, iid_surrogate)
        shufW = pool(uni, L, 5, shuffle_surrogate)
        for k in (2, 3, 4, 6):
            print(f"   {L:3d}   {k}    {align(comps(real, k), comps(iidW, k)):.3f}"
                  f"      {align(comps(real, k), comps(shufW, k)):.3f}")

    # --- Test B: split-half self-alignment (k=6) ---
    print("\nB. split-half self-alignment (top-6):")
    ha, hb = uni[0::2], uni[1::2]
    for L in (80, 260):
        rA, rB = comps(pool(ha, L, 5), 6), comps(pool(hb, L, 5), 6)
        sA = comps(pool(ha, L, 5, shuffle_surrogate), 6)
        sB = comps(pool(hb, L, 5, shuffle_surrogate), 6)
        print(f"   L={L:3d}   real-vs-real {align(rA, rB):.3f}   "
              f"surrogate-vs-surrogate {align(sA, sB):.3f}   "
              f"realA-vs-surrA {align(rA, sA):.3f}")

    # --- Test C: stride sensitivity at L=260 (top-6) ---
    print("\nC. L=260 real-vs-shuffle alignment by stride (overlap):")
    for stride in (5, 20, 60):
        real = pool(uni, 260, stride)
        shufW = pool(uni, 260, stride, shuffle_surrogate)
        print(f"   stride={stride:2d}  windows={len(real):>6d}  "
              f"align={align(comps(real, 6), comps(shufW, 6)):.3f}")


if __name__ == "__main__":
    main()
