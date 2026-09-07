"""Phase 3d: is the AE>PCA reconstruction bump at L~220 structure or instability?

The full grid (phase 3c) showed the non-linear AE reconstructing the held-out
data BETTER than PCA at L=200-250 (gap +0.076 at L=220) -- tripping criterion
(0). This coincides exactly with the long-scale instability diagnosed in phase
1b. The instability hypothesis makes a sharp prediction: PCA is *overfitting* an
unstable long-scale TRAIN subspace (few effectively-independent 260-day windows
under stride-5 overlap), so its train->test reconstruction gap is large and its
apparent loss to the regularised AE shrinks once we reduce overlap (larger
stride -> more independent windows). Genuine non-linear structure would instead
persist with independent windows.

We test at L=220 across strides. If PCA's test reconstruction recovers and the
AE-PCA gap collapses as stride grows, it is instability, not structure.
"""
from __future__ import annotations

import numpy as np

from matching.autoencoder import deflationary_fit
from matching.represent import vol_scaled_windows
from matching.universe import load_universe
from run_phase2_direction import TRAIN_FRAC

L, KC, HID, EPOCHS, P, LOOKBACK = 220, 4, 16, 60, 32, 60


def _collect(uni, stride):
    Ws, ds = [], []
    for _, logP, dates in uni:
        W, ends, _, _ = vol_scaled_windows(logP, L=L, P=P, stride=stride, lookback=LOOKBACK)
        ok = np.isfinite(W).all(axis=1)
        if ok.any():
            Ws.append(W[ok]); ds.append(dates[ends[ok]])
    W = np.vstack(Ws); d = np.concatenate(ds)
    order = np.argsort(d)
    return W[order], d[order]


def _ve(X, recon):
    return 1 - np.var(X - recon) / np.var(X)


def main():
    uni = load_universe()
    print(f"L={L}: is AE>PCA reconstruction structure or instability?\n")
    print("stride  windows   PCA_train  PCA_test  (overfit gap)   AE_test   AE-PCA(test)")
    for stride in (5, 10, 20, 40, 60):
        W, d = _collect(uni, stride)
        tr = d.astype("int64") <= np.quantile(d.astype("int64"), TRAIN_FRAC)
        _, _, Vt = np.linalg.svd(W[tr], full_matrices=False)
        comp = Vt[:KC]
        pca_tr = _ve(W[tr], (W[tr] @ comp.T) @ comp)
        pca_te = _ve(W[~tr], (W[~tr] @ comp.T) @ comp)
        aes = deflationary_fit(W[tr], k=KC, h=HID, epochs=EPOCHS, seed=0)
        resid = W[~tr].copy()
        for ae in aes:
            resid = resid - ae.reconstruct(resid)
        ae_te = _ve(W[~tr], W[~tr] - resid)
        print(f"  {stride:3d}  {len(W):>7d}    {pca_tr:.3f}     {pca_te:.3f}   "
              f"({pca_tr-pca_te:+.3f})       {ae_te:.3f}     {ae_te-pca_te:+.3f}")


if __name__ == "__main__":
    main()
