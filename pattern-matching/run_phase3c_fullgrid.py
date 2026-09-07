"""Phase 3c: the exhaustive scale map for the non-linear verdict.

Runs the deflationary non-linear autoencoder vs PCA at EVERY length L in 20..260
(step 10), recording, out-of-sample: final reconstruction variance explained
(AE and PCA) and combined direction IC (AE and PCA). The point is a complete
on-record map -- if the AE ever reconstructs better than PCA, or lifts direction
IC past ~0.05 / clearly beyond PCA, it will show here. Phase 1 (shapes generic)
and Phase 3/3b already make that unlikely; this leaves no scale unchecked.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matching.autoencoder import activations, deflationary_fit
from run_phase2_direction import _collect, STRIDE, TRAIN_FRAC
from run_phase3_nonlinear import _combined_ic
from matching.universe import load_universe

SCALES = list(range(20, 261, 10))
H, KC, HID, EPOCHS = 20, 4, 16, 60


def _var_expl(X, recon):
    return 1 - np.var(X - recon) / np.var(X)


def evaluate(uni, L):
    W, y, dcol = _collect(uni, L, H)
    tr = dcol.astype("int64") <= np.quantile(dcol.astype("int64"), TRAIN_FRAC)
    step = max(1, round(H / STRIDE))
    te = np.where(~tr)[0]; te_s = te[::step]

    aes = deflationary_fit(W[tr], k=KC, h=HID, epochs=EPOCHS, seed=0)
    resid = W[te].copy()
    for ae in aes:
        resid = resid - ae.reconstruct(resid)
    ae_cv = _var_expl(W[te], W[te] - resid)
    A = activations(aes, W)
    ae_ic, ae_z, _ = _combined_ic(A[tr], y[tr], A[te_s], y[te_s])

    _, _, Vt = np.linalg.svd(W[tr], full_matrices=False)
    comp = Vt[:KC]
    pca_cv = _var_expl(W[te], (W[te] @ comp.T) @ comp)
    Ap = W @ comp.T
    pca_ic, _, _ = _combined_ic(Ap[tr], y[tr], Ap[te_s], y[te_s])
    return ae_cv, pca_cv, ae_ic, ae_z, pca_ic


def main():
    uni = load_universe()
    print(f"universe: {len(uni)} stocks   full grid {SCALES[0]}..{SCALES[-1]}\n")
    aecv, pcacv, aeic, pcaic = [], [], [], []
    for L in SCALES:
        a_cv, p_cv, a_ic, a_z, p_ic = evaluate(uni, L)
        aecv.append(a_cv); pcacv.append(p_cv); aeic.append(a_ic); pcaic.append(p_ic)
        print(f"  L={L:3d}  recon AE {a_cv:.3f} / PCA {p_cv:.3f}  (gap {a_cv-p_cv:+.3f})"
              f"   dir IC AE {a_ic:+.4f}(z{a_z:+.1f}) / PCA {p_ic:+.4f}")

    gap = np.array(aecv) - np.array(pcacv)
    print(f"\nreconstruction gap AE-PCA: max {gap.max():+.4f} at L={SCALES[int(gap.argmax())]}"
          f"  (>0 anywhere = non-linear structure)")
    print(f"AE direction |IC|: max {max(abs(np.array(aeic))):.4f} (bar 0.05)")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(SCALES, aecv, "o-", label="AE"); a1.plot(SCALES, pcacv, "s--", label="PCA")
    a1.set_xlabel("L (days)"); a1.set_ylabel("test reconstruction var explained")
    a1.set_title("Reconstruction across all scales"); a1.legend()
    a2.axhline(0, color="k", lw=0.6)
    a2.axhline(0.05, color="r", ls="--", lw=0.8); a2.axhline(-0.05, color="r", ls="--", lw=0.8)
    a2.plot(SCALES, aeic, "o-", label="AE"); a2.plot(SCALES, pcaic, "s--", label="PCA")
    a2.set_xlabel("L (days)"); a2.set_ylabel("combined OOS direction IC")
    a2.set_title("Direction IC across all scales (±0.05 bar)"); a2.legend()
    fig.tight_layout(); fig.savefig("phase3c_fullgrid.png", dpi=110)
    print("saved phase3c_fullgrid.png")


if __name__ == "__main__":
    main()
