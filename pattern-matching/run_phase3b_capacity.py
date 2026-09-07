"""Phase 3b: capacity robustness check for the non-linear verdict.

Phase 3 used a small 1-D-per-component stack and found no non-linear structure
(AE reconstructed worse than PCA). Before we commit that negative, rule out
'the net was just too small'. We train progressively larger joint autoencoders
-- wider hidden, a second hidden layer, and a multi-unit TANH bottleneck up to
5 codes -- and compare each, at its own code dimension d, against PCA-with-d
(the optimal linear baseline). Everything is out-of-sample.

If some bigger net (0) reconstructs the held-out residual better than PCA-d, OR
(a) lifts direction IC above ~0.05 / clearly past PCA-d, the small-net verdict
was an artefact. If none do, the 'no exploitable non-linear structure' negative
is robust to capacity.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matching.autoencoder import MLPAutoencoder
from run_phase2_direction import _collect, _spearman, STRIDE, TRAIN_FRAC
from run_phase3_nonlinear import _combined_ic
from matching.universe import load_universe

H, EPOCHS = 20, 150
CONFIGS = [
    ("h32 x1  code5", [32], 5),
    ("h64 x1  code5", [64], 5),
    ("h64 x2  code5", [64, 64], 5),
    ("h64 x2  code3", [64, 64], 3),
    ("h64 x2  code1", [64, 64], 1),
]


def _var_expl(X, recon):
    ncols = X.shape[1]
    return 1 - np.var(X - recon) * ncols / (np.var(X) * ncols)


def _pca_fit(Wtr, d):
    _, _, Vt = np.linalg.svd(Wtr, full_matrices=False)
    return Vt[:d]


def evaluate(uni, L, configs):
    W, y, dcol = _collect(uni, L, H)
    tr = dcol.astype("int64") <= np.quantile(dcol.astype("int64"), TRAIN_FRAC)
    step = max(1, round(H / STRIDE))
    te = np.where(~tr)[0]
    te_s = te[::step]
    P = W.shape[1]
    rows = []
    for name, hid, d in configs:
        ae = MLPAutoencoder(P, enc_hidden=hid, code_dim=d, l2=1e-5, seed=0)
        ae.fit(W[tr], epochs=EPOCHS, seed=0)
        ae_cv = _var_expl(W[te], ae.reconstruct(W[te]))
        A = ae.encode(W)
        ae_ic, ae_z, _ = _combined_ic(A[tr], y[tr], A[te_s], y[te_s])

        comp = _pca_fit(W[tr], d)
        pca_recon = (W[te] @ comp.T) @ comp
        pca_cv = _var_expl(W[te], pca_recon)
        Ap = W @ comp.T
        pca_ic, pca_z, _ = _combined_ic(Ap[tr], y[tr], Ap[te_s], y[te_s])

        rows.append(dict(name=name, d=d, ae_cv=ae_cv, pca_cv=pca_cv,
                         ae_ic=ae_ic, ae_z=ae_z, pca_ic=pca_ic))
        print(f"  {name:16s} d={d}  recon(test) AE {ae_cv:.3f} / PCA {pca_cv:.3f}"
              f"   dir IC AE {ae_ic:+.4f}(z{ae_z:+.1f}) / PCA {pca_ic:+.4f}")
    return rows


def main():
    uni = load_universe()
    print(f"universe: {len(uni)} stocks   epochs={EPOCHS}\n")
    print("L=80 capacity sweep:")
    rows80 = evaluate(uni, 80, CONFIGS)
    print("\nL=40 (strongest-reversion scale), largest net:")
    rows40 = evaluate(uni, 40, [CONFIGS[2]])

    print("\n=== capacity verdict ===")
    struct = any(r["ae_cv"] > r["pca_cv"] + 0.005 for r in rows80 + rows40)
    beat_bar = any(abs(r["ae_ic"]) >= 0.05 for r in rows80 + rows40)
    beat_pca = any(abs(r["ae_ic"]) - abs(r["pca_ic"]) >= 0.01 for r in rows80 + rows40)
    print(f"(0) any net reconstructs test better than PCA-d  -> {'PASS' if struct else 'FAIL'}")
    print(f"(a) any net direction |IC| >= 0.05              -> {'PASS' if beat_bar else 'FAIL'}")
    print(f"(b) any net beats PCA-d direction IC by >=0.01  -> {'PASS' if beat_pca else 'FAIL'}")
    print(f"\nOVERALL: {'PASS -- capacity mattered' if (struct or beat_bar or beat_pca) else 'FAIL -- verdict robust to capacity'}")
    _plot(rows80)
    print("saved phase3b_capacity.png")


def _plot(rows):
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.bar(x - 0.2, [r["ae_cv"] for r in rows], 0.4, label="AE", alpha=0.8)
    a1.bar(x + 0.2, [r["pca_cv"] for r in rows], 0.4, label="PCA-d", alpha=0.8)
    a1.set_xticks(x); a1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    a1.set_ylabel("test reconstruction var explained"); a1.set_ylim(0, 1)
    a1.set_title("Reconstruction: AE vs PCA at equal code-dim"); a1.legend()
    a2.bar(x - 0.2, [r["ae_ic"] for r in rows], 0.4, label="AE", alpha=0.8)
    a2.bar(x + 0.2, [r["pca_ic"] for r in rows], 0.4, label="PCA-d", alpha=0.8)
    a2.axhline(0.05, color="r", ls="--", lw=0.8, label="0.05 bar")
    a2.axhline(-0.05, color="r", ls="--", lw=0.8)
    a2.set_xticks(x); a2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    a2.set_ylabel("combined OOS direction IC")
    a2.set_title("Direction IC: AE vs PCA"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("phase3b_capacity.png", dpi=110)


if __name__ == "__main__":
    main()
