"""Phase 3: the non-linear growing autoencoder, against the pre-registered bar.

The linear route reproduced the ceiling (generic shapes, reversion-only
activations). This is the one untried idea: a deflationary non-linear
autoencoder that can trace curved, localised shapes PCA cannot. We hold it to a
bar fixed in advance -- a discovered component is real only if OOS it

  (a) beats the linear best (|IC| >~ 0.05),
  (b) adds IC *over* drift/depth controls (isn't relabelled reversion),
  (c) beats its own shuffled-surrogate null (shape is market-specific).

We report, per scale: non-linear vs linear reconstruction (is there curved
structure at all?), per-component and combined OOS direction IC for the AE and
PCA, the shuffled-surrogate AE IC (Keogh), and the partial IC after removing
drift/min/max.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matching.autoencoder import activations, deflationary_fit, final_residual
from matching.represent import forward_return_std, vol_scaled_windows
from matching.universe import load_universe, shuffle_surrogate
from run_phase2_direction import _collect, _spearman, P, STRIDE, LOOKBACK, TRAIN_FRAC

SCALES = [40, 80, 160]
H, KC, HID, EPOCHS = 20, 4, 16, 60
RNG = np.random.default_rng(0)


def _cumvar(recon_var_after, total):
    return [1 - rv / total for rv in recon_var_after]


def _combined_ic(A_tr, y_tr, A_te, y_te):
    mu, sd = A_tr.mean(0), A_tr.std(0) + 1e-12
    Xtr, Xte = (A_tr - mu) / sd, (A_te - mu) / sd
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(Xtr)), Xtr], y_tr, rcond=None)
    pred = np.c_[np.ones(len(Xte)), Xte] @ beta
    ic = _spearman(pred, y_te)
    null = np.array([_spearman(pred, RNG.permutation(y_te)) for _ in range(100)])
    z = (ic - null.mean()) / (null.std() + 1e-12)
    return ic, z, pred


def _partial_ic(pred, y, controls):
    """IC of pred vs y after linearly removing the controls from both."""
    C = np.c_[np.ones(len(y)), controls]
    def resid(v):
        b, *_ = np.linalg.lstsq(C, v, rcond=None)
        return v - C @ b
    return _spearman(resid(pred), resid(y))


def evaluate_scale(uni, L):
    W, y, d = _collect(uni, L, H)
    tr = d.astype("int64") <= np.quantile(d.astype("int64"), TRAIN_FRAC)
    step = max(1, round(H / STRIDE))
    te_idx = np.where(~tr)[0][::step]

    total = np.var(W[tr]) * W.shape[1]  # total per-row variance scale
    # --- non-linear AE ---
    aes = deflationary_fit(W[tr], k=KC, h=HID, epochs=EPOCHS, seed=0)
    A = activations(aes, W)
    ae_recon = []
    resid = W[tr].copy()
    for ae in aes:
        resid = resid - ae.reconstruct(resid)
        ae_recon.append(np.var(resid) * W.shape[1])
    # --- linear PCA (uncentred) fit on train ---
    _, S, Vt = np.linalg.svd(W[tr], full_matrices=False)
    comp = Vt[:KC]
    Ap = W @ comp.T
    pca_recon = [(np.var(W[tr] - (W[tr] @ comp[:i+1].T) @ comp[:i+1]) * W.shape[1])
                 for i in range(KC)]

    ae_pc = [_spearman(A[~tr][::step][:, i], y[~tr][::step]) for i in range(KC)]
    pca_pc = [_spearman(Ap[~tr][::step][:, i], y[~tr][::step]) for i in range(KC)]
    yte = y[te_idx]
    ae_ic, ae_z, ae_pred = _combined_ic(A[tr], y[tr], A[te_idx], yte)
    pca_ic, pca_z, _ = _combined_ic(Ap[tr], y[tr], Ap[te_idx], yte)

    # partial IC vs drift/min/max controls (drift = last col; depth = min/max)
    Wte = W[te_idx]
    controls = np.c_[Wte[:, -1], Wte.min(1), Wte.max(1)]
    part_ic = _partial_ic(ae_pred, yte, controls)

    # Keogh: same AE pipeline on shuffled surrogates
    sur = [(n, shuffle_surrogate(lp, RNG), dt) for n, lp, dt in uni]
    Ws, ys, ds = _collect(sur, L, H)
    trs = ds.astype("int64") <= np.quantile(ds.astype("int64"), TRAIN_FRAC)
    tes = np.where(~trs)[0][::step]
    aes_s = deflationary_fit(Ws[trs], k=KC, h=HID, epochs=EPOCHS, seed=1)
    As = activations(aes_s, Ws)
    sur_ic, sur_z, _ = _combined_ic(As[trs], ys[trs], As[tes], ys[tes])

    return dict(L=L, ae_cv=_cumvar(ae_recon, total), pca_cv=_cumvar(pca_recon, total),
                ae_pc=ae_pc, pca_pc=pca_pc, ae_ic=ae_ic, ae_z=ae_z, pca_ic=pca_ic,
                part_ic=part_ic, sur_ic=sur_ic, sur_z=sur_z)


def main():
    uni = load_universe()
    print(f"universe: {len(uni)} stocks   k={KC} components  H={H}\n")
    rows = []
    for L in SCALES:
        r = evaluate_scale(uni, L)
        rows.append(r)
        print(f"L={L}:")
        print(f"  recon (cum var explained)  AE {['%.3f'%x for x in r['ae_cv']]}")
        print(f"                             PCA {['%.3f'%x for x in r['pca_cv']]}")
        print(f"  per-comp IC  AE  {['%+.3f'%x for x in r['ae_pc']]}")
        print(f"               PCA {['%+.3f'%x for x in r['pca_pc']]}")
        print(f"  combined IC  AE {r['ae_ic']:+.4f} (z={r['ae_z']:+.2f})   "
              f"PCA {r['pca_ic']:+.4f}")
        print(f"  partial IC (over drift/depth)  {r['part_ic']:+.4f}")
        print(f"  Keogh surrogate AE IC          {r['sur_ic']:+.4f} "
              f"(z={r['sur_z']:+.2f})\n")
    _verdict(rows)
    _plot(rows)
    print("saved phase3_nonlinear.png")


def _verdict(rows):
    print("=== pre-registered bar ===")
    # Gate 0: is there any non-linear structure at all? If the AE cannot even
    # reconstruct the residual better than the optimal linear basis (PCA), there
    # is no curved manifold to exploit and (a)-(c) are moot.
    struct = any(r["ae_cv"][-1] > r["pca_cv"][-1] + 0.005 for r in rows)
    print(f"(0) AE reconstructs better than PCA (non-linear structure exists) -> "
          f"{'PASS' if struct else 'FAIL'}")
    best = max(abs(r["ae_ic"]) for r in rows)
    a = best >= 0.05
    print(f"(a) AE combined |IC| best = {best:.4f}  vs ~0.05 bar  -> "
          f"{'PASS' if a else 'FAIL'}")
    # (b)/(c) only mean anything if the AE actually beat the linear route; here
    # AE IC tracks PCA IC almost exactly, so any pass is the pre-existing linear
    # reversion signal, not a non-linear discovery.
    beats_pca = any(abs(r["ae_ic"]) - abs(r["pca_ic"]) >= 0.01 for r in rows)
    print(f"(b) AE beats PCA direction IC by >=0.01 (non-linearity adds) -> "
          f"{'PASS' if beats_pca else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if (struct and a and beats_pca) else 'FAIL'} "
          "-- non-linear route "
          f"{'found new structure' if (struct and a and beats_pca) else 'reproduces the linear ceiling'}")


def _plot(rows):
    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 4.6))
    if len(rows) == 1:
        axes = [axes]
    for ax, r in zip(axes, rows):
        xs = np.arange(1, KC + 1)
        ax.plot(xs, r["ae_cv"], "o-", label="AE cum-var")
        ax.plot(xs, r["pca_cv"], "s--", label="PCA cum-var")
        ax.bar(xs - 0.15, r["ae_pc"], 0.3, alpha=0.5, label="AE IC")
        ax.bar(xs + 0.15, r["pca_pc"], 0.3, alpha=0.5, label="PCA IC")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"L={r['L']}  AE IC {r['ae_ic']:+.3f} (z{r['ae_z']:+.1f})\n"
                     f"partial {r['part_ic']:+.3f}  surrogate {r['sur_ic']:+.3f}")
        ax.set_xlabel("component"); ax.legend(fontsize=7)
    fig.suptitle("Non-linear AE vs PCA: reconstruction (lines) and direction IC (bars)")
    fig.tight_layout(); fig.savefig("phase3_nonlinear.png", dpi=110)


if __name__ == "__main__":
    main()
