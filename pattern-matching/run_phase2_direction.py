"""Phase 2: does the linear representation predict direction?

Phase 1 killed the component *shapes* as market structure. But the shapes are
just a basis; the question that survives is whether the *activations* -- how
strongly each mode is present in a given window -- forecast the forward return.
Component 1's activation is essentially trailing drift (a momentum/reversion
probe); component 2's is a depth/dip feature (the sibling project's best signal
was depth). So this is the honest 'shot at direction' with the linear route.

Discipline (matching the sibling benchmarks): strict temporal split; components
fit on TRAIN windows only; vol-standardised forward-return label; pooled
cross-sectional rank-IC on TEST; test thinned so sampled labels don't overlap;
a label-permutation null calibrates zero.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matching.represent import forward_return_std, vol_scaled_windows
from matching.universe import load_universe

P, STRIDE, LOOKBACK, K = 32, 5, 60, 6
TRAIN_FRAC = 0.6
GRID = [(20, 10), (40, 10), (80, 20), (160, 20)]
REP = (80, 20)
RNG = np.random.default_rng(0)


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def _collect(uni, L, H):
    Ws, ys, ds = [], [], []
    for _, logP, dates in uni:
        W, ends, starts, _ = vol_scaled_windows(
            logP, L=L, P=P, stride=STRIDE, lookback=LOOKBACK)
        y = forward_return_std(logP, ends, H=H, lookback=LOOKBACK)
        ok = np.isfinite(W).all(axis=1) & np.isfinite(y)
        if ok.any():
            Ws.append(W[ok]); ys.append(y[ok]); ds.append(dates[ends[ok]])
    W = np.vstack(Ws); y = np.concatenate(ys); d = np.concatenate(ds)
    order = np.argsort(d)
    return W[order], y[order], d[order]


def evaluate(uni, L, H, verbose=False):
    W, y, d = _collect(uni, L, H)
    cut = np.quantile(d.astype("int64"), TRAIN_FRAC)
    tr = d.astype("int64") <= cut
    # components from TRAIN only (uncentred, matching phase 0)
    _, _, Vt = np.linalg.svd(W[tr], full_matrices=False)
    comp = Vt[:K]
    for i in range(K):
        if comp[i, -1] < comp[i, 0]:
            comp[i] *= -1
    A = W @ comp.T                       # activations for all windows
    Atr, Ate = A[tr], A[~tr]
    ytr, yte = y[tr], y[~tr]

    # thin test so sampled forward-labels don't overlap in time
    step = max(1, round(H / STRIDE))
    sub = np.arange(0, len(yte), step)
    Ate_s, yte_s = Ate[sub], yte[sub]

    per_comp = [_spearman(Ate_s[:, i], yte_s) for i in range(K)]

    # combined linear predictor: standardise on train, OLS, predict test
    mu, sd = Atr.mean(0), Atr.std(0) + 1e-12
    Xtr = (Atr - mu) / sd
    Xte = (Ate_s - mu) / sd
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(Xtr)), Xtr], ytr, rcond=None)
    pred = np.c_[np.ones(len(Xte)), Xte] @ beta
    ic = _spearman(pred, yte_s)
    hit = float(np.mean(np.sign(pred) == np.sign(yte_s)))
    null = np.array([_spearman(pred, RNG.permutation(yte_s)) for _ in range(100)])
    z = (ic - null.mean()) / (null.std() + 1e-12)

    if verbose:
        print(f"  n_train={tr.sum()}  n_test(thinned)={len(sub)}")
        print("  per-component OOS rank-IC:")
        for i, c in enumerate(per_comp):
            print(f"    c{i+1}: {c:+.4f}")
        print(f"  combined OOS IC {ic:+.4f}  hit {hit*100:.1f}%  "
              f"null {null.mean():+.4f}±{null.std():.4f}  z={z:+.2f}")
    return dict(L=L, H=H, per_comp=per_comp, ic=ic, hit=hit, z=z,
                null_std=null.std())


def main():
    uni = load_universe()
    print(f"universe: {len(uni)} stocks\n")
    print("sweep (combined OOS IC / z):")
    rows = []
    for L, H in GRID:
        r = evaluate(uni, L, H)
        rows.append(r)
        print(f"  L={L:3d} H={H:2d}   combined IC {r['ic']:+.4f}  z={r['z']:+.2f}"
              f"   | c1 {r['per_comp'][0]:+.3f}  c2 {r['per_comp'][1]:+.3f}"
              f"  c3 {r['per_comp'][2]:+.3f}")
    print(f"\ndetailed at L={REP[0]} H={REP[1]}:")
    detail = evaluate(uni, REP[0], REP[1], verbose=True)
    _plot(rows, detail)
    print("\nsaved phase2_direction.png")


def _plot(rows, detail):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    labels = [f"L{r['L']}\nH{r['H']}" for r in rows]
    ics = [r["ic"] for r in rows]
    errs = [r["null_std"] for r in rows]
    a1.bar(labels, ics, yerr=np.array(errs) * 2, color="tab:blue", alpha=0.8, capsize=4)
    a1.axhline(0, color="k", lw=0.6)
    a1.set_ylabel("combined OOS rank-IC"); a1.set_title("Direction IC by scale/horizon\n(err = 2× perm-null std)")

    xs = np.arange(1, K + 1)
    a2.bar(xs, detail["per_comp"], color="tab:orange", alpha=0.8)
    a2.axhline(0, color="k", lw=0.6)
    a2.set_xticks(xs); a2.set_xticklabels([f"c{i}" for i in xs])
    a2.set_ylabel("OOS rank-IC")
    a2.set_title(f"Per-component IC at L={REP[0]} H={REP[1]}\n"
                 f"combined {detail['ic']:+.4f} (z={detail['z']:+.2f})")
    fig.tight_layout(); fig.savefig("phase2_direction.png", dpi=110)


if __name__ == "__main__":
    main()
