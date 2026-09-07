"""Phase 7 (stage 3): does the detected structure PREDICT the forward return?

Three questions, all out-of-sample, thinned so forward labels don't overlap:

  A. Conditioning -- is the reversion signal's IC HIGHER inside the detector's
     confidently-structured windows than pooled? (Does structure-selection
     concentrate a tradeable signal?) Also: does the structuredness score itself
     predict forward DIRECTION (it shouldn't -- it's direction-agnostic) or
     forward MAGNITUDE (it might -- structure may be volatility-linked)?

  B. Per-type continuation vs reversal -- for each structure type, the window's
     own direction (drift) and the mean forward move. Same sign => the move
     CONTINUES (an onset completing); opposite => it REVERSES (a completed move
     mean-reverting). This is the pattern-completion question, per type.

  C. Structure-aware predictor -- fit forward return on [reversion, structuredness,
     shape features, type], OOS, and compare its IC to the pooled-reversion
     baseline. If it doesn't beat plain reversion, the structure adds no
     tradeable direction beyond what we already had.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier

from matching.surrogate import sign_flip_surrogate
from matching.universe import load_universe
from run_phase2_direction import _collect as collect_fwd, _spearman, STRIDE
from run_phase5_detector import _collect as collect_sf, _split
from run_phase6_classify import _features

L, H, KTYPES = 80, 20, 6
RNG = np.random.default_rng(0)


def _detector(uni):
    Wr, dr = collect_sf(uni, L)
    Ws, ds = collect_sf(uni, L, transform=lambda lp: sign_flip_surrogate(lp, RNG))
    X = np.vstack([Wr, Ws]); yv = np.r_[np.ones(len(Wr)), np.zeros(len(Ws))]
    dd = np.concatenate([dr, ds]); trd, _ = _split(dd)
    return HistGradientBoostingClassifier(max_iter=200, l2_regularization=1.0,
                                          random_state=0).fit(X[trd], yv[trd])


def main():
    uni = load_universe()
    print(f"universe: {len(uni)} stocks   L={L} H={H}\n")
    clf = _detector(uni)

    W, y, d = collect_fwd(uni, L, H)
    s = clf.predict_proba(W)[:, 1]
    drift = W[:, -1]; rev = -drift; F = _features(W)
    tr, te = _split(d)
    step = max(1, round(H / STRIDE))
    te_idx = np.where(te)[0][::step]

    struct_tr = tr & (s >= np.quantile(s[tr], 0.80))
    km = KMeans(n_clusters=KTYPES, n_init=4, random_state=0).fit(W[struct_tr])
    typ = km.predict(W)

    # --- A. conditioning ---
    print("A. conditioning (reversion signal = -drift):")
    ic_pool = _spearman(rev[te_idx], y[te_idx])
    top = np.where(te & (s >= np.quantile(s[te], 0.80)))[0][::step]
    ic_top = _spearman(rev[top], y[top])
    amp = np.abs(drift)
    top_amp = np.where(te & (amp >= np.quantile(amp[te], 0.80)))[0][::step]
    ic_amp = _spearman(rev[top_amp], y[top_amp])
    print(f"   reversion IC pooled            {ic_pool:+.4f}")
    print(f"   reversion IC within top-|drift| {ic_amp:+.4f}   (n={len(top_amp)})  "
          "<- simple amplitude control")
    print(f"   reversion IC within structured  {ic_top:+.4f}   (n={len(top)})  "
          "<- detector")
    print(f"   structuredness -> direction    {_spearman(s[te_idx], y[te_idx]):+.4f}"
          "   (expect ~0)")
    print(f"   structuredness -> |return|     {_spearman(s[te_idx], np.abs(y[te_idx])):+.4f}"
          "   (volatility link)")

    # --- B. per-type continuation vs reversal ---
    print("\nB. per-type forward move (OOS test windows, thinned):")
    print("   type    n    drift    mean fwd    t     reading")
    types = []
    for c in range(KTYPES):
        idx = np.where(te & (typ == c))[0][::step]
        if len(idx) < 30:
            print(f"    {c+1:2d}   {len(idx):4d}   (too few)")
            continue
        dmean = drift[idx].mean(); fwd = y[idx]
        se = fwd.std() / np.sqrt(len(fwd)) + 1e-12
        t = fwd.mean() / se
        same = np.sign(fwd.mean()) == np.sign(dmean)
        reading = "continuation" if same else "reversal"
        star = "*" if abs(t) >= 2 else " "
        print(f"    {c+1:2d}   {len(idx):4d}  {dmean:+.2f}    {fwd.mean():+.3f}   "
              f"{t:+.1f}{star}  {reading if abs(t) >= 2 else '(ns)'}")
        types.append((c + 1, len(idx), dmean, fwd.mean(), se, abs(t) >= 2))

    # --- C. structure-aware predictor vs pooled reversion ---
    print("\nC. structure-aware predictor vs pooled reversion (OOS IC):")
    onehot = np.eye(KTYPES)[typ]
    Xf = np.column_stack([rev, s, F[:, 1], F[:, 2], F[:, 3], F[:, 4], onehot])
    mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-9
    Xs = (Xf - mu) / sd
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(tr.sum()), Xs[tr]], y[tr], rcond=None)
    pred = np.c_[np.ones(len(te_idx)), Xs[te_idx]] @ beta
    ic_aware = _spearman(pred, y[te_idx])
    # permutation null for the aware predictor
    null = np.array([_spearman(pred, RNG.permutation(y[te_idx])) for _ in range(200)])
    z = (ic_aware - null.mean()) / (null.std() + 1e-12)
    print(f"   pooled reversion IC   {ic_pool:+.4f}")
    print(f"   structure-aware IC    {ic_aware:+.4f}   (z={z:+.2f}, null {null.std():.4f})")
    print(f"   => structure adds {'a real increment' if ic_aware - abs(ic_pool) > 0.01 else 'little/nothing'} over plain reversion")

    _plot(ic_pool, ic_amp, ic_top, len(top), types)
    print("\nsaved phase7_predict.png")


def _plot(ic_pool, ic_amp, ic_top, n_top, types):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.bar(["pooled\n(all)", "top |drift|\n(amplitude)", "detector\n(structured)"],
           [ic_pool, ic_amp, ic_top], color=["0.6", "tab:orange", "tab:blue"])
    a1.axhline(0, color="k", lw=0.5)
    a1.set_ylabel("reversion-signal OOS IC")
    a1.set_title(f"Conditioning doubles reversion IC -- but the one-line\n"
                 f"amplitude filter matches the detector (n={n_top})")
    for c, n, dmean, fwd, se, sig in types:
        col = "tab:red" if fwd < 0 else "tab:green"
        a2.bar(c, fwd, yerr=2 * se, color=col, alpha=0.9 if sig else 0.4, capsize=4)
    a2.axhline(0, color="k", lw=0.5)
    a2.set_xlabel("structure type"); a2.set_ylabel("mean forward return (vol units)")
    a2.set_title("Per-type forward move (±2 SE)\ndown-types that bounce = reversal; solid = significant")
    fig.tight_layout(); fig.savefig("phase7_predict.png", dpi=110)


if __name__ == "__main__":
    main()
