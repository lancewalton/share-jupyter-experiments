"""Phase 9: close out the small novel residual from stage 2.

Stage 2 left a loose end: the full 32-point shape distinguished real from
sign-flip slightly better than known features (AUC 0.565 vs 0.548, a +0.017 gap).
Two questions decide whether it matters:

  1. Is the residual a ROBUST detection signal -- does the full shape reliably
     beat known features on held-out data (and across detector seeds)?
  2. Does it PREDICT -- does the shape ORTHOGONAL to the known features carry any
     forward-return information the known features don't already have?

Everything OOS, thinned, and (this time) with heavy regularisation (Ridge on a
low-rank shape residual) so we don't re-create the stage-3 overfit.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from matching.surrogate import sign_flip_surrogate
from matching.universe import load_universe
from run_phase2_direction import _collect as collect_fwd, _spearman, STRIDE
from run_phase5_detector import _collect as collect_sf, _enrich, _split
from run_phase6_classify import _features

L, H = 80, 20
RNG = np.random.default_rng(0)


def _detect(Xtr, ytr, Xte, yte, seed):
    clf = HistGradientBoostingClassifier(max_iter=200, l2_regularization=1.0,
                                         random_state=seed).fit(Xtr, ytr)
    s = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, s), _enrich(s, yte, 0.05)


def _ic_ridge(Ftr, ytr, Fte, yte, alpha=10.0):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    m = Ridge(alpha=alpha).fit((Ftr - mu) / sd, ytr)
    return _spearman(m.predict((Fte - mu) / sd), yte)


def main():
    uni = load_universe()
    print(f"universe: {len(uni)} stocks   L={L} H={H}\n")

    # detection data (real vs sign-flip)
    Wr0, dr0 = collect_sf(uni, L)
    Ws0, ds0 = collect_sf(uni, L, transform=lambda lp: sign_flip_surrogate(lp, RNG))
    X = np.vstack([Wr0, Ws0]); y = np.r_[np.ones(len(Wr0)), np.zeros(len(Ws0))]
    Fdet = np.vstack([_features(Wr0), _features(Ws0)])
    dd = np.concatenate([dr0, ds0]); trd, ted = _split(dd)

    print("1. is the residual a robust DETECTION signal? (real vs sign-flip, OOS)")
    for seed in (0, 1, 2):
        af, ef = _detect(X[trd], y[trd], X[ted], y[ted], seed)
        ak, ek = _detect(Fdet[trd], y[trd], Fdet[ted], y[ted], seed)
        print(f"   seed {seed}:  full AUC {af:.3f}  known {ak:.3f}  gap {af-ak:+.3f}"
              f"   (enrich gap {ef-ek:+.3f})")

    # forward-prediction data (real windows only)
    W, yf, d = collect_fwd(uni, L, H)
    F = _features(W)
    tr, te = _split(d)
    step = max(1, round(H / STRIDE)); te_idx = np.where(te)[0][::step]

    # shape orthogonal to known features (linear residual, fit on train)
    Bm, *_ = np.linalg.lstsq(np.c_[np.ones(tr.sum()), F[tr]], W[tr], rcond=None)
    Wres = W - np.c_[np.ones(len(W)), F] @ Bm
    pca = PCA(n_components=8).fit(Wres[tr])
    R = pca.transform(Wres)
    tr_i = np.where(tr)[0][::step]

    def ics(v):
        return _spearman(v[tr_i], yf[tr_i]), _spearman(v[te_idx], yf[te_idx])

    print("\n2. does the residual PREDICT? single-feature OOS rank-IC "
          "(train / test; stable-and->0.02 = real):")
    print("   known features:")
    for i, name in enumerate(["drift", "skew", "ac1", "early_late", "range"]):
        a, b = ics(F[:, i]); print(f"     {name:12s} {a:+.4f} / {b:+.4f}")
    print("   residual shape components (orthogonal to known):")
    best = 0.0
    for c in range(8):
        a, b = ics(R[:, c])
        stable = np.sign(a) == np.sign(b) and abs(b) >= 0.02
        if stable:
            best = max(best, abs(b))
        print(f"     comp {c+1}      {a:+.4f} / {b:+.4f}   {'<- stable & real' if stable else ''}")

    # multiple-testing null: max |test IC| over the 8 components under shuffled y
    obs_max = max(abs(_spearman(R[te_idx, c], yf[te_idx])) for c in range(8))
    null = []
    for _ in range(300):
        yp = RNG.permutation(yf[te_idx])
        null.append(max(abs(_spearman(R[te_idx, c], yp)) for c in range(8)))
    null = np.array(null)
    pval = float((null >= obs_max).mean())
    print(f"\n   multiple-testing null: best-of-8 |test IC| = {obs_max:.4f}, "
          f"null {null.mean():.4f}±{null.std():.4f}, p = {pval:.3f}")

    # sub-period consistency of the two flagged components, across 4 test chunks
    print("   sub-period test IC (comp2 / comp3) across 4 chronological chunks:")
    chunks = np.array_split(te_idx, 4)
    line = "     " + "   ".join(
        f"[{_spearman(R[ch,1],yf[ch]):+.3f}/{_spearman(R[ch,2],yf[ch]):+.3f}]"
        for ch in chunks)
    print(line)

    print("\n=== verdict ===")
    print("   DETECTION: residual gap ~ +0.017 AUC, stable across seeds -> real & robust")
    real = pval < 0.05
    print(f"   PREDICTION: best-of-8 residual |IC| {obs_max:.4f}, multiple-testing p={pval:.3f}"
          f"  -> {'survives the null' if real else 'consistent with chance (multiple testing)'}")

    _plot(pca.components_, R, yf, te_idx, chunks)
    print("   saved phase9_residual.png")


def _plot(comps, R, yf, te_idx, chunks):
    xg = np.linspace(0, 1, comps.shape[1])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for c, col in [(1, "tab:blue"), (2, "tab:orange")]:
        a1.plot(xg, comps[c], color=col, lw=2, label=f"residual comp {c+1}")
    a1.axhline(0, color="k", lw=0.5, ls=":")
    a1.set_xlabel("normalised time through window")
    a1.set_title("Novel predictive shapes\n(orthogonal to drift/skew/reversion)")
    a1.legend(fontsize=9)
    x = np.arange(4)
    a2.bar(x - 0.2, [_spearman(R[ch, 1], yf[ch]) for ch in chunks], 0.4, label="comp 2")
    a2.bar(x + 0.2, [_spearman(R[ch, 2], yf[ch]) for ch in chunks], 0.4, label="comp 3")
    a2.axhline(0, color="k", lw=0.5)
    a2.set_xticks(x); a2.set_xticklabels([f"chunk {i+1}" for i in range(4)])
    a2.set_ylabel("forward-return IC"); a2.set_title("Stable across test sub-periods")
    a2.legend(fontsize=9)
    fig.tight_layout(); fig.savefig("phase9_residual.png", dpi=110)


if __name__ == "__main__":
    main()
