"""Phase 6 (stage 2): what IS the detected directional structure?

Stage 1 showed real windows are distinguishable from a vol-identical sign-flipped
fake -- directional structure exists. Before asking whether it predicts (stage 3),
we characterise it: is it just KNOWN directional effects, or something novel?

Method. Build a few interpretable per-window features that capture the usual
suspects a sign-flip would destroy:
  * drift        -- net displacement (momentum / reversion axis)
  * skew         -- asymmetry of the within-window moves (crashes sharper than
                    rallies: the leverage / negative-skew effect)
  * ac1          -- lag-1 autocorrelation of moves (momentum vs reversion)
  * early_late   -- first-half minus second-half displacement (a biased-decay /
                    front-loaded-move signature)
  * range        -- peak-to-trough span

Then compare two classifiers on real-vs-sign-flip: the FULL 32-point shape vs
ONLY these known features. If known features reproduce the full discrimination,
the "structure" is known effects; if the full shape wins by a lot, there is novel
structure beyond them. We also report which known feature separates real from
fake most, and cluster the confidently-structured windows to see the types.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from matching.surrogate import sign_flip_surrogate
from matching.universe import load_universe
from run_phase5_detector import _collect, _enrich, _split, P

L = 80
RNG = np.random.default_rng(0)
FEATS = ["drift", "skew", "ac1", "early_late", "range"]


def _features(W):
    inc = np.diff(W, axis=1)
    mu = inc.mean(1, keepdims=True); sd = inc.std(1, keepdims=True) + 1e-9
    zs = (inc - mu) / sd
    skew = (zs ** 3).mean(1)
    a = inc[:, :-1] - mu; b = inc[:, 1:] - mu
    ac1 = (a * b).mean(1) / (inc.var(1) + 1e-12)
    early_late = W[:, P // 2] - (W[:, -1] - W[:, P // 2])
    rng_ = W.max(1) - W.min(1)
    return np.column_stack([W[:, -1], skew, ac1, early_late, rng_])


def main():
    uni = load_universe()
    print(f"universe: {len(uni)} stocks   L={L}\n")
    Wr, dr = _collect(uni, L)
    Ws, ds = _collect(uni, L, transform=lambda lp: sign_flip_surrogate(lp, RNG))

    X = np.vstack([Wr, Ws]); y = np.r_[np.ones(len(Wr)), np.zeros(len(Ws))]
    d = np.concatenate([dr, ds]); tr, te = _split(d)
    Fr, Fs = _features(Wr), _features(Ws)
    F = np.vstack([Fr, Fs])

    def run(mat, name):
        clf = HistGradientBoostingClassifier(max_iter=200, l2_regularization=1.0,
                                             random_state=0).fit(mat[tr], y[tr])
        s = clf.predict_proba(mat[te])[:, 1]
        auc, e5 = roc_auc_score(y[te], s), _enrich(s, y[te], 0.05)
        print(f"  {name:22s}  AUC {auc:.3f}   enrich@5% {e5:+.3f}")
        return auc, e5, clf

    print("real vs sign-flip -- how much do KNOWN features explain?")
    auc_full, e5_full, clf_full = run(X, "full 32-point shape")
    auc_known, e5_known, _ = run(F, "known features only")
    for i, f in enumerate(FEATS):
        auc_i, _, _ = run(F[:, i:i + 1], f)
    print(f"\n  novelty gap (full - known): AUC {auc_full-auc_known:+.3f}  "
          f"enrich {e5_full-e5_known:+.3f}")

    # which known feature separates real from fake most (standardised mean diff)
    print("\n  standardised mean difference (real - sign-flip), per feature:")
    for i, f in enumerate(FEATS):
        pooled = np.std(F[:, i]) + 1e-12
        smd = (Fr[:, i].mean() - Fs[:, i].mean()) / pooled
        print(f"    {f:12s} {smd:+.3f}")

    # cluster the confidently-structured real windows to see the types
    s_real = clf_full.predict_proba(Wr[_split(dr)[1]])[:, 1] if False else None
    clf_all = HistGradientBoostingClassifier(max_iter=200, l2_regularization=1.0,
                                             random_state=0).fit(X[tr], y[tr])
    score_r = clf_all.predict_proba(Wr)[:, 1]
    top = Wr[score_r >= np.quantile(score_r, 0.80)]
    km = KMeans(n_clusters=6, n_init=4, random_state=0).fit(top)
    _plot(km, top, auc_full, auc_known)
    print(f"\n  structured windows clustered (n={len(top)}); saved phase6_classify.png")


def _plot(km, top, auc_full, auc_known):
    xg = np.linspace(0, 1, P)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for c, ax in enumerate(axes.ravel()):
        members = top[km.labels_ == c]
        cen = members.mean(0)
        f = _features(cen[None, :])[0]
        ax.plot(xg, cen, lw=2.2, color="tab:blue")
        ax.fill_between(xg, np.percentile(members, 25, 0), np.percentile(members, 75, 0),
                        alpha=0.15, color="tab:blue")
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_title(f"type {c+1}  (n={len(members)})\n"
                     f"drift {f[0]:+.2f}  skew {f[1]:+.2f}  ac1 {f[2]:+.2f}", fontsize=9)
    fig.suptitle(f"Structure types among confidently-structured windows   "
                 f"(novelty: full AUC {auc_full:.3f} vs known {auc_known:.3f})")
    fig.tight_layout(); fig.savefig("phase6_classify.png", dpi=110)


if __name__ == "__main__":
    main()
