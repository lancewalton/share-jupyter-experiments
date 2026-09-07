"""Phase 5 (stage 1): is there ANY path structure beyond volatility?

Train a classifier to tell real price windows from vol-matched FHS surrogates
(same growth, same volatility envelope, directional order shuffled), pooled
across the universe, walk-forward. Because rare structure dilutes overall
accuracy, we score by TAIL ENRICHMENT -- how far above 50% the real fraction is
among the most-confident windows -- not by average accuracy.

Guards, because a rare real signal is easily swamped:
  * surrogate-vs-surrogate control -- two independent fakes; must score ~chance,
    or our pipeline has a bias and the real result is untrustworthy.
  * |standardised-residual| autocorrelation -- if ~0, the EWMA removed volatility
    clustering, so a positive result is directional structure, not leftover vol.

Then a SPIKE-IN rarity power curve: plant a known pattern into a fraction f of
otherwise-null windows and measure the detector's tail enrichment vs f -- the
frequency floor, so we know the rarest structure this machine could catch.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from matching.represent import vol_scaled_windows
from matching.surrogate import fhs_surrogate, resid_abs_autocorr
from matching.synth import biased_exp_decay
from matching.universe import load_universe

P, STRIDE, LOOKBACK, TRAIN_FRAC = 32, 5, 60, 0.6
RNG = np.random.default_rng(0)


def _collect(series, L, transform=None):
    Ws, ds = [], []
    for _, logP, dates in series:
        p = transform(logP) if transform else logP
        W, ends, _, _ = vol_scaled_windows(p, L=L, P=P, stride=STRIDE, lookback=LOOKBACK)
        ok = np.isfinite(W).all(axis=1)
        if ok.any():
            Ws.append(W[ok]); ds.append(dates[ends[ok]])
    return np.vstack(Ws), np.concatenate(ds)


def _split(dates):
    cut = np.quantile(dates.astype("int64"), TRAIN_FRAC)
    tr = dates.astype("int64") <= cut
    return tr, ~tr


def _enrich(scores, y, frac):
    k = max(1, int(frac * len(scores)))
    top = np.argsort(scores)[::-1][:k]
    return float(y[top].mean() - y.mean())


def _fit_score(X, y, dates):
    tr, te = _split(dates)
    clf = HistGradientBoostingClassifier(max_iter=200, l2_regularization=1.0,
                                         random_state=0).fit(X[tr], y[tr])
    s = clf.predict_proba(X[te])[:, 1]
    return dict(auc=roc_auc_score(y[te], s), e1=_enrich(s, y[te], 0.01),
                e5=_enrich(s, y[te], 0.05), e10=_enrich(s, y[te], 0.10),
                scores=s, te=te)


def _labelled(Wa, da, Wb, db):
    X = np.vstack([Wa, Wb]); y = np.r_[np.ones(len(Wa)), np.zeros(len(Wb))]
    return X, y, np.concatenate([da, db])


def part_a(uni):
    print("PART A -- real vs vol-matched surrogate (tail enrichment):")
    print("   L    AUC    enrich@1%  @5%  @10%   | control(surr-vs-surr) AUC  @5%")
    rows = []
    for L in (40, 80):
        Wr, dr = _collect(uni, L)
        Ws, ds = _collect(uni, L, transform=lambda lp: fhs_surrogate(lp, RNG))
        X, y, d = _labelled(Wr, dr, Ws, ds)
        r = _fit_score(X, y, d)
        Ws2, ds2 = _collect(uni, L, transform=lambda lp: fhs_surrogate(lp, RNG))
        Xc, yc, dc = _labelled(Ws, ds, Ws2, ds2)
        c = _fit_score(Xc, yc, dc)
        rows.append((L, r, c))
        print(f"  {L:3d}  {r['auc']:.3f}   {r['e1']:+.3f}   {r['e5']:+.3f}  {r['e10']:+.3f}"
              f"   |  {c['auc']:.3f}   {c['e5']:+.3f}")
    ac = np.mean([[resid_abs_autocorr(lp)[1], resid_abs_autocorr(lp)[5]]
                  for _, lp, _ in uni], axis=0)
    print(f"\n  null cleanliness: mean |resid| autocorr  lag1 {ac[0]:+.3f}  lag5 {ac[1]:+.3f}"
          "  (~0 => difference is directional, not leftover vol)")
    return rows


def part_b(uni, L=80, A_list=(1.0, 2.0), f_grid=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2)):
    print("\nPART B -- spike-in rarity power curve (L=80):")
    Wa, da = _collect(uni, L, transform=lambda lp: fhs_surrogate(lp, RNG))
    Wb, db = _collect(uni, L, transform=lambda lp: fhs_surrogate(lp, RNG))
    g = biased_exp_decay(P)
    curves = {}
    for A in A_list:
        print(f"  amplitude A={A}:")
        print("     f      enrich@5%   carrier-AUC")
        pts = []
        for f in f_grid:
            Wsp = Wa.copy()
            ncar = int(f * len(Wsp))
            car = np.zeros(len(Wsp), bool)
            if ncar:
                idx = RNG.choice(len(Wsp), ncar, replace=False)
                car[idx] = True
                signs = RNG.choice([-1.0, 1.0], ncar)
                Wsp[idx] += signs[:, None] * A * g[None, :]
            X, y, d = _labelled(Wsp, da, Wb, db)
            tr, te = _split(d)
            clf = HistGradientBoostingClassifier(max_iter=200, l2_regularization=1.0,
                                                 random_state=0).fit(X[tr], y[tr])
            s = clf.predict_proba(X[te])[:, 1]
            e5 = _enrich(s, y[te], 0.05)
            carrier_flag = np.r_[car, np.zeros(len(Wb), bool)][te]
            cauc = roc_auc_score(carrier_flag, s) if carrier_flag.any() else 0.5
            pts.append((f, e5, cauc))
            print(f"   {f:5.3f}    {e5:+.3f}      {cauc:.3f}")
        curves[A] = pts
    return curves


def main():
    uni = load_universe()
    print(f"universe: {len(uni)} stocks\n")
    rows = part_a(uni)
    curves = part_b(uni)

    print("\n=== stage-1 verdict ===")
    best_e5 = max(r["e5"] for _, r, _ in rows)
    ctrl_e5 = max(abs(c["e5"]) for _, _, c in rows)
    print(f"real-vs-surrogate best enrich@5% = {best_e5:+.3f}"
          f"   (surr-vs-surr control {ctrl_e5:.3f})")
    verdict = best_e5 > 0.03 and best_e5 > 3 * ctrl_e5
    print(f"=> {'STRUCTURE DETECTED beyond volatility' if verdict else 'no structure beyond volatility (control-level)'}")
    _plot(rows, curves)
    print("saved phase5_detector.png")


def _plot(rows, curves):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    Ls = [L for L, r, c in rows]
    a1.bar(np.array(Ls) - 4, [r["e5"] for L, r, c in rows], 8, label="real vs surrogate")
    a1.bar(np.array(Ls) + 4, [c["e5"] for L, r, c in rows], 8, label="surr vs surr (control)")
    a1.axhline(0, color="k", lw=0.5)
    a1.set_xlabel("window length L"); a1.set_ylabel("tail enrichment @5%")
    a1.set_title("Part A: is real distinguishable from vol-matched fake?"); a1.legend(fontsize=8)
    for A, pts in curves.items():
        fs = [p[0] for p in pts]
        a2.plot(fs, [p[1] for p in pts], "o-", label=f"enrich@5% (A={A})")
    a2.axhline(0, color="k", lw=0.5)
    a2.set_xlabel("structured fraction f"); a2.set_ylabel("tail enrichment @5%")
    a2.set_title("Part B: rarity power curve (spike-in)"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("phase5_detector.png", dpi=110)


if __name__ == "__main__":
    main()
