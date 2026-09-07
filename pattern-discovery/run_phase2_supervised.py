"""Phase 2: supervised cluster-then-test. Do shape-cluster edges persist OOS?

Cluster z-normed price-shape windows on the TRAIN period; each cluster gets a
train forward-return 'edge'. Assign TEST windows to the nearest train centroid.
Decisive question: does a cluster's TRAIN edge predict its TEST edge (across
clusters)? Plus OOS directional IC / hit-rate / long-short. A matched random walk
provides the null (its train edges should NOT predict its test edges).
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from mc.data import close_array
from mc.returns import log_returns
from patterns.windows import log_price, extract_windows, forward_return, temporal_split

L, K, H, STRIDE = 30, 64, 10, 1
MINC = 30          # min members per cluster (each of train & test) for the scatter
RNG = np.random.default_rng(0)


def _spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def _pipeline(logP, label):
    W, ends, _ = extract_windows(logP, L=L, stride=STRIDE, znorm=True)
    fwd = forward_return(logP, ends, H=H)
    ok = np.isfinite(fwd)
    W, ends, fwd = W[ok], ends[ok], fwd[ok]
    tr, te = temporal_split(ends, frac=0.6)

    km = KMeans(n_clusters=K, n_init=4, random_state=0).fit(W[tr])
    lab_tr, lab_te = km.labels_, km.predict(W[te])
    fwd_tr, fwd_te = fwd[tr], fwd[te]

    edge_tr = np.array([fwd_tr[lab_tr == c].mean() if (lab_tr == c).any() else np.nan
                        for c in range(K)])
    edge_te = np.array([fwd_te[lab_te == c].mean() if (lab_te == c).any() else np.nan
                        for c in range(K)])
    cnt_tr = np.array([(lab_tr == c).sum() for c in range(K)])
    cnt_te = np.array([(lab_te == c).sum() for c in range(K)])

    keep = (cnt_tr >= MINC) & (cnt_te >= MINC)
    persist = _spearman(edge_tr[keep], edge_te[keep])

    # OOS directional stats on NON-overlapping test windows (stride H)
    pred = edge_tr[lab_te]                         # predicted edge for each test window
    sub = np.arange(0, len(pred), H)              # thin to reduce overlap
    ic = _spearman(pred[sub], fwd_te[sub])
    hit = float(np.mean(np.sign(pred[sub]) == np.sign(fwd_te[sub])))
    # long-short by sign of predicted edge
    ps, ys = pred[sub], fwd_te[sub]
    ls = ys[ps > 0].mean() - ys[ps < 0].mean() if (ps > 0).any() and (ps < 0).any() else np.nan

    print(f"=== {label} ===  clusters kept {keep.sum()}/{K}  test-windows(thinned) {len(sub)}")
    print(f"  TRAIN-edge -> TEST-edge persistence (Spearman): {persist:+.3f}")
    print(f"  OOS directional IC: {ic:+.3f}   hit-rate: {hit*100:.1f}%   "
          f"L/S mean fwd: {ls:+.5f}\n")
    return edge_tr, edge_te, keep, cnt_tr, persist, km


def main():
    r = log_returns(close_array())
    logP = np.concatenate([[0.0], np.cumsum(r)])
    rw = np.concatenate([[0.0], np.cumsum(RNG.normal(0, r.std(), size=len(r)))])

    et, ee, keep, cnt, persist, km = _pipeline(logP, "FTSE")
    _, _, _, _, persist_rw, _ = _pipeline(rw, "random walk (null)")

    print(f"HEADLINE: FTSE edge-persistence {persist:+.3f}  vs  null {persist_rw:+.3f}")
    _plot(et, ee, keep, cnt, persist, persist_rw, km)
    print("\nsaved phase2_supervised.png")


def _plot(et, ee, keep, cnt, persist, persist_rw, km):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    sz = 20 + 180 * (cnt[keep] / cnt[keep].max())
    a1.scatter(et[keep] * 100, ee[keep] * 100, s=sz, alpha=0.7)
    a1.axhline(0, color="k", lw=0.6, ls=":"); a1.axvline(0, color="k", lw=0.6, ls=":")
    lim = np.nanmax(np.abs(np.r_[et[keep], ee[keep]])) * 100 * 1.1
    a1.plot([-lim, lim], [-lim, lim], "r--", lw=0.8)
    a1.set_xlim(-lim, lim); a1.set_ylim(-lim, lim)
    a1.set_xlabel("cluster TRAIN edge (%, fwd return)")
    a1.set_ylabel("cluster TEST edge (%)")
    a1.set_title(f"Does a shape's edge persist OOS?\nFTSE persistence {persist:+.3f}"
                 f"  (null {persist_rw:+.3f})")

    # show the 3 most bullish / bearish TRAIN clusters' shapes
    order = np.argsort(np.where(keep, et, np.nan))
    bear, bull = order[:3], order[-3:]
    for c in bull:
        a2.plot(km.cluster_centers_[c], color="tab:green", alpha=0.8)
    for c in bear:
        a2.plot(km.cluster_centers_[c], color="tab:red", alpha=0.8)
    a2.set_title("most bullish (green) / bearish (red)\nTRAIN-edge shape centroids")
    a2.set_xlabel("window position"); a2.set_yticks([])
    fig.tight_layout(); fig.savefig("phase2_supervised.png", dpi=110)


if __name__ == "__main__":
    main()
