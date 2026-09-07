"""Phase 3: cross-sectional cluster-then-test (high power).

Pool z-normed price-shape windows from the whole equity universe, split by
calendar date (strict), cluster the training windows, and test whether each
shape-cluster's TRAIN forward-return edge persists on the TEST period -- with an
order of magnitude more independent windows than the single-series pilot. A
label-permutation null calibrates 'zero'.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from mc.data import load_close
from mc.returns import log_returns
from patterns.windows import extract_windows, forward_return

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
L, K, H, STRIDE = 30, 100, 10, 5
MIN_ROWS, GLITCH, MINC = 1500, 0.6, 60
RNG = np.random.default_rng(0)


def _spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def _collect():
    Ws, fwds, dates = [], [], []
    n = 0
    for pattern in DIRS:
        for path in sorted(glob.glob(pattern)):
            try:
                s = load_close(path)
            except Exception:
                continue
            r = log_returns(s.to_numpy())
            if len(r) < MIN_ROWS or np.abs(r).max() > GLITCH:
                continue
            logP = np.log(s.to_numpy())
            W, ends, _ = extract_windows(logP, L=L, stride=STRIDE, znorm=True)
            f = forward_return(logP, ends, H=H)
            ok = np.isfinite(f) & np.isfinite(W).all(axis=1)
            if ok.sum() == 0:
                continue
            Ws.append(W[ok]); fwds.append(f[ok])
            dates.append(s.index.to_numpy()[ends[ok]])
            n += 1
    return np.vstack(Ws), np.concatenate(fwds), np.concatenate(dates), n


def main():
    W, fwd, dates, n = _collect()
    order = np.argsort(dates)
    W, fwd, dates = W[order], fwd[order], dates[order]
    cut = np.quantile(dates.astype("int64"), 0.6)
    tr = dates.astype("int64") <= cut
    print(f"universe: {n} stocks   windows: {len(W)}  "
          f"(train {tr.sum()}, test {(~tr).sum()})   L={L} K={K} H={H}\n")

    km = KMeans(n_clusters=K, n_init=4, random_state=0).fit(W[tr])
    lab_tr, lab_te = km.labels_, km.predict(W[~tr])
    fwd_tr, fwd_te = fwd[tr], fwd[~tr]

    def edges(lab, f):
        return np.array([f[lab == c].mean() if (lab == c).any() else np.nan
                         for c in range(K)]), np.array([(lab == c).sum() for c in range(K)])
    et, ct = edges(lab_tr, fwd_tr)
    ee, ce = edges(lab_te, fwd_te)
    keep = (ct >= MINC) & (ce >= MINC)
    persist = _spearman(et[keep], ee[keep])

    # label-permutation null: shuffle test labels' link to fwd
    null = []
    for _ in range(20):
        ee_p, _ = edges(lab_te, RNG.permutation(fwd_te))
        null.append(_spearman(et[keep], ee_p[keep]))
    null = np.array(null)

    pred = et[lab_te]
    sub = np.arange(0, len(pred), H)  # thin overlap
    ic = _spearman(pred[sub], fwd_te[sub])
    hit = float(np.mean(np.sign(pred[sub]) == np.sign(fwd_te[sub])))

    print(f"clusters kept {keep.sum()}/{K}")
    print(f"TRAIN-edge -> TEST-edge persistence: {persist:+.3f}")
    print(f"  permutation null: mean {null.mean():+.3f}  std {null.std():.3f}  "
          f"-> z = {(persist-null.mean())/null.std():+.2f}")
    print(f"OOS directional IC: {ic:+.3f}   hit-rate: {hit*100:.1f}%")

    _plot(et, ee, keep, ct, persist, null)
    print("\nsaved phase3_xsec.png")


def _plot(et, ee, keep, ct, persist, null):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    sz = 20 + 200 * (ct[keep] / ct[keep].max())
    a1.scatter(et[keep] * 100, ee[keep] * 100, s=sz, alpha=0.7)
    a1.axhline(0, color="k", lw=0.6, ls=":"); a1.axvline(0, color="k", lw=0.6, ls=":")
    lim = np.nanmax(np.abs(np.r_[et[keep], ee[keep]])) * 100 * 1.1
    a1.plot([-lim, lim], [-lim, lim], "r--", lw=0.8)
    a1.set_xlim(-lim, lim); a1.set_ylim(-lim, lim)
    a1.set_xlabel("cluster TRAIN edge (%)"); a1.set_ylabel("cluster TEST edge (%)")
    a1.set_title(f"Cross-sectional edge persistence: {persist:+.3f}\n"
                 f"(null {null.mean():+.3f} ± {null.std():.3f})")
    a2.hist(null, bins=12, color="0.7", label="permutation null")
    a2.axvline(persist, color="tab:red", lw=2, label=f"FTSE-universe {persist:+.3f}")
    a2.set_xlabel("edge-persistence correlation"); a2.set_title("signal vs null")
    a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("phase3_xsec.png", dpi=110)


if __name__ == "__main__":
    main()
