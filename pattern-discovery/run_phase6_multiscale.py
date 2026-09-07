"""Phase 6: are predictive price patterns SCALE-INVARIANT (the TA premise)?

Resample windows from several time scales to a common length so shapes are
scale-free, pool them, cluster once, and ask:
  (1) SELF-SIMILARITY: does a shape-cluster's forward edge agree ACROSS scales?
      (a bullish shape at 20 days should be bullish at 80 days if scale-invariant)
  (2) OOS TRANSFER: does the pooled edge predict out-of-sample, at each scale?
  (3) CONFLUENCE: does restricting to scale-consistent clusters sharpen the IC?
Forward horizon scales with the pattern (H = L/2); forward returns are
standardised per scale so edges are comparable.
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
from patterns.windows import extract_windows, forward_return, resample_to

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
SCALES = [10, 20, 40, 80, 160]
P, K, MINC = 32, 80, 20
MIN_ROWS, GLITCH = 1500, 0.6
RNG = np.random.default_rng(0)


def _spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def _collect():
    rows_W, rows_g, rows_d, rows_f = [], [], [], []
    stocks = 0
    for pattern in DIRS:
        for path in sorted(glob.glob(pattern)):
            try:
                s = load_close(path)
            except Exception:
                continue
            r = log_returns(s.to_numpy())
            if len(r) < MIN_ROWS or np.abs(r).max() > GLITCH:
                continue
            logP = np.log(s.to_numpy()); dates = s.index.to_numpy().astype("int64")
            for gi, L in enumerate(SCALES):
                stride = max(2, L // 4)
                W, ends, _ = extract_windows(logP, L=L, stride=stride, znorm=False)
                HL = max(3, round(0.5 * L))
                f = forward_return(logP, ends, H=HL)
                Wn = resample_to(W, P)
                mu = Wn.mean(1, keepdims=True); sd = Wn.std(1, keepdims=True)
                Wn = (Wn - mu) / (sd + 1e-9)
                ok = np.isfinite(f) & np.isfinite(Wn).all(1)
                rows_W.append(Wn[ok]); rows_g.append(np.full(ok.sum(), gi))
                rows_d.append(dates[ends[ok]]); rows_f.append(f[ok])
            stocks += 1
    return (np.vstack(rows_W), np.concatenate(rows_g),
            np.concatenate(rows_d), np.concatenate(rows_f), stocks)


def main():
    W, g, d, f, stocks = _collect()
    # standardise forward per scale (comparable edges), using TRAIN only for stats
    cut = np.quantile(d, 0.6)
    tr = d <= cut
    fz = f.copy()
    for gi in range(len(SCALES)):
        m = g == gi
        sd = f[m & tr].std()
        fz[m] = f[m] / (sd + 1e-12)
    print(f"stocks {stocks}   total windows {len(W)}   "
          f"per scale {[int((g==i).sum()) for i in range(len(SCALES))]}")
    print(f"train {tr.sum()}  test {(~tr).sum()}   P={P} K={K}\n")

    km = KMeans(n_clusters=K, n_init=3, random_state=0).fit(W[tr])
    lab = np.empty(len(W), int); lab[tr] = km.labels_; lab[~tr] = km.predict(W[~tr])

    # per-cluster, per-scale TRAIN edge
    edge = np.full((K, len(SCALES)), np.nan)
    for c in range(K):
        for gi in range(len(SCALES)):
            m = tr & (lab == c) & (g == gi)
            if m.sum() >= MINC:
                edge[c, gi] = fz[m].mean()

    # (1) self-similarity: mean pairwise cross-scale edge correlation
    pair = []
    for i in range(len(SCALES)):
        for j in range(i + 1, len(SCALES)):
            pair.append(_spearman(edge[:, i], edge[:, j]))
    consist = np.nanmean(pair)
    # null: shuffle fwd within scale on train, recompute edges' cross-scale corr
    null = []
    for _ in range(15):
        fzs = fz.copy()
        for gi in range(len(SCALES)):
            m = tr & (g == gi); fzs[m] = RNG.permutation(fzs[m])
        e2 = np.full((K, len(SCALES)), np.nan)
        for c in range(K):
            for gi in range(len(SCALES)):
                m = tr & (lab == c) & (g == gi)
                if m.sum() >= MINC:
                    e2[c, gi] = fzs[m].mean()
        pp = [_spearman(e2[:, a], e2[:, b]) for a in range(len(SCALES))
              for b in range(a + 1, len(SCALES))]
        null.append(np.nanmean(pp))
    null = np.array(null)
    print("(1) SELF-SIMILARITY — cross-scale edge consistency (mean pairwise rho):")
    print(f"    observed {consist:+.3f}   null {null.mean():+.3f} ± {null.std():.3f}"
          f"   z = {(consist-null.mean())/(null.std()+1e-9):+.2f}\n")

    # (2) OOS transfer: pooled train edge -> predict test fwd, overall & per scale
    pooled = np.array([fz[tr & (lab == c)].mean() if (tr & (lab == c)).any() else np.nan
                       for c in range(K)])
    pred = pooled[lab]
    ic_all = _spearman(pred[~tr], fz[~tr])
    print("(2) OOS TRANSFER — IC of pooled-edge vs test forward, per scale:")
    per_scale_ic = []
    for gi, L in enumerate(SCALES):
        m = (~tr) & (g == gi)
        ic = _spearman(pred[m], fz[m]); per_scale_ic.append(ic)
        print(f"    L={L:>3}: IC {ic:+.4f}")
    print(f"    overall: IC {ic_all:+.4f}\n")

    # (3) confluence: restrict to scale-consistent clusters (edge same sign across scales)
    signs = np.sign(edge)
    consistent = np.array([np.all(signs[c][np.isfinite(edge[c])] == signs[c][np.isfinite(edge[c])][0])
                           and np.isfinite(edge[c]).sum() >= 3 for c in range(K)])
    mask_te = (~tr) & consistent[lab]
    ic_conf = _spearman(pred[mask_te], fz[mask_te])
    print(f"(3) CONFLUENCE — {consistent.sum()}/{K} clusters agree in sign across >=3 scales")
    print(f"    OOS IC all clusters {ic_all:+.4f}  vs  scale-consistent only {ic_conf:+.4f}")

    _plot(edge, per_scale_ic, consist, null)
    print("\nsaved phase6_multiscale.png")


def _plot(edge, per_scale_ic, consist, null):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    order = np.argsort(np.nanmean(edge, axis=1))
    im = a1.imshow(edge[order], aspect="auto", cmap="RdBu_r",
                   vmin=-np.nanmax(np.abs(edge)), vmax=np.nanmax(np.abs(edge)))
    a1.set_xticks(range(len(SCALES))); a1.set_xticklabels(SCALES)
    a1.set_xlabel("time scale L (days)"); a1.set_ylabel("cluster (sorted by mean edge)")
    a1.set_title(f"Per-cluster forward edge by scale\ncross-scale consistency "
                 f"{consist:+.2f} (null {null.mean():+.2f})")
    fig.colorbar(im, ax=a1, shrink=0.8, label="std forward return")
    a2.bar([str(s) for s in SCALES], per_scale_ic, color="tab:orange")
    a2.axhline(0, color="k", lw=0.7, ls=":")
    a2.set_xlabel("time scale L (days)"); a2.set_ylabel("OOS transfer IC")
    a2.set_title("Does the pooled pattern predict at each scale?")
    fig.tight_layout(); fig.savefig("phase6_multiscale.png", dpi=110)


if __name__ == "__main__":
    main()
