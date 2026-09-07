"""Phase 7: does keeping AMPLITUDE (not just shape) recover predictive signal?

z-normalising each window discards how *big* the pattern is. Here we keep
amplitude, measured against an EXTERNAL reference -- trailing volatility x sqrt(L)
-- so a pattern's size is expressed relative to what a random walk would produce
over its own horizon (scale is accounted for, not divided out). At each scale we
compare three representations by OOS cluster-then-test IC:
  (1) SHAPE-ONLY   : self z-normed shape (the Phase-2/3 baseline)
  (2) AMP-AWARE    : vol/sqrt(L)-normalised shape + amplitude features
                     (range, peak-to-trough depth, net displacement, extrema pos)
  (3) AMP-ONLY     : the amplitude features alone, no shape
If (2) beats (1) and (3): amplitude x shape interaction carries real signal.
If (2) ~ (3): amplitude alone carries it (reversion/vol repackaged).
Forward horizon scales with the pattern: H = L/2.
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
SCALES = [20, 40, 80]
P, K, VOLW, MINC = 24, 60, 60, 40
MIN_ROWS, GLITCH = 1500, 0.6


def _spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return np.nan
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def _windows_for_scale(L):
    """Return per-window: self-shape, amp-shape, amp-features, forward, date."""
    SH, AS, AF, FW, DT = [], [], [], [], []
    HL = max(3, round(0.5 * L))
    stride = max(3, L // 4)
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
            dates = s.index.to_numpy().astype("int64")
            tvol = pd.Series(r).rolling(VOLW).std().to_numpy()  # tvol[i] uses r[i-VOLW+1..i]
            W, ends, starts = extract_windows(logP, L=L, stride=stride, znorm=False)
            f = forward_return(logP, ends, H=HL)
            for seg, st, en, fwd in zip(W, starts, ends, f):
                if not np.isfinite(fwd) or st < 1:
                    continue
                sigma = tvol[st - 1] if st - 1 < len(tvol) else np.nan  # causal: before window
                if not np.isfinite(sigma) or sigma <= 0:
                    continue
                amp = sigma * np.sqrt(L)
                # (1) self z-norm shape
                sh = resample_to(seg[None, :], P)[0]
                sh = (sh - sh.mean()) / (sh.std() + 1e-9)
                # (2) amp-preserving shape (external scale)
                ash = resample_to((seg - seg[0])[None, :], P)[0] / amp
                # amplitude features
                rng = (seg.max() - seg.min()) / amp
                net = (seg[-1] - seg[0]) / amp
                depth = 1.0 - np.exp(seg.min() - seg.max())   # peak-to-trough, price terms
                amin, amax = seg.argmin() / L, seg.argmax() / L
                SH.append(sh); AS.append(ash)
                AF.append([rng, net, depth, amin, amax])
                FW.append(fwd); DT.append(dates[en])
    return (np.array(SH), np.array(AS), np.array(AF, float),
            np.array(FW), np.array(DT))


def _oos_ic(rep, fwd, tr):
    """Standardise columns on train, cluster train, cluster-mean edge, OOS IC."""
    mu = rep[tr].mean(0); sd = rep[tr].std(0) + 1e-9
    X = (rep - mu) / sd
    km = KMeans(n_clusters=K, n_init=3, random_state=0).fit(X[tr])
    lab = np.empty(len(X), int); lab[tr] = km.labels_; lab[~tr] = km.predict(X[~tr])
    edge = np.array([fwd[tr & (lab == c)].mean() if (tr & (lab == c)).sum() >= MINC else np.nan
                     for c in range(K)])
    pred = edge[lab]
    return _spearman(pred[~tr], fwd[~tr])


def main():
    print(f"scales {SCALES}  P={P} K={K}  H=L/2\n")
    print(f"{'L':>4}{'windows':>9}{'shape-only':>12}{'amp-aware':>11}{'amp-only':>10}")
    res = {}
    for L in SCALES:
        SH, AS, AF, FW, DT = _windows_for_scale(L)
        cut = np.quantile(DT, 0.6); tr = DT <= cut
        fz = FW / (FW[tr].std() + 1e-12)
        ic1 = _oos_ic(SH, fz, tr)
        ic2 = _oos_ic(np.hstack([AS, AF]), fz, tr)
        ic3 = _oos_ic(AF, fz, tr)
        # direct OOS rank-IC of key amplitude features (col 0=range,1=net,2=depth)
        te = ~tr
        d_net = _spearman(AF[te, 1], fz[te])
        d_rng = _spearman(AF[te, 0], fz[te])
        d_dep = _spearman(AF[te, 2], fz[te])
        res[L] = (len(SH), ic1, ic2, ic3, d_net, d_rng, d_dep)
        print(f"{L:>4}{len(SH):>9}{ic1:>+12.4f}{ic2:>+11.4f}{ic3:>+10.4f}"
              f"   | direct: net {d_net:+.3f} range {d_rng:+.3f} depth {d_dep:+.3f}",
              flush=True)

    Ls = list(res)
    x = np.arange(len(Ls)); w = 0.25
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.bar(x - w, [res[L][1] for L in Ls], w, label="shape-only")
    a1.bar(x, [res[L][2] for L in Ls], w, label="amp-aware")
    a1.bar(x + w, [res[L][3] for L in Ls], w, label="amp-only")
    a1.axhline(0, color="k", lw=0.7, ls=":")
    a1.set_xticks(x); a1.set_xticklabels([f"L={L}" for L in Ls])
    a1.set_ylabel("OOS IC"); a1.set_title("Cluster-then-test (coarse extractor)\n~0 everywhere")
    a1.legend(fontsize=8)
    a2.plot(Ls, [res[L][4] for L in Ls], "o-", label="net displacement (reversion)")
    a2.plot(Ls, [res[L][5] for L in Ls], "o-", label="range (swing size)")
    a2.plot(Ls, [res[L][6] for L in Ls], "o-", label="depth (peak-to-trough)")
    a2.axhline(0, color="k", lw=0.7, ls=":")
    a2.set_xlabel("time scale L (days)"); a2.set_ylabel("direct OOS rank IC")
    a2.set_title("Amplitude features DO predict — depth grows with scale")
    a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("phase7_amplitude.png", dpi=110)
    print("\nsaved phase7_amplitude.png")


if __name__ == "__main__":
    main()
