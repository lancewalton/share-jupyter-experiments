"""Phase 4: shapelet / multi-timescale sweep.

Do SHORT local shapes (the shapelet regime, ~5-12 days) carry more out-of-sample
predictive signal than the 30-day gestalt (which gave IC ~0.006)? Sweep the
window length L, and at each length run the cross-sectional cluster-then-test of
Phase 3 (edge persistence + permutation null + OOS directional IC). Stocks are
loaded once and cached; only the windowing changes per L.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from mc.data import load_close
from mc.returns import log_returns
from patterns.windows import extract_windows, forward_return

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
LENGTHS = [5, 8, 12, 20, 30, 45, 60]
K, H, STRIDE = 100, 10, 5
MIN_ROWS, GLITCH, MINC = 1500, 0.6, 60
RNG = np.random.default_rng(0)


def _spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def _load_stocks():
    out = []
    for pattern in DIRS:
        for path in sorted(glob.glob(pattern)):
            try:
                s = load_close(path)
            except Exception:
                continue
            r = log_returns(s.to_numpy())
            if len(r) < MIN_ROWS or np.abs(r).max() > GLITCH:
                continue
            out.append((np.log(s.to_numpy()), s.index.to_numpy().astype("int64")))
    return out


def _metrics_for_L(stocks, L):
    Ws, fwds, dts = [], [], []
    for logP, dates in stocks:
        W, ends, _ = extract_windows(logP, L=L, stride=STRIDE, znorm=True)
        f = forward_return(logP, ends, H=H)
        ok = np.isfinite(f) & np.isfinite(W).all(axis=1)
        if ok.any():
            Ws.append(W[ok]); fwds.append(f[ok]); dts.append(dates[ends[ok]])
    W = np.vstack(Ws); fwd = np.concatenate(fwds); dt = np.concatenate(dts)
    o = np.argsort(dt); W, fwd, dt = W[o], fwd[o], dt[o]
    tr = dt <= np.quantile(dt, 0.6)

    km = KMeans(n_clusters=K, n_init=3, random_state=0).fit(W[tr])
    lab_tr, lab_te = km.labels_, km.predict(W[~tr])
    fwd_tr, fwd_te = fwd[tr], fwd[~tr]

    def edges(lab, f):
        return np.array([f[lab == c].mean() if (lab == c).any() else np.nan
                         for c in range(K)])
    et, ee = edges(lab_tr, fwd_tr), edges(lab_te, fwd_te)
    ct = np.array([(lab_tr == c).sum() for c in range(K)])
    ce = np.array([(lab_te == c).sum() for c in range(K)])
    keep = (ct >= MINC) & (ce >= MINC)
    persist = _spearman(et[keep], ee[keep])
    null = np.array([_spearman(et[keep], edges(lab_te, RNG.permutation(fwd_te))[keep])
                     for _ in range(15)])
    z = (persist - null.mean()) / (null.std() + 1e-9)

    pred = et[lab_te]; sub = np.arange(0, len(pred), H)
    ic = _spearman(pred[sub], fwd_te[sub])
    hit = float(np.mean(np.sign(pred[sub]) == np.sign(fwd_te[sub])))
    return len(W), persist, z, ic, hit


def _plot_sweep(triples):
    """triples: list of (L, null-z, OOS IC)."""
    Ls = [t[0] for t in triples]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.plot(Ls, [t[1] for t in triples], "o-")
    a1.axhline(2, color="tab:red", lw=0.8, ls="--", label="z=2 (significant)")
    a1.axhline(0, color="k", lw=0.6, ls=":")
    a1.set_xlabel("window length L (days)"); a1.set_ylabel("edge-persistence z vs null")
    a1.set_title("Is the shape effect real, by scale?"); a1.legend(fontsize=8)
    a2.plot(Ls, [t[2] for t in triples], "o-", color="tab:orange")
    a2.axhline(0, color="k", lw=0.7, ls=":")
    a2.set_xlabel("window length L (days)"); a2.set_ylabel("OOS directional IC")
    a2.set_title("Is it big enough to trade?  (|IC| < 0.02 everywhere)")
    fig.tight_layout(); fig.savefig("phase4_lengthsweep.png", dpi=110)


def main():
    stocks = _load_stocks()
    print(f"stocks: {len(stocks)}   K={K} H={H} stride={STRIDE}\n")
    print(f"{'L':>4}{'windows':>10}{'persist':>9}{'null-z':>8}{'OOS IC':>9}{'hit%':>7}")
    rows = []
    for L in LENGTHS:
        m = _metrics_for_L(stocks, L)
        rows.append((L, *m))
        print(f"{L:>4}{m[0]:>10}{m[1]:>+9.3f}{m[2]:>+8.2f}{m[3]:>+9.4f}{m[4]*100:>6.1f}%",
              flush=True)

    _plot_sweep([(r[0], r[3], r[4]) for r in rows])  # (L, null-z, OOS IC)
    print("\nsaved phase4_lengthsweep.png")


if __name__ == "__main__":
    main()
