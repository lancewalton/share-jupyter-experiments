"""Phase 3: is the structure predictive, or only descriptive?

Strand A (second moment — the thesis says this should pay): does the cross-asset
absorption ratio forecast forward equity volatility and forward drawdown, and
does it ADD anything over VIX (which already forecasts vol)?

Strand B (first moment — the long shot): does a segment's recent activation
predict its own forward return (factor momentum/reversal), and is there tradeable
lead-lag between segments?

All causal: the absorption ratio at day t uses only the window ending at t; every
target is strictly forward, thinned so windows don't overlap; permutation nulls.
"""
from __future__ import annotations

import numpy as np

from pca.data import _fred_level, build_returns
from pca.factors import pca, standardize

WIN, H = 126, 21


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return np.nan
    ra, rb = np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def _partial(x, y, z):
    """Spearman(x, y) controlling linearly for z."""
    X = np.c_[np.ones(len(z)), z]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return _spearman(rx, ry)


def daily_absorption(Z, win=WIN):
    n = len(Z)
    ar = np.full(n, np.nan)
    for t in range(win, n):
        w = np.linalg.eigvalsh(np.corrcoef(Z[t - win:t], rowvar=False))
        ar[t] = w[-1] / w.sum()
    return ar


def main():
    R = build_returns()
    Z, _, _ = standardize(R.to_numpy())
    var, loadings, scores = pca(Z)
    eq = R["US equity"].to_numpy()
    vix = _fred_level("VIXCLS").reindex(R.index).to_numpy()   # level, causal at t
    ar = daily_absorption(Z)
    n = len(eq)

    fwd_vol = np.full(n, np.nan); fwd_ret = np.full(n, np.nan)
    for t in range(n - H - 1):
        seg = eq[t + 1:t + 1 + H]
        fwd_vol[t] = np.std(seg) * np.sqrt(252) * 100
        fwd_ret[t] = seg.sum()

    step = H
    idx = np.arange(WIN, n - H - 1, step)
    a, v, fv, fr = ar[idx], vix[idx], fwd_vol[idx], fwd_ret[idx]
    ok = np.isfinite(a) & np.isfinite(v) & np.isfinite(fv)
    a, v, fv, fr = a[ok], v[ok], fv[ok], fr[ok]
    dar = np.r_[np.nan, np.diff(a)]

    print(f"panel {R.shape[1]} instruments, {len(a)} non-overlapping months, H={H}d\n")
    print("STRAND A — does cross-asset structure forecast forward EQUITY risk?")
    print(f"{'signal':22s} {'-> fwd vol':>10s} {'-> fwd ret':>10s}")
    print(f"{'absorption ratio (AR)':22s} {_spearman(a,fv):+10.3f} {_spearman(a,fr):+10.3f}")
    print(f"{'ΔAR (rising integ.)':22s} {_spearman(dar,fv):+10.3f} {_spearman(dar,fr):+10.3f}")
    print(f"{'VIX level (baseline)':22s} {_spearman(v,fv):+10.3f} {_spearman(v,fr):+10.3f}")
    print(f"{'AR | VIX (partial)':22s} {_partial(a,fv,v):+10.3f} {_partial(a,fr,v):+10.3f}")
    # permutation null for AR->fwd_ret (the drawdown claim)
    rng = np.random.default_rng(0)
    ic = _spearman(a, fr)
    null = np.array([_spearman(a, rng.permutation(fr)) for _ in range(300)])
    print(f"  AR->fwd_ret {ic:+.3f}  vs null {null.mean():+.3f}±{null.std():.3f} "
          f"(z={(ic-null.mean())/null.std():+.2f})")

    print("\nSTRAND B — first-moment long shot")
    names = ["Rates", "Dollar", "Risk-off", "Energy", "Oil–gas"]
    csum = lambda s: np.array([s[max(0,t-H):t].sum() for t in range(n)])
    print("  factor momentum (trailing vs forward segment return, OOS-style IC):")
    for i, nm in enumerate(names):
        s = scores[:, i]
        trail = csum(s)[idx]; fwd = np.array([s[t+1:t+1+H].sum() for t in idx])
        m = np.isfinite(trail) & np.isfinite(fwd)
        print(f"    {nm:9s} {_spearman(trail[m], fwd[m]):+.3f}")
    print("  lead-lag: max |corr(segment_i(t), segment_j(t+1))| across pairs:")
    best = (0, "", "")
    for i in range(5):
        for j in range(5):
            c = np.corrcoef(scores[:-1, i], scores[1:, j])[0, 1]
            if abs(c) > abs(best[0]):
                best = (c, names[i], names[j])
    print(f"    strongest next-day lead-lag: {best[1]} -> {best[2]}  corr {best[0]:+.3f}")


if __name__ == "__main__":
    main()
