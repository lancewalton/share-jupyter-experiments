"""Phase 9: does depth help the CALIBRATION side (forecasting envelope width)?

Calibration cares about the forward *volatility* (envelope width), not direction.
Test whether trailing peak-to-trough DEPTH predicts forward realised volatility
INCREMENTALLY over an EWMA vol forecast. If it adds, a depth-aware envelope should
calibrate better; if not, depth is a direction-only signal.

Part 1: pooled log-linear regression of forward realised vol on EWMA vol, with and
        without depth -> test-set R^2 gain and partial rank IC.
Part 2: run the mc calibration harness (coverage/CRPS) on FTSE for a plain
        EWMA-Gaussian envelope vs one whose width is scaled by the fitted depth
        factor -> does coverage/CRPS improve?
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from mc.data import close_array, load_close
from mc.returns import log_returns, realised_cumulative
from mc.walkforward import evaluate, NOMINAL_INTERVALS
from mc.forecasters import gaussian_ewma, _z

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
LD, H, VOLW, LAM = 80, 20, 60, 0.94
MIN_ROWS, GLITCH = 1500, 0.6


def _ewma_vol(r, lam=LAM, burn=60):
    var = np.empty(len(r)); var[:burn] = np.var(r[:burn])
    for i in range(burn, len(r)):
        var[i] = lam * var[i - 1] + (1 - lam) * r[i - 1] ** 2
    return np.sqrt(var)  # sigma[i] forecasts r[i] from data < i (causal)


def _spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def part1():
    EV, DP, RV, DT, CY = [], [], [], [], []
    for pattern in DIRS:
        for path in sorted(glob.glob(pattern)):
            try:
                s = load_close(path)
            except Exception:
                continue
            r = log_returns(s.to_numpy())
            if len(r) < MIN_ROWS or np.abs(r).max() > GLITCH:
                continue
            ev = _ewma_vol(r)
            logP = np.concatenate([[0.0], np.cumsum(r)])
            lp = pd.Series(logP)
            depth = (lp.rolling(LD).max() - lp.rolling(LD).min()).to_numpy()[1:]  # align to r
            # forward realised vol (RMS) and cumulative return over next H
            rv = np.full(len(r), np.nan); cy = np.full(len(r), np.nan)
            for t in range(len(r) - H):
                nxt = r[t + 1:t + 1 + H]
                rv[t] = np.sqrt(np.mean(nxt ** 2)); cy[t] = nxt.sum()
            dts = s.index.to_numpy().astype("int64")[1:]
            ok = np.isfinite(ev) & np.isfinite(depth) & np.isfinite(rv) & np.isfinite(cy) \
                & (ev > 0) & (depth > 0) & (rv > 0)
            EV.append(ev[ok]); DP.append(depth[ok]); RV.append(rv[ok]); DT.append(dts[ok]); CY.append(cy[ok])
    EV, DP, RV, DT, CY = map(np.concatenate, (EV, DP, RV, DT, CY))
    cut = np.quantile(DT, 0.6); tr = DT <= cut
    y, xe, xd = np.log(RV), np.log(EV), np.log(DP)

    def fit(cols):
        X = np.column_stack([np.ones(tr.sum())] + [c[tr] for c in cols])
        beta, *_ = np.linalg.lstsq(X, y[tr], rcond=None)
        Xt = np.column_stack([np.ones((~tr).sum())] + [c[~tr] for c in cols])
        pred = Xt @ beta
        ss_res = ((y[~tr] - pred) ** 2).sum(); ss_tot = ((y[~tr] - y[~tr].mean()) ** 2).sum()
        return beta, 1 - ss_res / ss_tot

    b_e, r2_e = fit([xe])
    b_ed, r2_ed = fit([xe, xd])
    # partial rank IC of depth after removing EWMA-vol (test)
    be = np.polyfit(xe[tr], y[tr], 1); yr = y[~tr] - np.polyval(be, xe[~tr])
    bd = np.polyfit(xe[tr], xd[tr], 1); dr = xd[~tr] - np.polyval(bd, xe[~tr])
    print(f"Part 1 — forecasting forward realised vol (pooled, {len(EV)} obs, test {(~tr).sum()}):")
    print(f"  IC(EWMA vol, fwd RV)      : {_spearman(xe[~tr], y[~tr]):+.3f}")
    print(f"  IC(depth,    fwd RV)      : {_spearman(xd[~tr], y[~tr]):+.3f}")
    print(f"  test R^2  EWMA-only       : {r2_e:.4f}")
    print(f"  test R^2  EWMA + depth    : {r2_ed:.4f}   (dR^2 = {r2_ed - r2_e:+.4f})")
    print(f"  depth coeff (log-log)     : {b_ed[2]:+.3f}")
    print(f"  PARTIAL IC(depth | EWMA)  : {_spearman(dr, yr):+.3f}\n")

    # In-domain pooled calibration: does depth-tilted vol give better H-day coverage?
    from statistics import NormalDist
    Nrm = NormalDist(); beta_d = b_ed[2]; bd = np.polyfit(xe[tr], xd[tr], 1)
    resid = xd - (bd[0] * xe + bd[1])
    sig_e = EV * np.sqrt(H)                       # H-day sigma, EWMA
    sig_d = EV * np.exp(beta_d * resid) * np.sqrt(H)   # + mean-preserving depth tilt
    te = ~tr
    print(f"Part 2a — IN-DOMAIN pooled equity calibration (test {te.sum()} obs):")
    print(f"  {'nominal':>8}{'ewma':>8}{'ewma+depth':>12}")
    for p in NOMINAL_INTERVALS:
        z = Nrm.inv_cdf(0.5 + p / 2)
        ce = np.mean(np.abs(CY[te]) <= z * sig_e[te])
        cd = np.mean(np.abs(CY[te]) <= z * sig_d[te])
        print(f"  {p:>7.0%}{ce:>8.3f}{cd:>12.3f}")
    print()
    return beta_d, bd


def _gaussian_depth(beta_d, bd, lam=LAM, burn=VOLW):
    """EWMA-Gaussian envelope with a MEAN-PRESERVING depth tilt: keep the (well-
    calibrated) EWMA level and multiply by exp(beta_d * depth_residual), where
    depth_residual is the part of log-depth NOT explained by log-EWMA-vol. When
    depth is typical for the current vol the factor is ~1 (level unchanged)."""
    def f(history, T, levels):
        r = np.asarray(history, float)
        ev = _ewma_vol(r, lam, burn)[-1]
        lp = np.concatenate([[0.0], np.cumsum(r)])
        depth = float(np.max(lp[-LD:]) - np.min(lp[-LD:]))
        if ev <= 0 or depth <= 0:
            sig = ev if ev > 0 else np.std(r[-burn:])
        else:
            resid = np.log(depth) - (bd[0] * np.log(ev) + bd[1])
            sig = ev * np.exp(beta_d * resid)
        z = _z(levels); h = np.arange(1, T + 1, dtype=float)
        return sig * np.outer(z, np.sqrt(h))
    return f


def part2(beta_d, bd):
    r = log_returns(close_array())
    base = gaussian_ewma(lam=LAM, min_obs=250)
    dep = _gaussian_depth(beta_d, bd)
    print("Part 2 — FTSE calibration (coverage vs nominal; CRPS) at h=1/5/10:")
    for name, fc in [("ewma-gaussian", base), ("ewma+depth", dep)]:
        res = evaluate(r, fc, T=10, min_history=max(250, LD + 5), stride=5)
        print(f"  [{name}]  (n={res.n_origins})")
        for p in NOMINAL_INTERVALS:
            v = res.coverage[p]
            print(f"      {p:>4.0%}: {v[0]:.3f} {v[4]:.3f} {v[9]:.3f}")
        print(f"      CRPS h1/5/10: {res.mean_crps[0]:.5f} {res.mean_crps[4]:.5f} "
              f"{res.mean_crps[9]:.5f}")


def main():
    beta_d, bd = part1()
    part2(beta_d, bd)


if __name__ == "__main__":
    main()
