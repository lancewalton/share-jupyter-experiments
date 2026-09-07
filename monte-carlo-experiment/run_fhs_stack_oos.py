"""Out-of-sample calibration: FHS with EWMA filter vs FHS with the haf expert stack.

Fit the stack conditional-variance coefficients on the FTSE train half, then run
the walk-forward calibration harness strictly on the OOS second half, comparing
fhs_ewma (single-EWMA filter) to fhs_stack (multi-expert filter). Also a modest
cross-asset check across equities (coefficients fit on the pooled train period).
"""
from __future__ import annotations

import glob
import os

import numpy as np

from mc.data import close_array, load_close
from mc.returns import log_returns
from mc.forecasters import fhs_ewma, fhs_stack, fit_stack_coeffs, _expert_var_features
from mc.walkforward import evaluate, NOMINAL_INTERVALS

T = 10


def _report(name, res):
    print(f"  [{name}]  (n={res.n_origins})")
    for p in NOMINAL_INTERVALS:
        v = res.coverage[p]
        print(f"      {p:>4.0%}: {v[0]:.3f} {v[4]:.3f} {v[9]:.3f}")
    print(f"      CRPS h1/5/10: {res.mean_crps[0]:.5f} {res.mean_crps[4]:.5f} {res.mean_crps[9]:.5f}")


def ftse():
    r = log_returns(close_array())
    split = len(r) // 2
    coeffs = fit_stack_coeffs(r[:split])
    print(f"FTSE: {len(r)} returns, coeffs fit on first {split}, OOS origins from {split}")
    print(f"  stack coeffs [const,ewma_s,ewma_m,ewma_l,long_mean,leverage]:\n   "
          f"{np.array2string(coeffs, precision=3)}\n")
    mh = split
    r_ew = evaluate(r, fhs_ewma(lam=0.94, n_sims=10_000, seed=0), T=T, min_history=mh, stride=5)
    r_st = evaluate(r, fhs_stack(coeffs, n_sims=10_000, seed=0), T=T, min_history=mh, stride=5)
    print("FTSE OOS calibration (coverage h1/5/10; CRPS):")
    _report("fhs_ewma", r_ew)
    _report("fhs_stack", r_st)


def _load_equities(min_rows=1500, glitch=0.6):
    out = []
    for pat in ["/Users/lance/Projects/shares/data/yfinance/*.csv",
                "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]:
        for path in sorted(glob.glob(pat)):
            try:
                r = log_returns(load_close(path).to_numpy())
            except Exception:
                continue
            if len(r) >= min_rows and np.abs(r).max() <= glitch:
                out.append(r)
    return out


def xsec(n_stocks=40, stride=10):
    stocks = _load_equities()[:n_stocks]
    # pooled coeffs from each stock's first-half
    F, Y = [], []
    for r in stocks:
        half = len(r) // 2
        Fi = _expert_var_features(r[:half])
        F.append(Fi[60:]); Y.append((r[:half] ** 2)[60:])
    coeffs, *_ = np.linalg.lstsq(np.vstack(F), np.concatenate(Y), rcond=None)
    print(f"\nCross-asset ({len(stocks)} equities, pooled coeffs), OOS second half:")
    agg = {"fhs_ewma": [], "fhs_stack": []}
    crps = {"fhs_ewma": [], "fhs_stack": []}
    for r in stocks:
        mh = len(r) // 2
        for name, fc in [("fhs_ewma", fhs_ewma(lam=0.94, n_sims=6000, seed=0)),
                         ("fhs_stack", fhs_stack(coeffs, n_sims=6000, seed=0))]:
            try:
                res = evaluate(r, fc, T=T, min_history=mh, stride=stride)
            except Exception:
                continue
            agg[name].append([res.coverage[p][9] for p in NOMINAL_INTERVALS])
            crps[name].append(res.mean_crps[9])
    print(f"  mean OOS coverage at h=10 (nominal {[f'{p:.0%}' for p in NOMINAL_INTERVALS]}):")
    for name in agg:
        m = np.mean(agg[name], axis=0)
        print(f"    {name:<10}: {' '.join(f'{x:.3f}' for x in m)}   CRPS {np.mean(crps[name]):.5f}")


def main():
    ftse()
    xsec()


if __name__ == "__main__":
    main()
