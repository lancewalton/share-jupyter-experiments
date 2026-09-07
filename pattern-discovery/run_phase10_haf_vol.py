"""Phase 10 (fusing haf): does the haf vol-expert stack improve FHS calibration?

FHS used a single EWMA vol filter. haf forecasts volatility better with a fixed
SIGNED stack of diverse experts (short/mid/long EWMA, long-run mean, momentum,
leverage). Here we build that stack to forecast forward realised vol, and test
whether using it as the envelope width improves in-domain equity calibration over
the plain EWMA(0.94) width. (Phase 9 warned a better vol forecast may not move
coverage if the increment is small; haf's stack is a bigger improvement.)
"""
from __future__ import annotations

import glob
import os
from statistics import NormalDist

import numpy as np
import pandas as pd

from mc.data import load_close
from mc.returns import log_returns
from mc.walkforward import NOMINAL_INTERVALS

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
H, MIN_ROWS, GLITCH = 20, 1500, 0.6
_N = NormalDist()


def _ewma_vol(r, lam=0.94, burn=60):
    var = np.empty(len(r)); var[:burn] = np.var(r[:burn])
    for i in range(burn, len(r)):
        var[i] = lam * var[i - 1] + (1 - lam) * r[i - 1] ** 2
    return np.sqrt(var)


def _experts(r):
    """Causal vol experts (predict forward vol from data up to each day)."""
    s = pd.Series(r); r2 = s ** 2
    ev = lambda hl: np.sqrt(r2.ewm(halflife=hl).mean().to_numpy())
    exp = {
        "ewma_s": ev(3), "ewma_m": ev(10), "ewma_l": ev(40),
        "long_mean": np.sqrt(r2.expanding().mean().to_numpy()),
        "leverage": np.sqrt((r2 * 2 * (s < 0)).ewm(halflife=10).mean().to_numpy()),
    }
    exp["momentum"] = exp["ewma_s"] - exp["ewma_m"]   # vol trend
    return exp


def _spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    ra, rb = np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    EX, EV, RV, DT = {k: [] for k in ["ewma_s", "ewma_m", "ewma_l", "long_mean",
                                      "leverage", "momentum"]}, [], [], []
    for pat in DIRS:
        for path in sorted(glob.glob(pat)):
            try:
                s = load_close(path)
            except Exception:
                continue
            r = log_returns(s.to_numpy())
            if len(r) < MIN_ROWS or np.abs(r).max() > GLITCH:
                continue
            ev = _ewma_vol(r); ex = _experts(r)
            rv = np.full(len(r), np.nan)
            for t in range(len(r) - H):
                rv[t] = np.sqrt(np.mean(r[t + 1:t + 1 + H] ** 2))
            dts = s.index.to_numpy().astype("int64")[1:]
            fin = np.isfinite(ev) & np.isfinite(rv) & (ev > 0) & (rv > 0)
            for k in EX:
                fin &= np.isfinite(ex[k])
            for k in EX:
                EX[k].append(ex[k][fin])
            EV.append(ev[fin]); RV.append(rv[fin]); DT.append(dts[fin])
    EX = {k: np.concatenate(v) for k, v in EX.items()}
    EV, RV, DT = map(np.concatenate, (EV, RV, DT))
    cut = np.quantile(DT, 0.6); tr = DT <= cut; te = ~tr

    names = ["ewma_s", "ewma_m", "ewma_l", "long_mean", "leverage", "momentum"]
    Xall = np.column_stack([np.ones(len(RV))] + [EX[n] for n in names])
    beta, *_ = np.linalg.lstsq(Xall[tr], RV[tr], rcond=None)
    stack = Xall @ beta
    stack = np.clip(stack, 1e-6, None)

    print("Signed vol-expert stack coefficients (train, predicting forward RV):")
    for n, b in zip(["const"] + names, beta):
        print(f"  {n:<10}: {b:+.4f}")
    print("\nForward-vol forecast skill on TEST (higher = better):")
    print(f"  EWMA(0.94)  : IC {_spearman(EV[te], RV[te]):+.3f}   "
          f"R2 {1 - np.sum((RV[te]-EV[te])**2)/np.sum((RV[te]-RV[te].mean())**2):+.3f}")
    print(f"  haf stack   : IC {_spearman(stack[te], RV[te]):+.3f}   "
          f"R2 {1 - np.sum((RV[te]-stack[te])**2)/np.sum((RV[te]-RV[te].mean())**2):+.3f}\n")

    # calibration: need forward cumulative return for coverage -> recompute per obs
    # (approx via RV*sqrt(H)*z two-sided). Use realised |cum| proxy: cum std ~ RV*sqrt(H).
    # Coverage: fraction of |z-scored cum| inside nominal band, per forecaster.
    # We reuse RV as the realised yardstick is circular; instead reload cum returns.
    print("In-domain pooled equity coverage (nominal vs empirical):")
    print(f"  {'nominal':>8}{'EWMA(0.94)':>12}{'haf stack':>11}")
    # reload cumulative forward returns aligned to the same finite mask ordering
    CY = _cum_returns()
    # align lengths defensively
    n = min(len(CY), len(EV))
    for p in NOMINAL_INTERVALS:
        z = _N.inv_cdf(0.5 + p / 2)
        ce = np.mean(np.abs(CY[te]) <= z * EV[te] * np.sqrt(H))
        cd = np.mean(np.abs(CY[te]) <= z * stack[te] * np.sqrt(H))
        print(f"  {p:>7.0%}{ce:>12.3f}{cd:>11.3f}")


def _cum_returns():
    """Forward cumulative returns pooled in the SAME order as main()'s finite mask."""
    CY = []
    for pat in DIRS:
        for path in sorted(glob.glob(pat)):
            try:
                s = load_close(path)
            except Exception:
                continue
            r = log_returns(s.to_numpy())
            if len(r) < MIN_ROWS or np.abs(r).max() > GLITCH:
                continue
            ev = _ewma_vol(r); ex = _experts(r)
            rv = np.full(len(r), np.nan); cy = np.full(len(r), np.nan)
            for t in range(len(r) - H):
                nxt = r[t + 1:t + 1 + H]
                rv[t] = np.sqrt(np.mean(nxt ** 2)); cy[t] = nxt.sum()
            fin = np.isfinite(ev) & np.isfinite(rv) & (ev > 0) & (rv > 0)
            for k in ex:
                fin &= np.isfinite(ex[k])
            CY.append(cy[fin])
    return np.concatenate(CY)


if __name__ == "__main__":
    main()
