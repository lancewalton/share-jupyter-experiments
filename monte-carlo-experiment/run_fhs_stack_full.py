"""Full cross-asset OOS: FHS-EWMA vs FHS-stack over all ~193 series (resumable).

For each series, fit the stack conditional-variance coefficients on its OWN train
half, then evaluate both forecasters strictly out-of-sample on the second half.
Per-series results (h=10 coverage at 5 levels, the h=1 tail, and h=10 CRPS) are
appended to a checkpoint CSV, so the run resumes after any interruption.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from mc.data import load_close
from mc.returns import log_returns
from mc.forecasters import fhs_ewma, fhs_stack, fit_stack_coeffs
from mc.walkforward import evaluate, NOMINAL_INTERVALS

T, STRIDE, NSIM, GLITCH = 10, 10, 6000, 0.6
SP = ("/private/tmp/claude-501/-Users-lance-Projects-share-jupyter-experiments-"
      "monte-carlo-experiment/ecdd1cf6-87c2-423e-ad16-60786d5aca36/scratchpad")
RESULTS = f"{SP}/fhs_stack_full.csv"
SOURCES = [
    ("yfinance", "/Users/lance/Projects/shares/data/yfinance/*.csv", 1500),
    ("ukinvest", "/Users/lance/Projects/shares/data/ukinvesting/*.csv", 1500),
    ("fx", f"{SP}/fx/*.csv", 1500),
]


def _done():
    d = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            next(f, None)
            for line in f:
                p = line.split(",")
                if len(p) >= 2:
                    d.add((p[0], p[1]))
    return d


def _returns(path, min_rows):
    try:
        r = log_returns(load_close(path).to_numpy())
    except Exception:
        return None
    if len(r) < min_rows or not np.isfinite(r).all() or np.abs(r).max() > GLITCH:
        return None
    return r


def main():
    done = _done()
    new = not os.path.exists(RESULTS)
    out = open(RESULTS, "a")
    if new:
        out.write("source,ticker,model,cov50,cov75,cov90,cov95,cov99,cov99_h1,crps10\n")
        out.flush()

    remaining = []
    for src, pat, mr in SOURCES:
        for path in sorted(glob.glob(pat)):
            tk = os.path.splitext(os.path.basename(path))[0]
            if (src, tk) not in done:
                remaining.append((src, tk, path, mr))
    print(f"{len(done)} done; {len(remaining)} remaining", flush=True)

    for i, (src, tk, path, mr) in enumerate(remaining):
        r = _returns(path, mr)
        if r is None:
            print(f"[{i+1}/{len(remaining)}] {src}/{tk} skip", flush=True)
            continue
        half = len(r) // 2
        try:
            coeffs = fit_stack_coeffs(r[:half])
            models = {"fhs_ewma": fhs_ewma(lam=0.94, n_sims=NSIM, seed=0),
                      "fhs_stack": fhs_stack(coeffs, n_sims=NSIM, seed=0)}
            for name, fc in models.items():
                res = evaluate(r, fc, T=T, min_history=half, stride=STRIDE)
                c = res.coverage
                out.write(f"{src},{tk},{name},{c[0.50][9]},{c[0.75][9]},{c[0.90][9]},"
                          f"{c[0.95][9]},{c[0.99][9]},{c[0.99][0]},{res.mean_crps[9]}\n")
            out.flush()
            print(f"[{i+1}/{len(remaining)}] {src}/{tk} ok", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(remaining)}] {src}/{tk} ERROR {e}", flush=True)
    out.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
