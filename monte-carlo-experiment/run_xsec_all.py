"""Full cross-asset OOS: all equities (2 sources) + FX/metals, daily.

Fixed models (EWMA benchmark, FHS gamma=0, FHS gamma=1) run walk-forward on
every series; we summarise per source and overall -- mean calibration, median
CRPS ratio, and the rate at which FHS beats EWMA. Per-series results are written
to scratchpad/xsec_results.csv for later reuse.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.data import load_close
from mc.returns import log_returns
from mc.forecasters import gaussian_ewma, fhs_ewma
from mc.walkforward import evaluate

T, MIN_HISTORY, STRIDE, NSIM = 10, 1000, 10, 4000
GLITCH = 0.6
SP = ("/private/tmp/claude-501/-Users-lance-Projects-share-jupyter-experiments-"
      "monte-carlo-experiment/ecdd1cf6-87c2-423e-ad16-60786d5aca36/scratchpad")

SOURCES = [
    ("yfinance", "/Users/lance/Projects/shares/data/yfinance/*.csv", 2500),
    ("ukinvest", "/Users/lance/Projects/shares/data/ukinvesting/*.csv", 2500),
    ("fx", f"{SP}/fx/*.csv", 1500),
]

MODELS = {
    "ewma": lambda: gaussian_ewma(lam=0.94, min_obs=MIN_HISTORY),
    "fhs0": lambda: fhs_ewma(lam=0.94, gamma=0.0, n_sims=NSIM, seed=0),
    "fhs1": lambda: fhs_ewma(lam=0.94, gamma=1.0, n_sims=NSIM, seed=0),
}


def _returns(path, min_rows):
    try:
        r = log_returns(load_close(path).to_numpy())
    except Exception:
        return None
    if len(r) < min_rows or not np.isfinite(r).all() or np.abs(r).max() > GLITCH:
        return None
    return r


RESULTS_CSV = f"{SP}/xsec_results.csv"


def _done_set() -> set:
    """(source, ticker) pairs already computed, for resume."""
    done = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as f:
            next(f, None)
            for line in f:
                parts = line.split(",")
                if len(parts) >= 2:
                    done.add((parts[0], parts[1]))
    return done


def main() -> None:
    done = _done_set()
    new_header = not os.path.exists(RESULTS_CSV)
    out = open(RESULTS_CSV, "a")
    if new_header:
        out.write("source,ticker,model,cov90,cov99,crps\n"); out.flush()

    remaining = []
    for src, pattern, min_rows in SOURCES:
        for path in sorted(glob.glob(pattern)):
            tk = os.path.splitext(os.path.basename(path))[0]
            if (src, tk) not in done:
                remaining.append((src, tk, path, min_rows))
    print(f"{len(done)} series already done; {len(remaining)} remaining", flush=True)

    for i, (src, tk, path, min_rows) in enumerate(remaining):
        r = _returns(path, min_rows)
        if r is None:
            print(f"[{i+1}/{len(remaining)}] {src}/{tk} SKIP (short/glitch)", flush=True)
            continue
        try:
            res = {m: evaluate(r, f(), T=T, min_history=MIN_HISTORY, stride=STRIDE)
                   for m, f in MODELS.items()}
        except Exception as e:
            print(f"[{i+1}/{len(remaining)}] {src}/{tk} ERROR {e}", flush=True)
            continue
        for m, v in res.items():
            out.write(f"{src},{tk},{m},{v.coverage[0.90][9]},"
                      f"{v.coverage[0.99][9]},{v.mean_crps[9]}\n")
        out.flush()
        print(f"[{i+1}/{len(remaining)}] {src}/{tk} ok", flush=True)
    out.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
