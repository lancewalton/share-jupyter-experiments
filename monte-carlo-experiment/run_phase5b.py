"""Phase 5b: does a GJR leverage term (gamma) improve FHS calibration?"""
from __future__ import annotations
import numpy as np
from mc.data import close_array
from mc.returns import log_returns
from mc.forecasters import gaussian_ewma, fhs_ewma
from mc.walkforward import evaluate, NOMINAL_INTERVALS

T, MIN_HISTORY, STRIDE = 10, 1000, 5
forecasters = {
    "gaussian_ewma(0.94)": gaussian_ewma(lam=0.94, min_obs=MIN_HISTORY),
    "fhs gamma=0.0": fhs_ewma(lam=0.94, gamma=0.0, n_sims=10_000, seed=0),
    "fhs gamma=0.5": fhs_ewma(lam=0.94, gamma=0.5, n_sims=10_000, seed=0),
    "fhs gamma=1.0": fhs_ewma(lam=0.94, gamma=1.0, n_sims=10_000, seed=0),
    "fhs gamma=1.5": fhs_ewma(lam=0.94, gamma=1.5, n_sims=10_000, seed=0),
}
r = log_returns(close_array())
print(f"log returns: {len(r)}  T={T}  min_history={MIN_HISTORY}  stride={STRIDE}\n")
for name, f in forecasters.items():
    res = evaluate(r, f, T=T, min_history=MIN_HISTORY, stride=STRIDE)
    print(f"=== {name}  (n={res.n_origins}) ===")
    for p in NOMINAL_INTERVALS:
        v = res.coverage[p]
        print(f"    {p:>4.0%}:  {v[0]:.3f}  {v[4]:.3f}  {v[9]:.3f}")
    print(f"  CRPS  h1={res.mean_crps[0]:.5f}  h5={res.mean_crps[4]:.5f}  h10={res.mean_crps[9]:.5f}\n")
