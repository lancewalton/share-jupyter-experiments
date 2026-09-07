"""Intraday-book calibration: RV's natural home (open->close, no overnight).

For a book that closes before end of day, the exposure is the INTRADAY return
r_intra = log(close/open) -- no overnight gap. Realised variance measures its
variance precisely. Test which forecast calibrates r_intra best:
  HAR-RV            -- right target, precise (intraday RV)
  EWMA(r_intra^2)   -- right target, noisy (one obs/day)
  EWMA(r_daily^2)   -- WRONG target: carries overnight variance the intraday
                       trader never bears -> should over-cover (envelope too wide)
Strictly FX.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

SP = ("/private/tmp/claude-501/-Users-lance-Projects-share-jupyter-experiments-"
      "monte-carlo-experiment/ecdd1cf6-87c2-423e-ad16-60786d5aca36/scratchpad")
LEVELS = [0.50, 0.75, 0.90, 0.95, 0.99]
MIN_DAYS = 1500
_N = NormalDist()


def _ewma_var(r, lam=0.94, burn=40):
    v = np.empty(len(r)); v[:burn] = np.var(r[:burn])
    for i in range(burn, len(r)):
        v[i] = lam * v[i - 1] + (1 - lam) * r[i - 1] ** 2
    return v


def _har(x):
    s = pd.Series(x)
    return (np.concatenate([[np.nan], x[:-1]]),
            s.rolling(5).mean().shift(1).to_numpy(),
            s.rolling(22).mean().shift(1).to_numpy())


def _cov(r, sig, p):
    return float(np.mean(np.abs(r) <= _N.inv_cdf(0.5 + p / 2) * sig))


def main():
    oc = pd.read_csv(f"{SP}/fx_oc.csv", header=None,
                     names=["tk", "date", "open", "close", "rv", "nb"], parse_dates=["date"])
    names = ["HAR-RV", "EWMA(r_intra²)", "EWMA(r_daily²)"]
    cov = {nm: {p: [] for p in LEVELS} for nm in names}
    used = []
    for pair, g in oc.groupby("tk"):
        g = g.sort_values("date")
        if len(g) < MIN_DAYS:
            continue
        op, cl, RV = g["open"].to_numpy(), g["close"].to_numpy(), g["rv"].to_numpy()
        if (op <= 0).any() or (cl <= 0).any():
            continue
        r_intra = np.log(cl / op)                                    # open -> close
        r_daily = np.concatenate([[np.nan], np.diff(np.log(cl))])    # close -> close
        if np.abs(r_intra).max() > 0.6 or np.abs(np.nan_to_num(r_daily)).max() > 0.6:
            continue
        n = len(r_intra)
        v_id = _ewma_var(r_intra)
        v_dl = _ewma_var(np.nan_to_num(r_daily))
        f_rv = _har(RV)
        ok = np.isfinite(f_rv[2]) & np.isfinite(r_daily)
        if ok.sum() < MIN_DAYS:
            continue
        cut = np.where(ok)[0][int(0.6 * ok.sum())]
        tr = (np.arange(n) <= cut) & ok; te = (np.arange(n) > cut) & ok
        X = np.column_stack([np.ones(n)] + list(f_rv))
        beta, *_ = np.linalg.lstsq(X[tr], (r_intra[tr] ** 2), rcond=None)
        v_rv = np.maximum(X @ beta, 1e-14)
        for nm, v in zip(names, [v_rv, v_id, v_dl]):
            for p in LEVELS:
                cov[nm][p].append(_cov(r_intra[te], np.sqrt(v[te]), p))
        used.append(pair)

    print(f"FX pairs: {len(used)}   INTRADAY return (open->close) coverage (OOS):")
    print(f"  {'nominal':>8}" + "".join(f"{nm:>18}" for nm in names))
    for p in LEVELS:
        print(f"  {p:>7.0%}" + "".join(f"{np.mean(cov[nm][p]):>18.3f}" for nm in names))
    print("\n  mean |coverage - nominal|:")
    for nm in names:
        err = np.mean([abs(np.mean(cov[nm][p]) - p) for p in LEVELS])
        print(f"    {nm:<18}{err:.4f}")
    # how much wider is the daily-EWMA envelope than it should be (95%)?
    print(f"\n  95% coverage: HAR-RV {np.mean(cov['HAR-RV'][0.95]):.3f} vs "
          f"daily-EWMA {np.mean(cov['EWMA(r_daily²)'][0.95]):.3f} "
          f"(daily carries overnight -> too wide)")


if __name__ == "__main__":
    main()
