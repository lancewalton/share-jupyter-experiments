"""FX daily calibration: does adding the OVERNIGHT gap to intraday RV fix it?

Earlier, HAR-RV (intraday only) under-covered the DAILY return's tail because it
misses the overnight gap. Now, from per-day open/close, reconstruct
    total daily variance = RV(intraday) + overnight_return^2
and test whether a forecast built on it calibrates the daily close-to-close
return as well as, or better than, the daily-return EWMA. Three forecasts:
  EWMA(r_daily^2)  -- the baseline used throughout
  HAR-RV           -- intraday only (the one that under-covered)
  HAR-total        -- RV + overnight^2  (the fix)
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
    """causal daily/weekly/monthly features of a variance series x (lagged 1)."""
    s = pd.Series(x)
    return (np.concatenate([[np.nan], x[:-1]]),
            s.rolling(5).mean().shift(1).to_numpy(),
            s.rolling(22).mean().shift(1).to_numpy())


def _fit_var(feats, target, tr, n):
    X = np.column_stack([np.ones(n)] + list(feats))
    beta, *_ = np.linalg.lstsq(X[tr], target[tr], rcond=None)
    return np.maximum(X @ beta, 1e-12)


def _cov(r, sig, p):
    return float(np.mean(np.abs(r) <= _N.inv_cdf(0.5 + p / 2) * sig))


def main():
    oc = pd.read_csv(f"{SP}/fx_oc.csv", header=None,
                     names=["tk", "date", "open", "close", "rv", "nb"], parse_dates=["date"])
    names = ["EWMA(r²)", "HAR-RV", "HAR-total(RV+ON²)"]
    cov = {nm: {p: [] for p in LEVELS} for nm in names}
    used = []
    for pair, g in oc.groupby("tk"):
        g = g.sort_values("date")
        if len(g) < MIN_DAYS:
            continue
        op, cl, RV = g["open"].to_numpy(), g["close"].to_numpy(), g["rv"].to_numpy()
        if (op <= 0).any() or (cl <= 0).any():
            continue
        logc = np.log(cl); logo = np.log(op)
        r_daily = np.concatenate([[np.nan], np.diff(logc)])          # close-to-close
        overnight = logo - np.concatenate([[np.nan], logc[:-1]])      # open - prev close
        if np.abs(np.nan_to_num(r_daily)).max() > 0.6:
            continue
        total = RV + np.nan_to_num(overnight) ** 2                    # total daily variance
        n = len(r_daily)
        r2 = np.nan_to_num(r_daily) ** 2

        v_e = _ewma_var(np.nan_to_num(r_daily))
        okmask = np.isfinite(overnight) & np.isfinite(v_e)
        # need HAR features finite (monthly rolling) -> start later
        f_rv = _har(RV); f_tot = _har(total)
        ok = okmask & np.isfinite(f_tot[2]) & np.isfinite(r_daily)
        if ok.sum() < MIN_DAYS:
            continue
        cut = np.where(ok)[0][int(0.6 * ok.sum())]
        tr = (np.arange(n) <= cut) & ok
        te = (np.arange(n) > cut) & ok
        v_rv = _fit_var(f_rv, r2, tr, n)
        v_tot = _fit_var(f_tot, r2, tr, n)
        for nm, v in zip(names, [v_e, v_rv, v_tot]):
            for p in LEVELS:
                cov[nm][p].append(_cov(r_daily[te], np.sqrt(v[te]), p))
        used.append(pair)

    print(f"FX pairs: {len(used)}   DAILY close-to-close coverage (OOS):")
    print(f"  {'nominal':>8}" + "".join(f"{nm:>20}" for nm in names))
    for p in LEVELS:
        print(f"  {p:>7.0%}" + "".join(f"{np.mean(cov[nm][p]):>20.3f}" for nm in names))
    print("\n  mean |coverage - nominal|:")
    for nm in names:
        err = np.mean([abs(np.mean(cov[nm][p]) - p) for p in LEVELS])
        print(f"    {nm:<22}{err:.4f}")


if __name__ == "__main__":
    main()
