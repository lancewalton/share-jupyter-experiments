"""FX-scoped: does the intraday-RV vol forecast calibrate FX envelopes better?

Strictly FX -- no equities. Two causal one-day conditional-variance forecasts:
  EWMA(r^2)  : the daily-return filter used throughout
  HAR-RV     : OLS of r_t^2 on lagged intraday realised variance (daily/weekly/
               monthly), which also absorbs the overnight-gap scale via its fit
Build drift-zero 1-day envelopes from each and compare out-of-sample coverage
(does the better vol MEASUREMENT propagate to a better-calibrated envelope?).
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from mc.data import load_close
from mc.returns import log_returns

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


def _cov(r, sig, p):
    z = _N.inv_cdf(0.5 + p / 2)
    return float(np.mean(np.abs(r) <= z * sig))


def main():
    rvall = pd.read_csv(f"{SP}/fx_rv_all.csv", header=None,
                        names=["tk", "date", "rv", "nb"], parse_dates=["date"])
    pairs = [t for t, n in rvall.groupby("tk").size().items() if n >= MIN_DAYS]
    cov_e = {p: [] for p in LEVELS}; cov_r = {p: [] for p in LEVELS}
    used = []
    for pair in sorted(pairs):
        rv = rvall[rvall.tk == pair].set_index("date")["rv"].sort_index()
        rv = rv[rv > 0]
        try:
            s = load_close(f"{SP}/fx/{pair}.csv")
        except Exception:
            continue
        r = pd.Series(log_returns(s.to_numpy()), index=s.index[1:])
        df = pd.DataFrame({"rv": rv, "r": r.reindex(rv.index)}).dropna()
        df = df[df["r"].abs() < 0.6]
        if len(df) < MIN_DAYS:
            continue
        rr = df["r"].to_numpy(); RV = df["rv"].to_numpy(); n = len(rr)
        # causal HAR-RV features (all shifted by 1 day -> predict day t from < t)
        rvd = np.concatenate([[np.nan], RV[:-1]])
        rvw = pd.Series(RV).rolling(5).mean().shift(1).to_numpy()
        rvm = pd.Series(RV).rolling(22).mean().shift(1).to_numpy()
        v_e = _ewma_var(rr)
        ok = np.isfinite(rvm) & np.isfinite(v_e)
        cut = np.where(ok)[0][int(0.6 * ok.sum())]
        tr = (np.arange(n) <= cut) & ok
        te = (np.arange(n) > cut) & ok
        # fit r^2 ~ [1, rvd, rvw, rvm] on train (variance space)
        X = np.column_stack([np.ones(n), rvd, rvw, rvm])
        beta, *_ = np.linalg.lstsq(X[tr], (rr[tr] ** 2), rcond=None)
        v_r = np.maximum(X @ beta, 1e-12)
        for p in LEVELS:
            cov_e[p].append(_cov(rr[te], np.sqrt(v_e[te]), p))
            cov_r[p].append(_cov(rr[te], np.sqrt(v_r[te]), p))
        used.append(pair)

    print(f"FX pairs used: {len(used)}  (>= {MIN_DAYS} days)   OOS coverage:")
    print(f"  {'nominal':>8}{'EWMA(r2)':>11}{'HAR-RV':>10}{'RV closer':>11}")
    for p in LEVELS:
        e, rr_ = np.mean(cov_e[p]), np.mean(cov_r[p])
        closer = np.mean(np.abs(np.array(cov_r[p]) - p) < np.abs(np.array(cov_e[p]) - p))
        print(f"  {p:>7.0%}{e:>11.3f}{rr_:>10.3f}{closer:>10.0%}")
    # overall calibration error (mean |coverage - nominal| across levels)
    err_e = np.mean([np.abs(np.mean(cov_e[p]) - p) for p in LEVELS])
    err_r = np.mean([np.abs(np.mean(cov_r[p]) - p) for p in LEVELS])
    print(f"\nmean |coverage - nominal|:  EWMA {err_e:.4f}   HAR-RV {err_r:.4f}"
          f"   ({(1-err_r/err_e)*100:+.0f}% error)")


if __name__ == "__main__":
    main()
