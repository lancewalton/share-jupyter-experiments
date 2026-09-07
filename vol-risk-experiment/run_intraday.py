"""Intraday realised volatility vs daily-return EWMA for forecasting vol.

Intraday data lets us MEASURE a day's volatility from ~1400 minute bars
(realised variance = sum of intraday squared returns) instead of a single noisy
daily squared return. We test whether that better measurement forecasts future
volatility better -- HAR-RV (Corsi) on intraday RV vs the RiskMetrics EWMA on
daily r^2 we have used throughout. Target: next-day log realised vol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.data import load_close
from mc.returns import log_returns

SP = ("/private/tmp/claude-501/-Users-lance-Projects-share-jupyter-experiments-"
      "monte-carlo-experiment/ecdd1cf6-87c2-423e-ad16-60786d5aca36/scratchpad")
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]


def _spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    ra, rb = np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def _r2(y, p, tr):
    te = ~tr
    return 1 - np.sum((y[te] - p[te]) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)


def _fit_predict(cols, y, tr):
    X = np.column_stack([np.ones(len(y))] + cols)
    beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
    return X @ beta


def _ewma_var(r, lam=0.94, burn=40):
    v = np.empty(len(r)); v[:burn] = np.var(r[:burn])
    for i in range(burn, len(r)):
        v[i] = lam * v[i - 1] + (1 - lam) * r[i - 1] ** 2
    return v


def main():
    rvdf = pd.read_csv(f"{SP}/fx_rv.csv", header=None,
                       names=["tk", "date", "rv", "nb"], parse_dates=["date"])
    rows = []
    persist_intraday, persist_daily = [], []
    for pair in PAIRS:
        rv = rvdf[rvdf.tk == pair].set_index("date")["rv"].sort_index()
        rv = rv[rv > 0]
        s = load_close(f"{SP}/fx/{pair}.csv")
        r = pd.Series(log_returns(s.to_numpy()), index=s.index[1:])
        df = pd.DataFrame({"rv": rv, "r": r.reindex(rv.index)}).dropna()
        df = df[df["r"].abs() < 0.6]
        lrv = np.log(df["rv"].to_numpy())
        rday = df["r"].to_numpy()
        n = len(lrv)

        # HAR components (causal): log of daily / weekly / monthly average RV
        RV = df["rv"].to_numpy()
        har_d = np.log(RV)
        har_w = np.log(pd.Series(RV).rolling(5).mean().to_numpy())
        har_m = np.log(pd.Series(RV).rolling(22).mean().to_numpy())
        ewma = np.log(_ewma_var(rday))                 # daily-r^2 EWMA (our standard)

        y = np.full(n, np.nan); y[:-1] = lrv[1:]        # target: next-day log RV
        ok = np.isfinite(y) & np.isfinite(har_m) & np.isfinite(ewma)
        idx = np.where(ok)[0]
        cut = idx[int(0.6 * len(idx))]
        tr = (np.arange(n) <= cut) & ok
        te = (np.arange(n) > cut) & ok

        def score(pred):
            return _r2(y[ok], pred[ok], tr[ok]), _spearman(pred[te], y[te])

        p_har = _fit_predict([har_d, har_w, har_m], np.nan_to_num(y), tr)
        p_ewma = _fit_predict([ewma], np.nan_to_num(y), tr)
        r2_h, ic_h = score(p_har)
        r2_e, ic_e = score(p_ewma)
        rows.append((pair, r2_e, ic_e, r2_h, ic_h))
        # measurement quality: does today's RV predict tomorrow's RV better than r^2 does?
        persist_intraday.append(_spearman(har_d[te], y[te]))
        persist_daily.append(_spearman(np.log(rday ** 2 + 1e-12)[te], y[te]))

    print("Forecasting next-day log realised vol (OOS):")
    print(f"{'pair':<8}{'EWMA r2  R2':>13}{'IC':>7}   {'HAR-RV  R2':>12}{'IC':>7}")
    for pair, r2e, ice, r2h, ich in rows:
        print(f"{pair:<8}{r2e:>13.3f}{ice:>7.3f}   {r2h:>12.3f}{ich:>7.3f}")
    r2e = np.mean([x[1] for x in rows]); r2h = np.mean([x[3] for x in rows])
    print(f"\nmean OOS R2:  daily EWMA(r^2) {r2e:.3f}   ->   intraday HAR-RV {r2h:.3f}"
          f"   ({(r2h-r2e)/max(r2e,1e-6)*100:+.0f}%)")
    print(f"as a single-lag SIGNAL, IC(today's vol proxy, tomorrow's RV):")
    print(f"  daily r^2      : {np.mean(persist_daily):+.3f}")
    print(f"  intraday RV    : {np.mean(persist_intraday):+.3f}   (cleaner measurement)")

    _plot(rows, rvdf, r)
    print("\nsaved intraday.png")


def _plot(rows, rvdf, r):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(rows)); w = 0.36
    a1.bar(x - w/2, [r[1] for r in rows], w, label="daily EWMA(r²)", color="tab:orange")
    a1.bar(x + w/2, [r[3] for r in rows], w, label="intraday HAR-RV", color="tab:blue")
    a1.set_xticks(x); a1.set_xticklabels([r[0] for r in rows])
    a1.set_ylabel("OOS R² (next-day log RV)"); a1.legend()
    a1.set_title("Intraday realised vol forecasts better")
    # RV vs |daily return| for EURUSD: RV is a smoother measure
    eur = rvdf[rvdf.tk == "EURUSD"].set_index("date").sort_index()
    a2.plot(eur.index, np.sqrt(eur["rv"]), color="tab:blue", lw=0.7,
            label="√RV (intraday)")
    a2.set_ylabel("daily volatility (EURUSD)"); a2.set_title("Realised vol from ~1400 bars/day")
    a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("intraday.png", dpi=110)


if __name__ == "__main__":
    main()
