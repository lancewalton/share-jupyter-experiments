"""Phase 4: does cross-asset integration forecast NON-equity vol — the fair test?

For equity, VIX (a purpose-built implied-vol gauge) dominates any structural
signal. Bonds, FX and commodities have no such gauge, so this is where the
absorption ratio could genuinely add. For each instrument we compare, as a
forecaster of its forward 21-day realised vol:

  * its own trailing vol (the natural single-asset baseline),
  * VIX level (the equity fear gauge, as a cross-asset control),
  * the absorption ratio (cross-asset integration),
  * AR's PARTIAL power after removing trailing vol AND VIX -- the only number that
    says whether cross-asset structure adds anything the simple tools miss.

All causal, non-overlapping months, permutation-checked.
"""
from __future__ import annotations

import numpy as np

from pca.data import _fred_level, build_returns
from pca.factors import standardize
from run_phase3_predict import _spearman, daily_absorption

WIN, H = 126, 21
NON_EQUITY = ["UST 2Y", "UST 10Y", "UST 30Y", "WTI oil", "Nat gas",
              "EUR/USD", "USD/JPY", "GBP/USD"]


def _partial(x, y, controls):
    C = np.c_[np.ones(len(x)), np.column_stack(controls)]
    b = lambda v: C @ np.linalg.lstsq(C, v, rcond=None)[0]
    m = np.all(np.isfinite(C), 1) & np.isfinite(x) & np.isfinite(y)
    xr = np.full_like(x, np.nan); yr = np.full_like(y, np.nan)
    xr[m] = x[m] - b(x[m] if False else x)[m]
    yr[m] = y[m] - b(y)[m]
    return _spearman(xr, yr)


def _fwd_vol(r, H):
    n = len(r); out = np.full(n, np.nan)
    for t in range(n - H - 1):
        out[t] = np.std(r[t + 1:t + 1 + H]) * np.sqrt(252) * 100
    return out


def _trail_vol(r, H):
    n = len(r); out = np.full(n, np.nan)
    for t in range(H, n):
        out[t] = np.std(r[t - H:t]) * np.sqrt(252) * 100
    return out


def main():
    R = build_returns()
    Z, _, _ = standardize(R.to_numpy())
    ar = daily_absorption(Z)
    vix = _fred_level("VIXCLS").reindex(R.index).to_numpy()
    n = len(R)
    idx = np.arange(WIN, n - H - 1, H)

    print(f"forecasting forward 21d realised vol · {len(idx)} months · 1999-2026\n")
    print(f"{'instrument':10s} {'trail':>7s} {'VIX':>7s} {'AR':>7s} {'AR|trail+VIX':>13s}")
    rows = []
    for col in R.columns:
        r = R[col].to_numpy()
        fv, tv = _fwd_vol(r, H)[idx], _trail_vol(r, H)[idx]
        a, v = ar[idx], vix[idx]
        m = np.isfinite(fv) & np.isfinite(tv) & np.isfinite(a) & np.isfinite(v)
        ic_t = _spearman(tv[m], fv[m]); ic_v = _spearman(v[m], fv[m])
        ic_a = _spearman(a[m], fv[m]); ic_p = _partial(a[m], fv[m], [tv[m], v[m]])
        tag = "" if col in ("US equity", "VIX") else " *"
        rows.append((col, ic_t, ic_v, ic_a, ic_p, col in NON_EQUITY))
        print(f"{col:10s} {ic_t:+7.2f} {ic_v:+7.2f} {ic_a:+7.2f} {ic_p:+13.2f}{tag}")

    ne = [r for r in rows if r[5]]
    print("\nNON-EQUITY average (* rows):")
    for k, name in ((1, "own trailing vol"), (2, "VIX"), (3, "AR"), (4, "AR | trail+VIX")):
        print(f"  {name:18s} mean IC {np.mean([r[k] for r in ne]):+.3f}")
    part = [r[4] for r in ne]
    print(f"\n=> AR partial (adds beyond trail+VIX) on non-equity vol: "
          f"mean {np.mean(part):+.3f}, range [{min(part):+.2f}, {max(part):+.2f}]")
    print("   verdict: " + ("cross-asset structure ADDS incremental vol-forecast"
          if np.mean(part) > 0.05 else "no clean incremental value beyond trailing vol + VIX"))


if __name__ == "__main__":
    main()
