"""Test B: cross-sectional reversion vs momentum across the equity universe.

For each stock and date compute:
  reversion signal  = long_drift(Nl) - short_drift(Ns)   (high = below long-run -> expect rebound)
  momentum signal   = past return over L days, skipping the last SKIP days
At each rebalance (every H days) rank stocks cross-sectionally and measure:
  * rank IC = Spearman(signal, forward H-day return) across stocks, averaged over dates
  * a long-short quintile portfolio (long top signal / short bottom), its Sharpe
Signals use only past data; forward windows are non-overlapping (rebalance = H).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.data import load_close
from mc.returns import log_returns

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
Ns, Nl, H = 60, 1000, 20
L, SKIP = 120, 20          # momentum lookback / skip recent
MIN_ROWS = Nl + H + 60
GLITCH = 0.6
MIN_NAMES = 20             # need this many stocks cross-sectionally on a date


def _spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < MIN_NAMES:
        return np.nan
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def _series(path):
    try:
        s = load_close(path)
    except Exception:
        return None
    r = pd.Series(log_returns(s.to_numpy()), index=s.index[1:])
    if len(r) < MIN_ROWS or np.abs(r.to_numpy()).max() > GLITCH:
        return None
    return r


def main():
    rev, mom, fwd = {}, {}, {}
    used = []
    for pattern in DIRS:
        for path in sorted(glob.glob(pattern)):
            r = _series(path)
            if r is None:
                continue
            tk = os.path.splitext(os.path.basename(path))[0] + ":" + \
                 ("uk" if "ukinvesting" in path else "yf")
            rev[tk] = r.rolling(Nl).mean() - r.rolling(Ns).mean()
            mom[tk] = r.rolling(L).sum().shift(SKIP)
            fwd[tk] = r.rolling(H).sum().shift(-H)
            used.append(tk)
    print(f"universe: {len(used)} series\n")

    REV = pd.DataFrame(rev); MOM = pd.DataFrame(mom); FWD = pd.DataFrame(fwd)
    dates = REV.index.sort_values()
    rebal = dates[Nl::H]  # start after long-window warmup, step H

    out = {}
    for name, SIG in (("reversion", REV), ("momentum", MOM)):
        ics, ls = [], []
        for d in rebal:
            if d not in SIG.index or d not in FWD.index:
                continue
            s = SIG.loc[d].to_numpy(); y = FWD.loc[d].to_numpy()
            ic = _spearman(s, y)
            if np.isnan(ic):
                continue
            ics.append(ic)
            m = np.isfinite(s) & np.isfinite(y)
            sv, yv = s[m], y[m]
            k = max(1, len(sv) // 5)
            hi = yv[np.argsort(sv)[-k:]].mean()   # long top-signal
            lo = yv[np.argsort(sv)[:k]].mean()    # short bottom-signal
            ls.append(hi - lo)
        ics = np.array(ics); ls = np.array(ls)
        per_yr = 252 / H
        sharpe = ls.mean() / ls.std() * np.sqrt(per_yr) if ls.std() > 0 else 0.0
        out[name] = (ics, ls)
        print(f"=== {name} signal ===")
        print(f"  rebalances: {len(ics)}   mean rank IC: {ics.mean():+.4f}"
              f"   IC t-stat: {ics.mean()/ics.std()*np.sqrt(len(ics)):+.2f}")
        print(f"  long-short mean/period: {ls.mean():+.5f}   ann.Sharpe: {sharpe:+.2f}"
              f"   %periods>0: {np.mean(ls>0)*100:.0f}%\n")

    _plot(out, rebal)
    print("saved xsec_direction.png")


def _plot(out, rebal):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, (ics, ls) in out.items():
        a1.hist(ics, bins=25, histtype="step", label=f"{name} (mean {ics.mean():+.3f})")
        a2.plot(np.cumsum(ls), label=f"{name} L/S cum. return")
    a1.axvline(0, color="k", lw=0.7, ls=":")
    a1.set_xlabel("cross-sectional rank IC per rebalance"); a1.set_ylabel("count")
    a1.set_title("rank-IC distribution"); a1.legend(fontsize=8)
    a2.axhline(0, color="k", lw=0.7, ls=":")
    a2.set_xlabel("rebalance #"); a2.set_ylabel("cumulative L/S log return")
    a2.set_title(f"long-short equity curve (Ns={Ns},Nl={Nl},H={H})"); a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("xsec_direction.png", dpi=110)


if __name__ == "__main__":
    main()
