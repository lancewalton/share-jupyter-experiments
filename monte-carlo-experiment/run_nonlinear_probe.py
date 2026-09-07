"""Probe: is there exploitable non-linearity a NN could capture?

Cheapest rung of the escalation. Pool (stock, date) observations, split
temporally (train early / test late), and compare, on the TEST set, the
cross-sectional rank IC of:
  * the single linear signal (drift divergence)              -- baseline
  * a linear-in-INTERACTIONS OLS (div, z, vol, mom + products) -- first non-linearity
Plus a conditional IC by volatility tercile: if the reversion IC is roughly flat
across vol regimes, there is little interaction structure for a NN to exploit.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from mc.data import load_close
from mc.returns import log_returns

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
Ns, Nl, H, VOLW, L, SKIP = 60, 1000, 20, 60, 120, 20
MIN_ROWS, GLITCH, MIN_NAMES = Nl + H + 60, 0.6, 20


def _series(path):
    try:
        s = load_close(path)
    except Exception:
        return None
    r = pd.Series(log_returns(s.to_numpy()), index=s.index[1:])
    if len(r) < MIN_ROWS or np.abs(r.to_numpy()).max() > GLITCH:
        return None
    return r


def _panel():
    cols = {}
    for pattern in DIRS:
        for path in sorted(glob.glob(pattern)):
            r = _series(path)
            if r is None:
                continue
            tk = os.path.splitext(os.path.basename(path))[0] + \
                 (":uk" if "ukinvesting" in path else ":yf")
            sd, ld = r.rolling(Ns).mean(), r.rolling(Nl).mean()
            se = r.rolling(Ns).std() / np.sqrt(Ns)
            cols[tk] = pd.DataFrame({
                "div": ld - sd, "z": (ld - sd) / se, "vol": r.rolling(VOLW).std(),
                "mom": r.rolling(L).sum().shift(SKIP), "y": r.rolling(H).sum().shift(-H),
            })
    return cols


def _rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def _xs_ic(pred, y, dates):
    """mean cross-sectional (per-date) rank IC."""
    ics = []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < MIN_NAMES:
            continue
        p, yy = pred[m], y[m]
        ics.append(np.corrcoef(_rank(p), _rank(yy))[0, 1])
    return float(np.nanmean(ics)), len(ics)


def main():
    cols = _panel()
    all_dates = sorted(set().union(*[df.index for df in cols.values()]))
    rebal = pd.DatetimeIndex(all_dates)[Nl::H]

    rows = []
    for tk, df in cols.items():
        sub = df.reindex(rebal).dropna()
        for d, rr in sub.iterrows():
            rows.append((d, rr["div"], rr["z"], rr["vol"], rr["mom"], rr["y"]))
    R = pd.DataFrame(rows, columns=["date", "div", "z", "vol", "mom", "y"])
    fcols = ["div", "z", "vol", "mom", "y"]
    R = R[np.isfinite(R[fcols].to_numpy()).all(axis=1)].reset_index(drop=True)
    print(f"pooled obs: {len(R)}   dates: {R['date'].nunique()}   "
          f"stocks: {len(cols)}\n")

    # temporal split
    cut = R["date"].quantile(0.6)
    tr, te = R[R["date"] <= cut], R[R["date"] > cut]
    print(f"train obs {len(tr)} (<= {cut.date()}),  test obs {len(te)} (> {cut.date()})\n")

    def z(col, ref):  # standardise using train stats
        return (col - ref.mean()) / (ref.std() + 1e-12)

    feats = ["div", "z", "vol", "mom"]
    Xtr = {f: z(tr[f], tr[f]).to_numpy() for f in feats}
    Xte = {f: z(te[f], tr[f]).to_numpy() for f in feats}

    def design(X):
        base = [np.ones_like(X["div"]), X["div"], X["z"], X["vol"], X["mom"]]
        inter = [X["div"] * X["vol"], X["div"] * X["z"], X["z"] * X["vol"],
                 X["div"] ** 2, X["vol"] ** 2]
        return np.column_stack(base + inter)

    beta, *_ = np.linalg.lstsq(design(Xtr), tr["y"].to_numpy(), rcond=None)
    pred_model = design(Xte) @ beta

    ic_lin, n1 = _xs_ic(te["div"].to_numpy(), te["y"].to_numpy(), te["date"].to_numpy())
    ic_z, _ = _xs_ic(te["z"].to_numpy(), te["y"].to_numpy(), te["date"].to_numpy())
    ic_mod, n2 = _xs_ic(pred_model, te["y"].to_numpy(), te["date"].to_numpy())
    print("TEST-set cross-sectional rank IC (higher = better ranker):")
    print(f"  linear single signal  (div)     : {ic_lin:+.4f}")
    print(f"  linear single signal  (z-score) : {ic_z:+.4f}")
    print(f"  OLS with interactions (9 terms) : {ic_mod:+.4f}   ({n2} test dates)\n")

    # conditional IC by volatility tercile (whole sample) -> interaction with vol?
    print("Reversion IC (div vs y) by volatility tercile — flat => little interaction:")
    vt = pd.qcut(R["vol"], 3, labels=["low vol", "mid vol", "high vol"])
    for lab in ["low vol", "mid vol", "high vol"]:
        sub = R[vt == lab]
        ic, nd = _xs_ic(sub["div"].to_numpy(), sub["y"].to_numpy(), sub["date"].to_numpy())
        print(f"  {lab:<9}: IC {ic:+.4f}")


if __name__ == "__main__":
    main()
