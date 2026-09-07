"""Phase 11 (borrowing haf): signed vs equal-weight signal combination.

haf found a FIXED SIGNED stack (negative coefficients allowed, to subtract
correlated experts) beats a convex/equal-weight mix. Phase 8 combined our
directional signals (reversion, momentum, shape, depth) by equal-weight rank
averaging -- a convex mix. Here we fit a SIGNED least-squares combiner on the
train period (cross-sectionally demeaned forward returns ~ standardised signals)
and compare OOS IC and market-neutral selection Sharpe to the equal-weight mix.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from mc.data import load_close
from mc.returns import log_returns
from patterns.windows import extract_windows, forward_return

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
Ns, Nl, L, K, H, VOLW, MOM_L, MOM_SKIP, LD = 60, 1000, 30, 100, 20, 60, 120, 20, 80
MIN_ROWS, GLITCH, SEL, MIN_NAMES, COST = Nl + H + 60, 0.6, 0.2, 30, 0.001


def _load():
    out = {}
    for pat in DIRS:
        for path in sorted(glob.glob(pat)):
            try:
                s = load_close(path)
            except Exception:
                continue
            r = log_returns(s.to_numpy())
            if len(r) < MIN_ROWS or np.abs(r).max() > GLITCH:
                continue
            tk = os.path.splitext(os.path.basename(path))[0] + (":uk" if "ukinvesting" in path else ":yf")
            out[tk] = (s, pd.Series(r, index=s.index[1:]))
    return out


def _rank_xs(df):
    return df.rank(axis=1).sub(0.5).div(df.notna().sum(axis=1), axis=0) - 0.5


def _xs_ic(sig, FWD, rebal):
    ics = []
    for d in rebal:
        if d in sig.index and d in FWD.index:
            a, b = sig.loc[d].to_numpy(), FWD.loc[d].to_numpy()
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() >= MIN_NAMES:
                ra, rb = np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m]))
                ics.append(np.corrcoef(ra, rb)[0, 1])
    return float(np.nanmean(ics))


def _backtest(SIG, FWD, VOL, rebal):
    gross, turn = [], []
    prev = pd.Series(dtype=float)
    for d in rebal:
        if d not in SIG.index or d not in FWD.index:
            continue
        s, y, v = SIG.loc[d], FWD.loc[d], VOL.loc[d]
        ok = s.notna() & y.notna() & v.notna() & (v > 0)
        s, y, v = s[ok], y[ok], v[ok]
        if len(s) < MIN_NAMES:
            continue
        k = max(1, int(SEL * len(s))); rank = s.rank()
        w = pd.Series(0.0, index=s.index)
        lo, sh_ = rank > len(s) - k, rank <= k
        w[lo] = 1 / v[lo]; w[lo] /= w[lo].sum() * 2
        w[sh_] = -(1 / v[sh_]); w[sh_] /= -w[sh_].sum() * 2
        gross.append(float((w * y).sum()))
        al = prev.index.union(w.index)
        turn.append(float((w.reindex(al).fillna(0) - prev.reindex(al).fillna(0)).abs().sum()))
        prev = w
    gross, turn = np.array(gross), np.array(turn)
    net = gross - COST * turn
    per = 252 / H
    sh = lambda x: x.mean() / x.std() * np.sqrt(per) if x.std() > 0 else 0.0
    dd = float((np.cumsum(net) - np.maximum.accumulate(np.cumsum(net))).min())
    return sh(gross), sh(net), dd, turn.mean()


def main():
    data = _load()
    all_dates = pd.DatetimeIndex(sorted(set().union(*[r.index for _, r in data.values()])))
    cut = all_dates[int(0.6 * len(all_dates))]

    Wtr, ftr = [], []
    for s, r in data.values():
        logP = np.log(s.to_numpy())
        W, ends, _ = extract_windows(logP, L=L, stride=5, znorm=True)
        f = forward_return(logP, ends, H=H); dt = s.index.to_numpy()[ends]
        ok = np.isfinite(f) & np.isfinite(W).all(1) & (dt <= cut.to_datetime64())
        Wtr.append(W[ok]); ftr.append(f[ok])
    km = KMeans(n_clusters=K, n_init=3, random_state=0).fit(np.vstack(Wtr))
    ftr = np.concatenate(ftr)
    edge = np.array([ftr[km.labels_ == c].mean() if (km.labels_ == c).any() else 0.0 for c in range(K)])

    rev, mom, shp, dep, vol, fwd = {}, {}, {}, {}, {}, {}
    for tk, (s, r) in data.items():
        logP_s = pd.Series(np.log(s.to_numpy()), index=s.index)
        rev[tk] = (r.rolling(Nl).mean() - r.rolling(Ns).mean()) / (r.rolling(Ns).std() / np.sqrt(Ns))
        mom[tk] = r.rolling(MOM_L).sum().shift(MOM_SKIP)
        vol[tk] = r.rolling(VOLW).std()
        fwd[tk] = r.rolling(H).sum().shift(-H)
        dep[tk] = (logP_s.rolling(LD).max() - logP_s.rolling(LD).min()).reindex(r.index)
        W, ends, _ = extract_windows(np.log(s.to_numpy()), L=L, stride=1, znorm=True)
        good = np.isfinite(W).all(1); lab = np.full(len(W), -1); lab[good] = km.predict(W[good])
        shp[tk] = pd.Series(np.where(lab >= 0, edge[lab], np.nan), index=s.index[ends])
    REV, MOM, SHP, DEP, VOL, FWD = (pd.DataFrame(d) for d in (rev, mom, shp, dep, vol, fwd))
    idx = FWD.index
    Z = {"rev": _rank_xs(REV.reindex(idx)), "mom": _rank_xs(MOM.reindex(idx)),
         "shape": _rank_xs(SHP.reindex(idx)), "depth": _rank_xs(DEP.reindex(idx))}
    rebal = all_dates[all_dates > cut][::H]

    # equal-weight combine (Phase 8)
    eqw = pd.concat(list(Z.values())).groupby(level=0).mean()

    # signed LS combine: fit on TRAIN, cross-sectionally-demeaned forward ~ signals
    names = list(Z)
    tr_dates = all_dates[(all_dates <= cut)]
    rows_X, rows_y = [], []
    fwd_xs = FWD.sub(FWD.mean(axis=1), axis=0)  # cross-sectional demean per date
    for d in idx:
        if d > cut:
            continue
        yy = fwd_xs.loc[d]
        xs = np.column_stack([Z[n].loc[d].to_numpy() for n in names])
        m = np.isfinite(yy.to_numpy()) & np.isfinite(xs).all(1)
        if m.sum() >= MIN_NAMES:
            rows_X.append(xs[m]); rows_y.append(yy.to_numpy()[m])
    X = np.vstack(rows_X); y = np.concatenate(rows_y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    print("Signed LS coefficients (train):")
    for n, b in zip(names, beta):
        print(f"  {n:<7}: {b:+.4f}")
    signed = sum(beta[i] * Z[names[i]] for i in range(len(names)))

    print("\nOOS cross-sectional IC:")
    print(f"  equal-weight combine : {_xs_ic(eqw, FWD, rebal):+.4f}")
    print(f"  signed LS combine    : {_xs_ic(signed, FWD, rebal):+.4f}\n")

    print(f"{'book':<22}{'gross':>7}{'net':>7}{'maxDD':>8}{'turn':>7}")
    for nm, sg in [("equal-weight", eqw), ("signed LS", signed)]:
        gs, ns, dd, tn = _backtest(sg, FWD, VOL, rebal)
        print(f"{nm:<22}{gs:>+7.2f}{ns:>+7.2f}{dd:>+8.2f}{tn:>7.2f}")


if __name__ == "__main__":
    main()
