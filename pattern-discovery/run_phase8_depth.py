"""Phase 8: does the amplitude/DEPTH signal improve the selection strategy?

Adds two amplitude signals at scale L=80 to the Phase-5 market-neutral selection:
  depth  = trailing log peak-to-trough range (the Phase-7 validated signal)
  ddown  = current drawdown from the trailing peak, in vol units (buy-the-dip)
We measure their correlation with the existing signals (rev/mom/shape), their
standalone OOS cross-sectional IC, and whether adding the better one to the
combined score raises net Sharpe over the 3-signal book.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from mc.data import load_close
from mc.returns import log_returns
from patterns.windows import extract_windows, forward_return

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
Ns, Nl, L, K, H = 60, 1000, 30, 100, 20
VOLW, MOM_L, MOM_SKIP, LD = 60, 120, 20, 80
MIN_ROWS, GLITCH, SEL, MIN_NAMES, COST = Nl + H + 60, 0.6, 0.2, 30, 0.001


def _load():
    out = {}
    for pattern in DIRS:
        for path in sorted(glob.glob(pattern)):
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
                ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
                ics.append(np.corrcoef(ra, rb)[0, 1])
    return float(np.nanmean(ics))


def _backtest(SIG, FWD, VOL, rebal):
    dts, gross, turn = [], [], []
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
        dts.append(d); prev = w
    gross, turn = np.array(gross), np.array(turn)
    net = gross - COST * turn
    per = 252 / H
    sh = lambda x: x.mean() / x.std() * np.sqrt(per) if x.std() > 0 else 0.0
    dd = float((np.cumsum(net) - np.maximum.accumulate(np.cumsum(net))).min())
    return sh(gross), sh(net), dd, turn.mean(), pd.to_datetime(dts), np.cumsum(net)


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

    rev, mom, shp, dep, ddn, vol, fwd = {}, {}, {}, {}, {}, {}, {}
    for tk, (s, r) in data.items():
        logP_s = pd.Series(np.log(s.to_numpy()), index=s.index)
        vol_s = r.rolling(VOLW).std().reindex(s.index)
        rmax = logP_s.rolling(LD).max()
        rev[tk] = (r.rolling(Nl).mean() - r.rolling(Ns).mean()) / (r.rolling(Ns).std() / np.sqrt(Ns))
        mom[tk] = r.rolling(MOM_L).sum().shift(MOM_SKIP)
        vol[tk] = r.rolling(VOLW).std()
        fwd[tk] = r.rolling(H).sum().shift(-H)
        dep[tk] = (rmax - logP_s.rolling(LD).min()).reindex(r.index)         # peak-to-trough
        ddn[tk] = ((rmax - logP_s) / (vol_s * np.sqrt(LD))).reindex(r.index)  # drawdown from peak
        W, ends, _ = extract_windows(np.log(s.to_numpy()), L=L, stride=1, znorm=True)
        good = np.isfinite(W).all(1); lab = np.full(len(W), -1); lab[good] = km.predict(W[good])
        shp[tk] = pd.Series(np.where(lab >= 0, edge[lab], np.nan), index=s.index[ends])
    REV, MOM, SHP, DEP, DDN, VOL, FWD = (pd.DataFrame(d) for d in (rev, mom, shp, dep, ddn, vol, fwd))
    idx = FWD.index
    zrev, zmom, zshp = _rank_xs(REV.reindex(idx)), _rank_xs(MOM.reindex(idx)), _rank_xs(SHP.reindex(idx))
    zdep, zddn = _rank_xs(DEP.reindex(idx)), _rank_xs(DDN.reindex(idx))
    rebal = all_dates[all_dates > cut][::H]

    te = idx[idx > cut]
    def pcorr(a, b):
        a, b = a.reindex(te).to_numpy().ravel(), b.reindex(te).to_numpy().ravel()
        m = np.isfinite(a) & np.isfinite(b); return float(np.corrcoef(a[m], b[m])[0, 1])
    print("Depth-signal correlation with existing signals (test):")
    print(f"  depth : rev {pcorr(zdep,zrev):+.2f}  mom {pcorr(zdep,zmom):+.2f}  shape {pcorr(zdep,zshp):+.2f}")
    print(f"  ddown : rev {pcorr(zddn,zrev):+.2f}  mom {pcorr(zddn,zmom):+.2f}  shape {pcorr(zddn,zshp):+.2f}"
          f"   depth-ddown {pcorr(zdep,zddn):+.2f}\n")

    print("Standalone OOS cross-sectional IC:")
    for nm, sg in [("reversion", zrev), ("momentum", zmom), ("shape", zshp),
                   ("depth", zdep), ("ddown", zddn)]:
        print(f"  {nm:<10}: {_xs_ic(sg, FWD, rebal):+.4f}")

    best = zdep if _xs_ic(zdep, FWD, rebal) >= _xs_ic(zddn, FWD, rebal) else zddn
    best_nm = "depth" if best is zdep else "ddown"
    c3 = pd.concat([zrev, zmom, zshp]).groupby(level=0).mean()
    c4 = pd.concat([zrev, zmom, zshp, best]).groupby(level=0).mean()
    print(f"\n  combined-3 (rev+mom+shape)          IC {_xs_ic(c3, FWD, rebal):+.4f}")
    print(f"  combined-4 (+{best_nm})                 IC {_xs_ic(c4, FWD, rebal):+.4f}\n")

    print(f"{'book':<26}{'gross':>7}{'net':>7}{'maxDD':>8}{'turn':>7}")
    curves = {}
    for nm, sg in [("combined-3", c3), (f"combined-4 (+{best_nm})", c4), (f"{best_nm} only", best)]:
        gs, ns, dd, tn, dts, cur = _backtest(sg, FWD, VOL, rebal)
        curves[nm] = (dts, cur)
        print(f"{nm:<26}{gs:>+7.2f}{ns:>+7.2f}{dd:>+8.2f}{tn:>7.2f}")

    fig, ax = plt.subplots(figsize=(11, 5))
    for nm, (dts, cur) in curves.items():
        ax.plot(dts, cur, label=nm)
    ax.axhline(0, color="k", lw=0.7, ls=":"); ax.legend()
    ax.set_title("Does adding the amplitude/depth signal improve the book? (net, after costs)")
    ax.set_ylabel("cumulative net log return"); ax.set_xlabel("date")
    fig.tight_layout(); fig.savefig("phase8_depth.png", dpi=110)
    print("\nsaved phase8_depth.png")


if __name__ == "__main__":
    main()
