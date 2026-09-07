"""Robustness pass on the Phase-5 market-neutral combined selection strategy.

(1) Cost sensitivity: net annualised Sharpe as costs rise from 0 to 50 bps.
(2) Sub-period stability: net Sharpe (at 10 bps) across thirds of the OOS period.
Rebuilds the same combined signal as run_phase5, then decomposes each rebalance
into (gross return, turnover) so costs/periods can be varied without refitting.
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

# self-contained (avoids run_phase5 name collision with the MC sibling project)
DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
Ns, Nl, L, K, H = 60, 1000, 30, 100, 20
VOLW, MOM_L, MOM_SKIP = 60, 120, 20
MIN_ROWS, GLITCH, SEL, MIN_NAMES = Nl + H + 60, 0.6, 0.2, 30


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
            tk = os.path.splitext(os.path.basename(path))[0] + \
                 (":uk" if "ukinvesting" in path else ":yf")
            out[tk] = (s, pd.Series(r, index=s.index[1:]))
    return out


def _rank_xs(df):
    return df.rank(axis=1).sub(0.5).div(df.notna().sum(axis=1), axis=0) - 0.5


def _build():
    data = _load()
    all_dates = pd.DatetimeIndex(sorted(set().union(*[r.index for _, r in data.values()])))
    cut = all_dates[int(0.6 * len(all_dates))]
    Wtr, ftr = [], []
    for s, r in data.values():
        logP = np.log(s.to_numpy())
        W, ends, _ = extract_windows(logP, L=L, stride=5, znorm=True)
        f = forward_return(logP, ends, H=H)
        dt = s.index.to_numpy()[ends]
        ok = np.isfinite(f) & np.isfinite(W).all(axis=1) & (dt <= cut.to_datetime64())
        Wtr.append(W[ok]); ftr.append(f[ok])
    km = KMeans(n_clusters=K, n_init=3, random_state=0).fit(np.vstack(Wtr))
    ftr = np.concatenate(ftr)
    edge = np.array([ftr[km.labels_ == c].mean() if (km.labels_ == c).any() else 0.0
                     for c in range(K)])

    rev, mom, shp, vol, fwd = {}, {}, {}, {}, {}
    for tk, (s, r) in data.items():
        rev[tk] = (r.rolling(Nl).mean() - r.rolling(Ns).mean()) / (r.rolling(Ns).std() / np.sqrt(Ns))
        mom[tk] = r.rolling(MOM_L).sum().shift(MOM_SKIP)
        vol[tk] = r.rolling(VOLW).std()
        fwd[tk] = r.rolling(H).sum().shift(-H)
        logP = np.log(s.to_numpy())
        W, ends, _ = extract_windows(logP, L=L, stride=1, znorm=True)
        good = np.isfinite(W).all(axis=1)
        lab = np.full(len(W), -1); lab[good] = km.predict(W[good])
        shp[tk] = pd.Series(np.where(lab >= 0, edge[lab], np.nan), index=s.index[ends])
    REV, MOM, SHP, VOL, FWD = (pd.DataFrame(d) for d in (rev, mom, shp, vol, fwd))
    idx = FWD.index
    combined = pd.concat([_rank_xs(REV.reindex(idx)), _rank_xs(MOM.reindex(idx)),
                          _rank_xs(SHP.reindex(idx))]).groupby(level=0).mean()
    rebal = all_dates[all_dates > cut][::H]
    return combined, FWD, VOL, rebal


def _components(SIG, FWD, VOL, rebal):
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
        k = max(1, int(SEL * len(s)))
        rank = s.rank()
        w = pd.Series(0.0, index=s.index)
        lo, sh_ = rank > len(s) - k, rank <= k
        w[lo] = (1 / v[lo]); w[lo] /= w[lo].sum() * 2
        w[sh_] = -(1 / v[sh_]); w[sh_] /= -w[sh_].sum() * 2
        gross.append(float((w * y).sum()))
        al = prev.index.union(w.index)
        turn.append(float((w.reindex(al).fillna(0) - prev.reindex(al).fillna(0)).abs().sum()))
        dts.append(d); prev = w
    return pd.to_datetime(dts), np.array(gross), np.array(turn)


def _sharpe(x):
    return x.mean() / x.std() * np.sqrt(252 / H) if x.std() > 0 else 0.0


def main():
    combined, FWD, VOL, rebal = _build()
    dts, gross, turn = _components(combined, FWD, VOL, rebal)
    print(f"rebalances: {len(gross)}   {dts[0].date()} -> {dts[-1].date()}\n")

    print("Cost sensitivity (annualised net Sharpe):")
    bps_list = [0, 5, 10, 20, 30, 50]
    sharpes = []
    for bps in bps_list:
        net = gross - (bps / 1e4) * turn
        sharpes.append(_sharpe(net))
        print(f"  {bps:>2} bps: {_sharpe(net):+.2f}")
    # break-even cost
    be = next((b for b, s in zip(bps_list, sharpes) if s <= 0), None)

    print("\nSub-period stability (net Sharpe @10bps, thirds of OOS):")
    net10 = gross - 10 / 1e4 * turn
    thirds = np.array_split(np.arange(len(net10)), 3)
    labels = []
    for i, ix in enumerate(thirds):
        lab = f"{dts[ix[0]].date()}..{dts[ix[-1]].date()}"
        labels.append(lab)
        print(f"  {lab}: {_sharpe(net10[ix]):+.2f}  (mean/period {net10[ix].mean():+.5f})")

    _plot(bps_list, sharpes, thirds, net10, labels, dts)
    print(f"\nbreak-even cost ~ {be if be is not None else '>50'} bps")
    print("saved phase5_robustness.png")


def _plot(bps_list, sharpes, thirds, net10, labels, dts):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.plot(bps_list, sharpes, "o-")
    a1.axhline(0, color="k", lw=0.7, ls=":")
    a1.set_xlabel("transaction cost (bps per unit turnover)")
    a1.set_ylabel("annualised net Sharpe"); a1.set_title("Cost sensitivity")
    subs = [_sharpe(net10[ix]) for ix in thirds]
    a2.bar(range(len(subs)), subs, color=["tab:green", "tab:orange", "tab:blue"])
    a2.set_xticks(range(len(subs))); a2.set_xticklabels(labels, fontsize=7, rotation=10)
    a2.axhline(0, color="k", lw=0.7, ls=":")
    a2.set_ylabel("net Sharpe @10bps"); a2.set_title("Sub-period stability")
    fig.tight_layout(); fig.savefig("phase5_robustness.png", dpi=110)


if __name__ == "__main__":
    main()
