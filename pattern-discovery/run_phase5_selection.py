"""Phase 5: market-neutral cross-sectional SELECTION backtest.

Not "buy on a raw signal" but: each rebalance, rank all instruments by a combined
score, go long the best / short the worst (market-neutral, inverse-vol sized),
with a no-trade band. Signals combined: reversion divergence (from the MC work),
momentum, and the L=30 shape-cluster edge. We first measure signal independence
and standalone vs combined OOS IC (does combining amplify?), then run the
backtest on the TEST period with turnover costs and a market-beta check.
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
VOLW, MOM_L, MOM_SKIP = 60, 120, 20
MIN_ROWS, GLITCH = Nl + H + 60, 0.6
SEL, ZGATE, COST, MIN_NAMES = 0.2, 0.5, 0.001, 30


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


def _rank_xs(df):  # cross-sectional standardise per date (rank -> ~uniform -> center)
    return df.rank(axis=1).sub(0.5).div(df.notna().sum(axis=1), axis=0) - 0.5


def _xs_ic(sig, fwd, dates):
    ics = []
    for d in dates:
        if d in sig.index and d in fwd.index:
            a, b = sig.loc[d].to_numpy(), fwd.loc[d].to_numpy()
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() >= MIN_NAMES:
                ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
                ics.append(np.corrcoef(ra, rb)[0, 1])
    return float(np.nanmean(ics))


def main():
    data = _load()
    all_dates = pd.DatetimeIndex(sorted(set().union(*[r.index for _, r in data.values()])))
    cut = all_dates[int(0.6 * len(all_dates))]
    print(f"universe {len(data)}   split at {cut.date()}\n")

    # --- fit shape clustering on TRAIN windows pooled across stocks ---
    Wtr, ftr = [], []
    for s, r in data.values():
        logP = np.log(s.to_numpy())
        W, ends, _ = extract_windows(logP, L=L, stride=5, znorm=True)
        f = forward_return(logP, ends, H=H)
        dt = s.index.to_numpy()[ends]
        ok = np.isfinite(f) & np.isfinite(W).all(axis=1) & (dt <= cut.to_datetime64())
        Wtr.append(W[ok]); ftr.append(f[ok])
    Wtr, ftr = np.vstack(Wtr), np.concatenate(ftr)
    km = KMeans(n_clusters=K, n_init=3, random_state=0).fit(Wtr)
    edge = np.array([ftr[km.labels_ == c].mean() if (km.labels_ == c).any() else 0.0
                     for c in range(K)])

    # --- per-stock signal panels ---
    rev, mom, shp, vol, fwd = {}, {}, {}, {}, {}
    for tk, (s, r) in data.items():
        rev[tk] = (r.rolling(Nl).mean() - r.rolling(Ns).mean()) / (
            r.rolling(Ns).std() / np.sqrt(Ns))
        mom[tk] = r.rolling(MOM_L).sum().shift(MOM_SKIP)
        vol[tk] = r.rolling(VOLW).std()
        fwd[tk] = r.rolling(H).sum().shift(-H)
        logP = np.log(s.to_numpy())
        W, ends, _ = extract_windows(logP, L=L, stride=1, znorm=True)
        good = np.isfinite(W).all(axis=1)
        lab = np.full(len(W), -1); lab[good] = km.predict(W[good])
        sig = np.where(lab >= 0, edge[lab], np.nan)
        shp[tk] = pd.Series(sig, index=s.index[ends])
    REV, MOM, SHP, VOL, FWD = (pd.DataFrame(d) for d in (rev, mom, shp, vol, fwd))

    # align & cross-sectional standardise
    idx = FWD.index
    zrev, zmom, zshp = _rank_xs(REV.reindex(idx)), _rank_xs(MOM.reindex(idx)), _rank_xs(SHP.reindex(idx))
    combined = pd.concat([zrev, zmom, zshp]).groupby(level=0).mean()

    rebal = all_dates[(all_dates > cut)][::H]
    print("Signal independence (pooled corr on test):")
    te = idx[idx > cut]
    def flat(df): return df.reindex(te).to_numpy().ravel()
    def pcorr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1])
    fr, fm, fs = flat(zrev), flat(zmom), flat(zshp)
    print(f"  rev-mom {pcorr(fr,fm):+.2f}   rev-shape {pcorr(fr,fs):+.2f}   "
          f"mom-shape {pcorr(fm,fs):+.2f}\n")

    print("Standalone vs combined OOS cross-sectional rank IC (test):")
    for name, sig in [("reversion", zrev), ("momentum", zmom), ("shape", zshp),
                      ("COMBINED", combined)]:
        print(f"  {name:<10}: {_xs_ic(sig, FWD, rebal):+.4f}")
    print()

    curves = {}
    curves["reversion-only"] = _backtest("reversion-only", zrev, FWD, VOL, rebal)
    curves["combined"] = _backtest("combined", combined, FWD, VOL, rebal)
    curves["combined+threshold"] = _backtest("combined+threshold", combined, FWD, VOL,
                                             rebal, gate=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, (dts, net) in curves.items():
        ax.plot(dts, np.cumsum(net), label=name)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_title("Market-neutral selection: cumulative NET return (after 10bps costs)")
    ax.set_ylabel("cumulative net log return"); ax.set_xlabel("date"); ax.legend()
    fig.tight_layout(); fig.savefig("phase5_selection.png", dpi=110)
    print("\nsaved phase5_selection.png")


def _maxdd(c):
    return float((c - np.maximum.accumulate(c)).min())


def _backtest(name, SIG, FWD, VOL, rebal, gate=False):
    rets, turn, mkt, dts = [], [], [], []
    prev = pd.Series(dtype=float)
    for d in rebal:
        if d not in SIG.index or d not in FWD.index:
            continue
        s, y, v = SIG.loc[d], FWD.loc[d], VOL.loc[d]
        ok = s.notna() & y.notna() & v.notna() & (v > 0)
        s, y, v = s[ok], y[ok], v[ok]
        if len(s) < MIN_NAMES:
            continue
        mkt_val = float(y.mean())  # equal-weight market proxy
        if gate:
            zz = (s - s.mean()) / (s.std() + 1e-12)
            s = s[zz.abs() >= ZGATE]; y, v = y[s.index], v[s.index]
            if len(s) < MIN_NAMES:
                rets.append(0.0); turn.append(0.0); mkt.append(mkt_val); dts.append(d)
                prev = pd.Series(dtype=float); continue
        k = max(1, int(SEL * len(s)))
        rank = s.rank()
        longs = rank > len(s) - k
        shorts = rank <= k
        w = pd.Series(0.0, index=s.index)
        w[longs] = (1 / v[longs]); w[longs] /= w[longs].sum() * 2      # +0.5 gross
        w[shorts] = -(1 / v[shorts]); w[shorts] /= -w[shorts].sum() * 2  # -0.5 gross
        rets.append(float((w * y).sum()))
        al = prev.index.union(w.index)
        turn.append(float((w.reindex(al).fillna(0) - prev.reindex(al).fillna(0)).abs().sum()))
        mkt.append(mkt_val); dts.append(d); prev = w
    rets, turn, mkt = np.array(rets), np.array(turn), np.array(mkt)
    per = 252 / H
    def sh(x): return x.mean() / x.std() * np.sqrt(per) if x.std() > 0 else 0.0
    net = rets - COST * turn
    beta = np.polyfit(mkt, rets, 1)[0]
    print(f"[{name}]  n={len(rets)}  Sharpe gross {sh(rets):+.2f}  net {sh(net):+.2f}  "
          f"maxDD {_maxdd(np.cumsum(net)):+.2f}  turnover {turn.mean():.2f}  mkt-beta {beta:+.2f}")
    return pd.to_datetime(dts), net


if __name__ == "__main__":
    main()
