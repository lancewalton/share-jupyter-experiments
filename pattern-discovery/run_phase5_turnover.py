"""Phase 5 turnover reduction: can we lift net Sharpe by trading less?

Net Sharpe (0.36) sits far below gross (0.64) purely because of turnover (~1.4).
Test standard reducers against the baseline top/bottom-20% re-selection:
  - SMOOTH   : EWMA the combined signal over time before ranking
  - PARTIAL  : move weights only fraction alpha toward target each period
  - BUFFER   : rank hysteresis -- enter top/bottom 20%, hold until leaving 35%
  - combos
For each: turnover, gross Sharpe, net Sharpe @10bps, and break-even cost (bps).
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
    SIG = combined.reindex(rebal); FW = FWD.reindex(rebal); VO = VOL.reindex(rebal)
    return SIG, FW, VO


def _target_weights(s, v, k, held, enter_frac, exit_frac):
    """Inverse-vol dollar-neutral target. With buffer, held names survive until
    they leave the exit band; otherwise plain top/bottom-k selection."""
    n = len(s)
    rank = s.rank()  # 1=lowest .. n=highest
    top_enter = rank > n * (1 - enter_frac)
    bot_enter = rank <= n * enter_frac
    if held is None:
        longs, shorts = rank > n - k, rank <= k
    else:
        hl, hs = held
        keep_long = s.index.isin(hl) & (rank > n * (1 - exit_frac))
        keep_short = s.index.isin(hs) & (rank <= n * exit_frac)
        longs = keep_long | top_enter
        shorts = keep_short | bot_enter
    w = pd.Series(0.0, index=s.index)
    if longs.any():
        w[longs] = (1 / v[longs]); w[longs] /= w[longs].sum() * 2
    if shorts.any():
        w[shorts] = -(1 / v[shorts]); w[shorts] /= -w[shorts].sum() * 2
    return w, (set(s.index[longs]), set(s.index[shorts]))


def _run(SIG, FW, VO, smooth_hl=None, alpha=1.0, buffer=False):
    sig = SIG.ewm(halflife=smooth_hl).mean() if smooth_hl else SIG
    gross, turn = [], []
    prev = pd.Series(dtype=float); held = None
    enter_f, exit_f = SEL, (0.35 if buffer else SEL)
    for d in sig.index:
        s, y, v = sig.loc[d], FW.loc[d], VO.loc[d]
        ok = s.notna() & y.notna() & v.notna() & (v > 0)
        s, y, v = s[ok], y[ok], v[ok]
        if len(s) < MIN_NAMES:
            continue
        k = max(1, int(SEL * len(s)))
        tgt, held = _target_weights(s, v, k, held if buffer else None, enter_f, exit_f)
        w = tgt if alpha >= 1 else (alpha * tgt.add((1 - alpha) * prev, fill_value=0))
        # renormalise gross to 1 after partial blending
        gr = np.abs(w).sum()
        if gr > 0:
            w = w / gr
        al = prev.index.union(w.index)
        turn.append(float((w.reindex(al).fillna(0) - prev.reindex(al).fillna(0)).abs().sum()))
        gross.append(float((w * y.reindex(w.index).fillna(0)).sum()))
        prev = w
    return np.array(gross), np.array(turn)


def _stats(gross, turn):
    per = 252 / H
    def sh(x): return x.mean() / x.std() * np.sqrt(per) if x.std() > 0 else 0.0
    net10 = gross - 0.001 * turn
    be = 1e4 * gross.mean() / turn.mean() if turn.mean() > 0 else np.inf
    return turn.mean(), sh(gross), sh(net10), be


def main():
    SIG, FW, VO = _build()
    variants = {
        "baseline (reselect 20%)":      dict(),
        "smooth hl=3":                  dict(smooth_hl=3),
        "partial alpha=0.4":            dict(alpha=0.4),
        "buffer 20/35%":                dict(buffer=True),
        "smooth3 + partial0.5":         dict(smooth_hl=3, alpha=0.5),
        "smooth3 + buffer":             dict(smooth_hl=3, buffer=True),
    }
    print(f"{'variant':<26}{'turnover':>9}{'gross':>7}{'net@10':>8}{'break-even bps':>15}")
    rows = {}
    for name, kw in variants.items():
        g, t = _run(SIG, FW, VO, **kw)
        tm, gs, ns, be = _stats(g, t)
        rows[name] = (tm, gs, ns, be)
        print(f"{name:<26}{tm:>9.2f}{gs:>+7.2f}{ns:>+8.2f}{be:>15.1f}")

    names = list(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.bar(range(len(names)), [rows[n][2] for n in names], color="tab:blue")
    a1.set_xticks(range(len(names))); a1.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    a1.set_ylabel("net Sharpe @10bps"); a1.set_title("Net Sharpe by turnover-reduction variant")
    a1.axhline(rows[names[0]][2], color="k", lw=0.8, ls="--", label="baseline")
    a1.legend(fontsize=8)
    a2.scatter([rows[n][0] for n in names], [rows[n][3] for n in names])
    for n in names:
        a2.annotate(n, (rows[n][0], rows[n][3]), fontsize=6)
    a2.set_xlabel("turnover / period"); a2.set_ylabel("break-even cost (bps)")
    a2.set_title("Lower turnover -> higher cost tolerance")
    fig.tight_layout(); fig.savefig("phase5_turnover.png", dpi=110)
    print("\nsaved phase5_turnover.png")


if __name__ == "__main__":
    main()
