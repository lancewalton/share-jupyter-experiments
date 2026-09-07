"""Test B2: confidence-gated, vol-normalised cross-sectional reversion.

Ranking signal is the signed z-score of the divergence:
    z = (long_drift - short_drift) / SE(short_drift)
which is vol-normalised and confidence-weighted at once (high +z = significantly
below long-run drift -> strong long under reversion). We:
  * GATE to names with |z| >= ZGATE (only significant divergences);
  * size positions inverse-vol (FHS-envelope proxy), dollar-neutral, gross=1;
  * charge turnover costs; report Sharpe (gross & net), max drawdown, gated IC.
Compared against the naive quintile L/S of the raw signal (Test B baseline).
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
Ns, Nl, H, VOLW = 60, 1000, 20, 60
ZGATE = 1.0
MIN_ROWS = Nl + H + 60
GLITCH = 0.6
MIN_NAMES, MIN_GATED = 20, 12
COST = 0.001  # 10 bps per unit turnover (per side)


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


def _maxdd(cum):
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def main():
    sig, zc, vol, fwd = {}, {}, {}, {}
    for pattern in DIRS:
        for path in sorted(glob.glob(pattern)):
            r = _series(path)
            if r is None:
                continue
            tk = os.path.splitext(os.path.basename(path))[0] + \
                 (":uk" if "ukinvesting" in path else ":yf")
            sd = r.rolling(Ns).mean(); ld = r.rolling(Nl).mean()
            se = r.rolling(Ns).std() / np.sqrt(Ns)
            sig[tk] = ld - sd
            zc[tk] = (ld - sd) / se
            vol[tk] = r.rolling(VOLW).std()
            fwd[tk] = r.rolling(H).sum().shift(-H)
    SIG, Z, VOL, FWD = (pd.DataFrame(d) for d in (sig, zc, vol, fwd))
    print(f"universe: {SIG.shape[1]} series\n")

    dates = SIG.index.sort_values()
    rebal = dates[Nl::H]

    ic_raw, ic_gated = [], []
    ret_naive, ret_g_gross, ret_g_net, turn = [], [], [], []
    curve_dates, prev_w = [], pd.Series(dtype=float)

    for d in rebal:
        if d not in SIG.index or d not in FWD.index:
            continue
        s = SIG.loc[d]; z = Z.loc[d]; v = VOL.loc[d]; y = FWD.loc[d]
        ok = s.notna() & y.notna() & v.notna() & (v > 0) & z.notna()
        if ok.sum() < MIN_NAMES:
            continue
        s, z, v, y = s[ok], z[ok], v[ok], y[ok]
        ic_raw.append(_spearman(s.to_numpy(), y.to_numpy()))

        # naive quintile L/S on raw signal (Test-B baseline)
        k = max(1, len(s) // 5)
        order = np.argsort(s.to_numpy())
        yv = y.to_numpy()
        ret_naive.append(yv[order[-k:]].mean() - yv[order[:k]].mean())

        # gated + inverse-vol, dollar-neutral, gross=1
        g = z.abs() >= ZGATE
        if g.sum() < MIN_GATED:
            ret_g_gross.append(0.0); ret_g_net.append(0.0); turn.append(0.0)
            curve_dates.append(d)
            ic_gated.append(np.nan)
            prev_w = pd.Series(dtype=float)
            continue
        zg, vg, yg = z[g], v[g], y[g]
        ic_gated.append(_spearman(zg.to_numpy(), yg.to_numpy()))
        w = zg / vg
        w = w - w.mean()                         # dollar-neutral
        w = w / np.abs(w).sum()                  # gross exposure = 1
        gross = float((w * yg).sum())
        aligned = w.reindex(prev_w.index.union(w.index)).fillna(0.0) - \
                  prev_w.reindex(prev_w.index.union(w.index)).fillna(0.0)
        t = float(np.abs(aligned).sum())
        ret_g_gross.append(gross); ret_g_net.append(gross - COST * t); turn.append(t)
        curve_dates.append(d); prev_w = w

    def stats(x):
        x = np.array(x); per = 252 / H
        shp = x.mean() / x.std() * np.sqrt(per) if x.std() > 0 else 0.0
        return x.mean(), shp, _maxdd(np.cumsum(x)), np.mean(x > 0)

    print(f"rebalances scored: {len(ret_naive)}   gate |z|>={ZGATE}  "
          f"avg turnover/period: {np.mean(turn):.2f}\n")
    print(f"{'strategy':<26}{'IC':>8}{'Sharpe':>8}{'maxDD':>9}{'%>0':>6}")
    m, shp, dd, w0 = stats(ret_naive)
    print(f"{'naive quintile (raw)':<26}{np.nanmean(ic_raw):>8.3f}{shp:>8.2f}{dd:>9.3f}{w0*100:>5.0f}%")
    m, shp, dd, w0 = stats(ret_g_gross)
    print(f"{'gated+volnorm (gross)':<26}{np.nanmean(ic_gated):>8.3f}{shp:>8.2f}{dd:>9.3f}{w0*100:>5.0f}%")
    m, shp, dd, w0 = stats(ret_g_net)
    print(f"{'gated+volnorm (net 10bps)':<26}{'':>8}{shp:>8.2f}{dd:>9.3f}{w0*100:>5.0f}%")
    print(f"\n  ungated rank IC {np.nanmean(ic_raw):+.3f}  vs  gated rank IC "
          f"{np.nanmean(ic_gated):+.3f}   (Test-A predicted gating lifts IC)")

    _plot(curve_dates, ret_naive, ret_g_gross, ret_g_net)
    print("\nsaved xsec_gated.png")


def _plot(dates, naive, gg, gn):
    fig, ax = plt.subplots(figsize=(11, 5))
    dts = pd.to_datetime(dates)
    ax.plot(dts, np.cumsum(naive[:len(dts)]), label="naive quintile (raw signal)")
    ax.plot(dts, np.cumsum(gg), label="gated + vol-norm (gross)")
    ax.plot(dts, np.cumsum(gn), label="gated + vol-norm (net 10bps)")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_ylabel("cumulative L/S log return"); ax.set_xlabel("date")
    ax.set_title(f"Cross-sectional reversion: naive vs gated+vol-norm "
                 f"(Ns={Ns},Nl={Nl},H={H})")
    ax.legend()
    fig.tight_layout(); fig.savefig("xsec_gated.png", dpi=110)


if __name__ == "__main__":
    main()
