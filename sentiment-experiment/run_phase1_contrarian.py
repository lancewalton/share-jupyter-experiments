"""Phase 1: does aggregate fear predict forward equity returns contrarianly?

For each stress/fear gauge, measure the causal rank-IC with the forward K-week
equity return, the forward return by signal quintile (does the fearful extreme
lead higher returns?), and whether it ADDS over VIX. Contrarian hypothesis: high
stress -> positive forward return.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sent.data import SIGNALS, weekly_panel

K = 4   # forward horizon in weeks (~1 month)


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return np.nan
    ra, rb = np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def _partial(x, y, z):
    Z = np.c_[np.ones(len(z)), z]
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return _spearman(res(x), res(y))


def main():
    df = weekly_panel()
    r = df["ret"].to_numpy()
    n = len(df)
    fwd = np.full(n, np.nan)
    for t in range(n - K):
        fwd[t] = r[t + 1:t + 1 + K].sum()
    sub = np.arange(0, n - K, K)                          # non-overlapping
    vix = df["VIX"].to_numpy()

    print(f"weekly panel {n} weeks  {df.index[0].date()}..{df.index[-1].date()}  "
          f"(overlap of NFCI 1993+ / STLFSI 1993+)\n")
    print("contrarian: does high fear predict the forward 4-week return?")
    print(f"{'signal':8s} {'IC':>7s} {'IC|VIX':>8s}  {'bottom-Q fwd':>12s} {'top-Q fwd':>10s}")
    ics = {}
    for s in SIGNALS:
        sv = df[s].to_numpy()
        a, f = sv[sub], fwd[sub]
        ic = _spearman(a, f)
        icp = _partial(a[np.isfinite(f)], f[np.isfinite(f)], vix[sub][np.isfinite(f)]) if s != "VIX" else np.nan
        q = np.nanquantile(a, [0.2, 0.8])
        botf = np.nanmean(f[a <= q[0]]); topf = np.nanmean(f[a >= q[1]])
        ics[s] = (ic, botf, topf)
        icp_s = f"{icp:+.3f}" if s != "VIX" else "   —"
        print(f"{s:8s} {ic:+7.3f} {icp_s:>8s}  {botf*100:+11.1f}% {topf*100:+9.1f}%")

    print("\n  (top-Q = most fearful weeks; contrarian works if top-Q fwd > bottom-Q fwd)")
    _plot(df, fwd, sub, ics)
    print("saved phase1_contrarian.png")


def _plot(df, fwd, sub, ics):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    names = SIGNALS
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, [ics[s][1] * 100 for s in names], 0.4, label="calm (bottom 20%)", color="#4a8ca8")
    ax[0].bar(x + 0.2, [ics[s][2] * 100 for s in names], 0.4, label="fearful (top 20%)", color="#a8631a")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set_xticks(x); ax[0].set_xticklabels(names)
    ax[0].set_ylabel("mean forward 4-week return (%)")
    ax[0].set_title("Forward return after calm vs fearful weeks"); ax[0].legend(fontsize=8)
    # quintile monotonicity for VIX
    sv = df["VIX"].to_numpy()[sub]; f = fwd[sub]
    qs = np.nanquantile(sv, np.linspace(0, 1, 6))
    means = [np.nanmean(f[(sv >= qs[i]) & (sv <= qs[i + 1])]) * 100 for i in range(5)]
    ax[1].bar(range(1, 6), means, color="#0c757f")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("VIX quintile (1=calm, 5=fearful)")
    ax[1].set_ylabel("mean forward 4-week return (%)")
    ax[1].set_title("The contrarian gradient (VIX)")
    fig.tight_layout(); fig.savefig("phase1_contrarian.png", dpi=110)


if __name__ == "__main__":
    main()
