"""Phase 2: the dominant FX factor — carry — vs value and momentum.

Carry: long high-interest-rate currencies, short low. It earns the rate
differential and historically pays well, with violent crash risk in risk-off.
On total returns (spot + carry) we compare three dollar-neutral factors: carry
(rank by rate differential), value (long the cheap by 3y), momentum (long the
strong by 1y). Which one actually pays in currencies?
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fx.data import build_total_returns


def _ls(score, S, t, N):
    """Dollar-neutral weights: long top-N score, short bottom-N."""
    o = np.argsort(score); w = np.zeros(S.shape[1])
    w[o[-N:]] = 1.0 / N; w[o[:N]] = -1.0 / N
    return w


def _stats(r):
    m = np.isfinite(r); x = r[m]
    eq = np.cumsum(x); dd = (eq - np.maximum.accumulate(eq)).min()
    return dict(sharpe=x.mean() / x.std() * np.sqrt(12), ann=x.mean() * 12,
                worst=x.min(), mdd=dd)


def main():
    total, diff = build_total_returns()
    T = total.to_numpy(); D = diff.to_numpy(); dates = total.index; n, p = T.shape
    print(f"{p} currencies · {n} months · {dates[0].date()}..{dates[-1].date()}\n  "
          + ", ".join(total.columns) + "\n")

    N = 4
    def factor(kind, L=None):
        ret = np.full(n, np.nan)
        for t in range(n):
            if kind == "carry":
                sc = D[t - 1] if t > 0 else None                 # yesterday's differential
            elif t >= L:
                sc = np.array([T[t - L:t, s].sum() for s in range(p)])
                sc = sc if kind == "momentum" else -sc           # value = long the cheap
            else:
                sc = None
            if sc is not None and np.isfinite(sc).all() and t > 0:
                ret[t] = _ls(sc, T, t, N) @ T[t]
        return ret

    books = {"carry (high rate)": factor("carry"),
             "momentum (1y)": factor("momentum", 12),
             "value (3y)": factor("value", 36)}
    print(f"{'factor':20s} {'ann.ret':>8s} {'Sharpe':>7s} {'worst mo':>9s} {'maxDD':>7s}")
    for name, r in books.items():
        s = _stats(r)
        print(f"{name:20s} {s['ann']*100:+7.1f}% {s['sharpe']:+7.2f} "
              f"{s['worst']*100:+8.1f}% {s['mdd']*100:+6.0f}%")

    # worst months for carry — the crash-risk signature
    cr = books["carry (high rate)"]; ok = np.isfinite(cr)
    order = np.argsort(cr[ok])[:4]; d_ok = dates[ok]
    print("\ncarry's worst months (the crash risk):  " +
          ", ".join(f"{str(d_ok[i])[:7]} ({cr[ok][i]*100:+.1f}%)" for i in order))

    _plot(dates, books)
    print("\nsaved phase2_carry.png")


def _plot(dates, books):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    cols = {"carry (high rate)": "#0c757f", "momentum (1y)": "#a8631a", "value (3y)": "#4a8ca8"}
    for name, r in books.items():
        s = r.copy(); s[~np.isfinite(s)] = 0
        ax.plot(dates, np.cumsum(s), lw=1.7, color=cols[name],
                label=f"{name}  (Sharpe {np.nanmean(r)/np.nanstd(r)*np.sqrt(12):+.2f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("cumulative factor return (total, dollar-neutral)")
    ax.set_title("FX factors — carry vs value vs momentum (2002–2026)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig("phase2_carry.png", dpi=110)


if __name__ == "__main__":
    main()
