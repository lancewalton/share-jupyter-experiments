"""Phase 1: do currencies REVERT (contrarian pays) or TREND (momentum pays)?

FX is the ultimate valuation-bounded asset — anchored to purchasing-power parity —
so the programme's taxonomy predicts it should REVERT (contrarian works), unlike
equity regions which trend. We test it: cross-sectional reversion IC, and the
Sharpe of a dollar-neutral contrarian (buy the cheapest) vs momentum (buy the
strongest) long/short currency factor, by formation horizon.

Spot only — no carry (the dominant FX premium), so absolute returns are small;
the SIGN of the signal (does the beaten-down currency bounce?) is the point.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fx.data import build_fx_returns


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    ra, rb = np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    R = build_fx_returns()
    S = R.to_numpy(); dates = R.index; n, p = S.shape
    print(f"{p} currencies · {n} months · {dates[0].date()}..{dates[-1].date()}\n  "
          + ", ".join(R.columns) + "\n")

    print("cross-sectional reversion — rank-IC(trailing, forward); <0 = contrarian/value:")
    print(f"{'':9s}" + "".join(f"  fwd {h}y" for h in (1, 3, 5)))
    for L in (1, 3, 5):
        row = f"trail {L}y"
        for H in (1, 3, 5):
            ics = []
            for t in range(12 * L, n - 12 * H, 12 * H):
                tr = np.array([S[t - 12 * L:t, s].sum() for s in range(p)])
                fw = np.array([S[t:t + 12 * H, s].sum() for s in range(p)])
                ics.append(_spearman(tr, fw))
            row += f"  {np.mean(ics):+7.2f}"
        print(f"  {row}")

    # dollar-neutral long/short factor: long bottom-N, short top-N by trailing return
    RB, N = 12, 4
    def run(sign, L_form):
        w = np.zeros(p); ret = np.full(n, np.nan)
        for t in range(n):
            if t >= L_form and (t - L_form) % RB == 0:
                trail = np.array([S[t - L_form:t, s].sum() for s in range(p)])
                o = np.argsort(trail); w = np.zeros(p)
                w[o[:N]] = sign / N; w[o[-N:]] = -sign / N        # long cheapest / short dearest
            if t > 0:
                ret[t] = w @ S[t]
        return ret

    def sh(r):
        m = np.isfinite(r); return r[m].mean() / r[m].std() * np.sqrt(12)
    print("\ndollar-neutral factor Sharpe by formation (long/short 4 of 16):")
    print(f"{'formation':11s} {'contrarian/value':>16s} {'momentum':>10s}")
    series = {}
    for L in (12, 36, 60):
        c, m = run(+1, L), run(-1, L)
        if L == 36:
            series = {"contrarian / value (3y)": c, "momentum (3y)": m}
        print(f"  {L//12}y{'':8s} {sh(c):+16.2f} {sh(m):+10.2f}")

    _plot(dates, series)
    print("\nsaved phase1_fx.png")


def _plot(dates, series):
    fig, ax = plt.subplots(figsize=(11, 5))
    cols = {"contrarian / value (3y)": "#0c757f", "momentum (3y)": "#a8631a"}
    for name, r in series.items():
        ax.plot(dates, np.nancumsum(np.where(np.isfinite(r), r, 0.0)), lw=1.7,
                label=f"{name}  (Sharpe {r[np.isfinite(r)].mean()/r[np.isfinite(r)].std()*np.sqrt(12):+.2f})",
                color=cols[name])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("cumulative factor return (log, spot only)")
    ax.set_title("FX contrarian/value vs momentum — dollar-neutral (1999–2026)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig("phase1_fx.png", dpi=110)


if __name__ == "__main__":
    main()
