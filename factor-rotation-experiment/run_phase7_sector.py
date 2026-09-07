"""Phase 7: contrarian within-equity SECTOR rotation vs buy-and-hold market.

The cross-asset contrarian (Phase 6) only matched equity because it had to
underweight the era's secular winner. This version stays fully in equities and
rotates between out-of-favour SECTORS -- removing that drag. Ken French 12
industry portfolios (value-weighted total returns), the market for the benchmark.

  1. reversion: cross-sectional rank-IC(trailing L-year sector return, forward
     H-year) -- <0 means the beaten-down sector wins next; contrarian premise holds.
  2. backtest: fully-invested contrarian sector book (overweight worst trailing)
     vs buy-hold market, equal-weight sectors, and momentum. Excess-return Sharpe,
     net of turnover.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pca.french import load_ff_market, load_industry_vw

START = "1945-07"
COST = 0.0005   # 5 bps per unit turnover at each annual rebalance


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    ra, rb = np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ind = load_industry_vw().loc[START:]
    ff = load_ff_market().reindex(ind.index)
    S = ind.to_numpy(); mkt = ff["Mkt"].to_numpy(); rf = ff["RF"].to_numpy()
    dates = ind.index; n, p = S.shape
    print(f"{p} sectors · {n} months · {dates[0].date()}..{dates[-1].date()}\n")

    # --- 1. cross-sectional long-horizon reversion ---
    def cum(a, s, w):
        return np.array([np.expm1(np.log1p(a[t - w:t, s]).sum()) if t - w >= 0 else np.nan
                         for t in range(n)])
    print("cross-sectional reversion — rank-IC(trailing, forward); <0 = contrarian wins:")
    print(f"{'':9s}" + "".join(f"  fwd {h}y" for h in (1, 3, 5)))
    for L in (1, 3, 5):
        row = f"trail {L}y"
        for H in (1, 3, 5):
            ics = []
            for t in range(12 * L, n - 12 * H, 12 * H):
                tr = np.array([np.log1p(S[t - 12 * L:t, s]).sum() for s in range(p)])
                fw = np.array([np.log1p(S[t:t + 12 * H, s]).sum() for s in range(p)])
                ics.append(_spearman(tr, fw))
            row += f"  {np.mean(ics):+7.2f}"
        print(f"  {row}")

    # --- 2. backtest: CONCENTRATED tilt (hold only worst-N / best-N sectors) ---
    RB, N = 12, 4
    def run(pick, L_form):
        w = np.ones(p) / p; ret = np.full(n, np.nan); to = 0.0
        for t in range(n):
            if t >= L_form and (t - L_form) % RB == 0:
                trail = np.array([np.log1p(S[t - L_form:t, s]).sum() for s in range(p)])
                nw = np.zeros(p); nw[pick(trail)] = 1.0 / N
                to = np.abs(nw - w).sum(); w = nw
            if t > 0:
                ret[t] = w @ S[t] - (COST * to if t >= L_form and (t - L_form) % RB == 0 else 0)
        return ret
    worst = lambda tr: np.argsort(tr)[:N]
    best = lambda tr: np.argsort(tr)[-N:]

    def sh(r):
        m = np.isfinite(r); ex = (r - rf)[m]; return ex.mean() / ex.std() * np.sqrt(12)
    print("\nconcentrated backtest — Sharpe by formation length (hold only 4 of 12):")
    print(f"{'formation':11s} {'contrarian':>11s} {'momentum':>10s} {'equal-wt':>9s}")
    forms = [36, 60, 120]
    eqw_r = np.array([np.ones(p).dot(S[t]) / p if t > 0 else np.nan for t in range(n)])
    for L in forms:
        c, m = run(worst, L), run(best, L)
        print(f"  {L//12}y{'':8s} {sh(c):+11.2f} {sh(m):+10.2f} {sh(eqw_r):+9.2f}")
    print(f"  buy-hold market Sharpe {sh(mkt):+.2f}  (equal-weight {sh(eqw_r):+.2f})")

    books = {"contrarian (worst 4, 5y)": run(worst, 60),
             "momentum (best 4, 5y)": run(best, 60),
             "equal-weight sectors": eqw_r, "buy-hold market": mkt.copy()}
    print("\ndetail (5y formation):")
    print(f"{'strategy':27s} {'ann.ret':>8s} {'vol':>6s} {'Sharpe':>7s} {'end $1':>8s}")
    for name, r in books.items():
        m = np.isfinite(r)
        print(f"{name:27s} {r[m].mean()*12*100:+7.1f}% {r[m].std()*np.sqrt(12)*100:5.1f}% "
              f"{sh(r):+7.2f} {1+np.expm1(np.log1p(r[m]).sum()):8.0f}x")

    _plot(dates, books)
    print("\nsaved phase7_sector.png")


def _plot(dates, books):
    fig, ax = plt.subplots(figsize=(11, 5.4))
    cols = {"contrarian (worst 4, 5y)": "#0c757f", "momentum (best 4, 5y)": "#a8631a",
            "equal-weight sectors": "#4a8ca8", "buy-hold market": "#14181b"}
    for name, r in books.items():
        eq = np.exp(np.nancumsum(np.log1p(np.where(np.isfinite(r), r, 0.0))))
        ax.plot(dates, eq, lw=1.7, label=name, color=cols[name],
                ls="--" if name == "buy-hold market" else "-")
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log, total return)")
    ax.set_title("Contrarian sector rotation vs buy-and-hold market (1945–2026)")
    ax.legend(fontsize=9); ax.axhline(1, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig("phase7_sector.png", dpi=110)


if __name__ == "__main__":
    main()
