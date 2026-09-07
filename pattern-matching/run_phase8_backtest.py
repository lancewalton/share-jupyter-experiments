"""Phase 8: back-test the reversion-after-large-moves one-liner.

Stage 3 found the usable signal is amplitude-conditioned reversion: after a large
recent move, price tends to revert. The natural embodiment is a market-neutral
cross-sectional book -- each month, long the biggest losers and short the biggest
winners over an L-day formation (the tail quantiles ARE the large movers),
inverse-vol sized, net of costs.

No parameters are fitted, so the whole history is a fair out-of-sample test. We
report the sibling projects' honest battery: gross/net Sharpe, drawdown, market
beta, turnover, a momentum sign-check (must lose if reversion is real), cost
break-even, sub-period stability, and an L x frac sensitivity grid.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.data import load_close
from mc.returns import log_returns
from matching.xsec import ann_return, beta, max_drawdown, sharpe, tail_weights

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
MIN_ROWS, GLITCH = 1500, 0.6
L, REBAL, FRAC, VOL_LB, COST = 80, 21, 0.2, 60, 0.0010   # 10 bps per unit turnover


def _panel():
    cols = {}
    for pattern in DIRS:
        for path in sorted(glob.glob(pattern)):
            try:
                s = load_close(path)
            except Exception:
                continue
            r = log_returns(s.to_numpy())
            if len(r) < MIN_ROWS or np.abs(r).max() > GLITCH:
                continue
            cols[path.rsplit("/", 1)[-1].rsplit(".", 1)[0]] = s
    P = pd.DataFrame(cols).sort_index()
    return P


def _signal(logP, dret, ti, L):
    with np.errstate(all="ignore"):
        move = logP[ti] - logP[ti - L]
        vol = np.nanstd(dret[ti - VOL_LB:ti], axis=0, ddof=1)
        drift = np.where(vol > 0, move / (vol * np.sqrt(L)), np.nan)
    return drift, vol


def backtest(logP, dret, L, frac, rebal, cost, sgn):
    """Cross-sectional: sgn=+1 reversion (long losers), sgn=-1 momentum."""
    N, S = logP.shape
    daily = np.full(N, np.nan); tos = []; w_prev = np.zeros(S)
    for ti in range(max(L, VOL_LB) + 1, N - 1, rebal):
        drift, vol = _signal(logP, dret, ti, L)
        w = tail_weights(sgn * (-drift), vol, frac)
        to = np.abs(w - w_prev).sum(); tos.append(to)
        for d in range(ti + 1, min(ti + rebal, N - 1) + 1):
            daily[d] = np.nansum(w * dret[d])
        daily[ti + 1] -= to * cost
        w_prev = w
    return daily, float(np.mean(tos))


def backtest_ts(logP, dret, L, rebal, cost, hedge_mkt):
    """Time-series reversion: each stock weighted by -own drift (inv-vol), unit
    gross. Optionally hedge out the market so it isn't a disguised beta bet."""
    N, S = logP.shape
    daily = np.full(N, np.nan); tos = []; w_prev = np.zeros(S)
    mkt = np.nanmean(dret, axis=1)
    for ti in range(max(L, VOL_LB) + 1, N - 1, rebal):
        drift, vol = _signal(logP, dret, ti, L)
        with np.errstate(all="ignore"):
            raw = np.where(np.isfinite(drift) & (vol > 0), -drift / vol, 0.0)
        g = np.abs(raw).sum()
        w = raw / g if g > 0 else raw
        to = np.abs(w - w_prev).sum(); tos.append(to)
        for d in range(ti + 1, min(ti + rebal, N - 1) + 1):
            r = np.nansum(w * dret[d])
            daily[d] = r - (w.sum() * mkt[d] if hedge_mkt else 0.0)
        daily[ti + 1] -= to * cost
        w_prev = w
    return daily, float(np.mean(tos))


def _stats(daily, mkt):
    ok = np.isfinite(daily)
    d, m = daily[ok], mkt[ok]
    return dict(sharpe=sharpe(d), ann=ann_return(d), vol=d.std(ddof=1) * np.sqrt(252),
                mdd=max_drawdown(d), beta=beta(d, m), n=len(d))


def main():
    P = _panel()
    logP = np.log(P.to_numpy())
    dret = np.vstack([np.full((1, logP.shape[1]), np.nan), np.diff(logP, axis=0)])
    mkt = np.nanmean(dret, axis=1)
    print(f"panel: {P.shape[1]} stocks x {P.shape[0]} days "
          f"({P.index[0].date()}..{P.index[-1].date()})\n")

    rev, to_rev = backtest(logP, dret, L, FRAC, REBAL, COST, sgn=+1)
    gross, _ = backtest(logP, dret, L, FRAC, REBAL, 0.0, sgn=+1)
    mom, _ = backtest(logP, dret, L, FRAC, REBAL, COST, sgn=-1)
    s_net, s_gross, s_mom = _stats(rev, mkt), _stats(gross, mkt), _stats(mom, mkt)

    print("reversion-after-large-moves (long losers / short winners, monthly, inv-vol):")
    print(f"  gross Sharpe {s_gross['sharpe']:+.2f}   net Sharpe {s_net['sharpe']:+.2f}"
          f"   ann {s_net['ann']*100:+.1f}%  vol {s_net['vol']*100:.1f}%")
    print(f"  maxDD {s_net['mdd']*100:.0f}%   market-beta {s_net['beta']:+.2f}"
          f"   avg turnover {to_rev:.2f}/rebal")
    print(f"  momentum sign-check (should LOSE): net Sharpe {s_mom['sharpe']:+.2f}")

    # time-series embodiment (the pooled/time-series IC's proper form)
    ts_raw, to_ts = backtest_ts(logP, dret, L, REBAL, COST, hedge_mkt=False)
    ts_hed, _ = backtest_ts(logP, dret, L, REBAL, COST, hedge_mkt=True)
    sr, sh_ = _stats(ts_raw, mkt), _stats(ts_hed, mkt)
    print("\ntime-series reversion (per-stock buy-the-dip, unit gross):")
    print(f"  directional net Sharpe {sr['sharpe']:+.2f} (beta {sr['beta']:+.2f}, "
          f"maxDD {sr['mdd']*100:.0f}%)   market-hedged net Sharpe {sh_['sharpe']:+.2f}")

    # cost break-even
    print("\n  cost sensitivity (net Sharpe):")
    costs = [0.0, 0.0005, 0.0010, 0.0020, 0.0030]
    sh = []
    for c in costs:
        dd, _ = backtest(logP, dret, L, FRAC, REBAL, c, sgn=+1)
        sh.append(_stats(dd, mkt)["sharpe"])
        print(f"    {int(c*1e4):2d} bps: {sh[-1]:+.2f}")
    be = np.interp(0, sh[::-1], [c * 1e4 for c in costs][::-1]) if sh[0] > 0 > sh[-1] else np.nan
    print(f"    break-even ~ {be:.0f} bps" if np.isfinite(be) else "    (no zero crossing in range)")

    # sub-periods (thirds)
    ok = np.where(np.isfinite(rev))[0]
    print("\n  sub-period net Sharpe (thirds):", end=" ")
    for part in np.array_split(ok, 3):
        print(f"{sharpe(rev[part]):+.2f}", end="  ")
    print()

    # L x frac sensitivity
    print("\n  sensitivity net Sharpe   (rows L, cols frac):")
    fracs = [0.1, 0.2, 0.3]; Ls = [40, 80, 120]
    print("        " + "  ".join(f"f={f}" for f in fracs))
    grid = []
    for Lx in Ls:
        row = []
        for fx in fracs:
            dd, _ = backtest(logP, dret, Lx, fx, REBAL, COST, sgn=+1)
            row.append(_stats(dd, mkt)["sharpe"])
        grid.append(row)
        print(f"  L={Lx:3d}  " + "  ".join(f"{v:+.2f}" for v in row))

    _plot(rev, gross, mom, mkt, ok, costs, sh, grid, Ls, fracs)
    print("\nsaved phase8_backtest.png")


def _plot(rev, gross, mom, mkt, ok, costs, sh, grid, Ls, fracs):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for series, lab in [(gross, "reversion gross"), (rev, "reversion net"),
                        (mom, "momentum net"), (mkt, "market (EW)")]:
        eq = np.exp(np.nancumsum(np.where(np.isfinite(series), series, 0.0)[ok]))
        ax[0].plot(eq, label=lab, lw=1.4)
    ax[0].set_yscale("log"); ax[0].set_title("Equity curves (log)"); ax[0].legend(fontsize=8)
    ax[0].set_xlabel("trading day (test span)")
    ax[1].plot([c * 1e4 for c in costs], sh, "o-")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_xlabel("cost (bps/turnover)")
    ax[1].set_ylabel("net Sharpe"); ax[1].set_title("Cost break-even")
    im = ax[2].imshow(grid, cmap="RdYlGn", vmin=-0.5, vmax=0.5, aspect="auto")
    ax[2].set_xticks(range(len(fracs))); ax[2].set_xticklabels([f"f={f}" for f in fracs])
    ax[2].set_yticks(range(len(Ls))); ax[2].set_yticklabels([f"L={L}" for L in Ls])
    for i in range(len(Ls)):
        for j in range(len(fracs)):
            ax[2].text(j, i, f"{grid[i][j]:+.2f}", ha="center", va="center", fontsize=9)
    ax[2].set_title("Net Sharpe sensitivity"); fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.tight_layout(); fig.savefig("phase8_backtest.png", dpi=110)


if __name__ == "__main__":
    main()
