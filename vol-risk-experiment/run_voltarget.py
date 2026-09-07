"""Part 1: volatility targeting -- the first real use of a calibrated vol forecast.

No directional view. Each day size the position inversely to the FORECAST
volatility so risk stays ~constant: w_t = target_vol / forecast_vol_t (causal,
capped). Compare to buy-and-hold on return, volatility, Sharpe and max drawdown.
The point: even with zero directional skill, sizing by a vol forecast raises
risk-adjusted return and cuts drawdowns -- and it is leverage-scalable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.data import load_close
from mc.returns import log_returns
from mc.forecasters import _ewma_vol_series

ANN = 252
TARGET_VOL = 0.15          # annualised risk target
MAX_LEV = 3.0
COST = 0.0001              # 10 bps per unit turnover


def _stats(daily):
    g = np.cumsum(daily)                       # cumulative log return
    ann_ret = daily.mean() * ANN
    ann_vol = daily.std() * np.sqrt(ANN)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    dd = float((g - np.maximum.accumulate(g)).min())
    return ann_ret, ann_vol, sharpe, dd, g


def voltarget(r, lam=0.94, burn=60):
    """Return (weights, gross daily P&L, net daily P&L) for the vol-managed book."""
    sig1 = np.sqrt(_ewma_vol_series(r, lam=lam, burn=burn))  # _series returns VARIANCE
    sig_ann = sig1 * np.sqrt(ANN)                            # annualised 1-day vol forecast
    w = np.clip(TARGET_VOL / np.where(sig_ann > 0, sig_ann, np.nan), 0, MAX_LEV)
    w = np.nan_to_num(w)
    gross = w * r                                       # position held over day t earns r[t]
    turn = np.abs(np.diff(w, prepend=w[0]))
    net = gross - COST * turn
    return w, gross, net, turn


def main():
    r = log_returns(load_close().to_numpy())            # FTSE
    r = r[np.isfinite(r)]
    w, gross, net, turn = voltarget(r)

    print(f"FTSE daily returns: {len(r)}   target vol {TARGET_VOL:.0%}  max lev {MAX_LEV}")
    print(f"avg leverage {w.mean():.2f}   avg daily turnover {turn.mean():.3f}\n")
    print(f"{'strategy':<24}{'ann.ret':>9}{'ann.vol':>9}{'Sharpe':>8}{'maxDD':>8}")
    for name, d in [("buy & hold", r), ("vol-managed (gross)", gross),
                    ("vol-managed (net 10bps)", net)]:
        ar, av, sh, dd, _ = _stats(d)
        print(f"{name:<24}{ar:>+9.3f}{av:>9.3f}{sh:>+8.2f}{dd:>+8.2f}")

    # equal-risk comparison: scale vol-managed to buy&hold's realised vol
    bh_vol = r.std()
    k = bh_vol / net.std()
    ar_s, av_s, sh_s, dd_s, _ = _stats(net * k)
    ar_b, av_b, sh_b, dd_b, _ = _stats(r)
    print(f"\nAt MATCHED risk (vol-managed levered to buy&hold vol {av_b:.1%}):")
    print(f"  buy & hold   : ann.ret {ar_b:+.3f}  maxDD {dd_b:+.2f}")
    print(f"  vol-managed  : ann.ret {ar_s:+.3f}  maxDD {dd_s:+.2f}   "
          f"(same risk, {'MORE' if ar_s>ar_b else 'less'} return, "
          f"{'shallower' if dd_s>dd_b else 'deeper'} drawdown)")

    # sub-period Sharpes (thirds)
    print("\nSharpe by sub-period (buy&hold vs vol-managed net):")
    for lo, hi in [(0, len(r)//3), (len(r)//3, 2*len(r)//3), (2*len(r)//3, len(r))]:
        _, _, shb, _, _ = _stats(r[lo:hi]); _, _, shv, _, _ = _stats(net[lo:hi])
        print(f"  segment {lo:>5}-{hi:<5}: B&H {shb:+.2f}   vol-managed {shv:+.2f}")

    _xsec()
    _plot(r, gross, net, w)
    print("\nsaved voltarget.png")


def _xsec():
    import glob
    dirs = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
            "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
    d_bh, d_vm = [], []
    for pat in dirs:
        for path in sorted(glob.glob(pat)):
            try:
                rr = log_returns(load_close(path).to_numpy())
            except Exception:
                continue
            if len(rr) < 1500 or np.abs(rr).max() > 0.6:
                continue
            _, _, net, _ = voltarget(rr)
            _, _, sh_b, _, _ = _stats(rr)
            _, _, sh_v, _, _ = _stats(net)
            d_bh.append(sh_b); d_vm.append(sh_v)
    d_bh, d_vm = np.array(d_bh), np.array(d_vm)
    print(f"\nCross-section ({len(d_bh)} equities) Sharpe:")
    print(f"  mean buy&hold {d_bh.mean():+.2f}   mean vol-managed {d_vm.mean():+.2f}   "
          f"vol-managed wins {np.mean(d_vm > d_bh):.0%}")


def _plot(r, gross, net, w):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(np.cumsum(r), label="buy & hold", color="tab:gray")
    a1.plot(np.cumsum(gross), label="vol-managed (gross)", color="tab:blue")
    a1.plot(np.cumsum(net), label="vol-managed (net 10bps)", color="tab:orange")
    a1.set_ylabel("cumulative log return"); a1.legend()
    a1.set_title("Volatility targeting vs buy & hold (FTSE)")
    a2.plot(w, color="tab:green", lw=0.5); a2.axhline(1, color="k", lw=0.6, ls=":")
    a2.set_ylabel("leverage w_t"); a2.set_xlabel("day")
    fig.tight_layout(); fig.savefig("voltarget.png", dpi=110)


if __name__ == "__main__":
    main()
