"""Phase 4: is the regime-timing edge robust to the rebalance grid?

Phase 3 gated on a monthly grid whose decision for the 2020 crash happened to land
after VIX had already risen. If that timing luck is doing the work, shifting the
grid start (offsets 0..20) should destroy the edge. We re-run naive / vol-gate /
VIX-gate at every offset and look at the distribution of Sharpe and worst-month.
A robust edge means the gated Sharpe beats naive at (almost) every offset.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.returns import log_returns
from vrp.data import panel
from vrp.forecast import har_walk_forward, trailing_rv
from vrp.premium import forward_realised_vol
from vrp.strategy import sharpe, varswap_pnl

H, MIN_TRAIN = 21, 252


def main():
    df = panel()
    rets = np.concatenate([[np.nan], log_returns(df["spx"].to_numpy())])
    vixv = df["vix"].to_numpy()
    feat = np.column_stack([trailing_rv(rets, (1, 5, 22)), vixv])
    target = forward_realised_vol(rets, H)
    pred = har_walk_forward(feat, target, H, MIN_TRAIN)   # once
    base = MIN_TRAIN + H + 1
    N = len(df)

    res = {r: {"sh": [], "worst": []} for r in ("naive", "vol-gate", "VIX-gate")}
    for off in range(H):
        idx = np.arange(base + off, N - H - 1, H)
        fc, vx, rv = pred[idx], vixv[idx], target[idx]
        pnl1 = varswap_pnl(vx, rv)
        ok = np.isfinite(fc) & np.isfinite(pnl1)
        fc, vx, pnl1 = fc[ok], vx[ok], pnl1[ok]
        med_fc = np.array([np.median(fc[:max(i, 1)]) for i in range(len(fc))])
        med_vx = np.array([np.median(vx[:max(i, 1)]) for i in range(len(vx))])
        gates = {"naive": np.ones(len(fc)),
                 "vol-gate": (fc < med_fc).astype(float),
                 "VIX-gate": (vx < med_vx).astype(float)}
        for r, g in gates.items():
            gn = g / (g.mean() + 1e-9)
            p = gn * pnl1
            res[r]["sh"].append(sharpe(p, 12))
            res[r]["worst"].append(float(p.min()))

    print(f"across {H} rebalance offsets:")
    print(f"{'rule':10s}  {'Sharpe mean':>12s}  {'[min, max]':>16s}   {'worst-month median':>18s}")
    for r in res:
        sh = np.array(res[r]["sh"]); wr = np.array(res[r]["worst"])
        print(f"{r:10s}  {sh.mean():+12.2f}  [{sh.min():+.2f}, {sh.max():+.2f}]"
              f"   {np.median(wr):+18.0f}")
    beat = np.mean([res["vol-gate"]["sh"][i] > res["naive"]["sh"][i] for i in range(H)])
    print(f"\nvol-gate beats naive at {100*beat:.0f}% of offsets")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    x = range(H)
    for r, c in (("naive", "tab:blue"), ("vol-gate", "tab:orange"), ("VIX-gate", "tab:green")):
        a1.plot(x, res[r]["sh"], "o-", color=c, label=r)
        a2.plot(x, res[r]["worst"], "o-", color=c, label=r)
    a1.set_title("Sharpe vs rebalance offset"); a1.set_xlabel("offset (days)")
    a1.set_ylabel("Sharpe"); a1.legend(fontsize=8); a1.axhline(0, color="k", lw=0.5)
    a2.set_title("Worst month vs rebalance offset"); a2.set_xlabel("offset (days)")
    a2.set_ylabel("worst month P&L"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("phase4_robustness.png", dpi=110)
    print("saved phase4_robustness.png")


if __name__ == "__main__":
    main()
