"""Phase 1: is the variance risk premium real, on VIX + S&P 500?

Compare VIX (30-day implied vol) at each day t with the S&P's *subsequent*
21-day realised vol. If implied systematically exceeds realised, the premium is
there to harvest (by selling variance) -- and we quantify how big, how often, and
where it goes wrong (the vol spikes that punish sellers).
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.returns import log_returns
from vrp.data import panel
from vrp.premium import forward_realised_vol, variance_premium

H = 21


def main():
    df = panel()
    rets = np.concatenate([[np.nan], log_returns(df["spx"].to_numpy())])
    vix = df["vix"].to_numpy()
    fwd_rv = forward_realised_vol(rets, H)
    vol_p, var_p = variance_premium(vix, fwd_rv)

    ok = np.isfinite(vol_p)
    vp = vol_p[ok]; dates = df.index.to_numpy()[ok]
    print(f"panel: {ok.sum()} days  {df.index[0].date()}..{df.index[-1].date()}\n")
    print(f"mean VIX (implied)         {np.nanmean(vix[ok]):.2f}")
    print(f"mean forward realised vol  {np.nanmean(fwd_rv[ok]):.2f}")
    print(f"mean premium (VIX - RV)    {vp.mean():+.2f} vol points")
    print(f"  premium > 0 on           {100*np.mean(vp > 0):.0f}% of days")
    print(f"  median / 5th pct / min   {np.median(vp):+.2f} / {np.percentile(vp,5):+.2f} / {vp.min():+.2f}")
    # does implied predict realised at all? (level correlation)
    m = np.isfinite(vix) & np.isfinite(fwd_rv)
    print(f"corr(VIX, forward RV)      {np.corrcoef(vix[m], fwd_rv[m])[0,1]:+.2f}")
    # worst seller episodes (most negative premium)
    order = np.argsort(vp)[:5]
    print("\nworst 5 days for a variance seller (realised >> implied):")
    for i in order:
        print(f"  {str(dates[i])[:10]}  premium {vp[i]:+.1f}  (VIX {vix[ok][i]:.1f}, RV {fwd_rv[ok][i]:.1f})")

    # sub-period means
    print("\nmean premium by period:")
    for part in np.array_split(np.arange(len(vp)), 5):
        yr = str(dates[part[0]])[:4] + "-" + str(dates[part[-1]])[:4]
        print(f"  {yr}: {vp[part].mean():+.2f}")

    _plot(dates, vix[ok], fwd_rv[ok], vp)
    print("\nsaved phase1_premium.png")


def _plot(dates, vix, fwd, vp):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].plot(dates, vix, lw=0.7, label="VIX (implied)")
    ax[0].plot(dates, fwd, lw=0.7, alpha=0.8, label="forward 21d realised")
    ax[0].set_title("Implied vs subsequent realised vol"); ax[0].legend(fontsize=8)
    ax[1].scatter(fwd, vix, s=3, alpha=0.15)
    lim = max(vix.max(), fwd.max())
    ax[1].plot([0, lim], [0, lim], "r--", lw=0.8)
    ax[1].set_xlabel("forward realised vol"); ax[1].set_ylabel("VIX (implied)")
    ax[1].set_title("VIX usually above the line = premium")
    ax[2].hist(vp, bins=80, color="0.5")
    ax[2].axvline(0, color="r", lw=1); ax[2].axvline(vp.mean(), color="tab:blue", lw=1.5)
    ax[2].set_title(f"Premium distribution (mean {vp.mean():+.1f}, left tail = seller losses)")
    ax[2].set_xlabel("VIX - realised (vol points)")
    fig.tight_layout(); fig.savefig("phase1_premium.png", dpi=110)


if __name__ == "__main__":
    main()
