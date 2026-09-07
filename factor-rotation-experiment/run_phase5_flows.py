"""Phase 5: where has the money gone, and where has it come from (1999-2026)?

Two views of performance (not activity):

  * INSTRUMENTS -- cumulative return of each asset: the intuitive winners and
    losers. (Price/level trend; no dividends/carry/roll, so directional not exact.)
  * FACTOR FLOWS -- each PCA factor is a long/short spread; its cumulative return
    is a net flow from its short leg to its long leg. A factor that drifts made
    persistent money one way; one that round-trips moved money and gave it back.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pca.data import PANEL, _fred_level, build_returns
from pca.factors import pca, standardize


def _legs(load, labels, k=3):
    order = np.argsort(-load)
    longs = ", ".join(labels[j] for j in order[:k] if load[j] > 0.15)
    shorts = ", ".join(labels[j] for j in order[::-1][:k] if load[j] < -0.15)
    return longs or "—", shorts or "—"


def main():
    R = build_returns()
    labels = list(R.columns)
    dates = R.index
    Z, _, _ = standardize(R.to_numpy())
    var, loadings, scores = pca(Z)

    # --- instruments: clean endpoint-to-endpoint total (no panel-drop artefact) ---
    total, cls = {}, {}
    for fid, kind, label, c in PANEL:
        if label == "VIX":
            continue
        lv = _fred_level(fid).loc[dates[0]:dates[-1]]        # the 1999-2026 window
        total[label] = (-(lv.iloc[-1] - lv.iloc[0]) if kind == "yield"
                        else float(np.log(lv.iloc[-1] / lv.iloc[0])))
        cls[label] = c
    final = dict(sorted(total.items(), key=lambda kv: -kv[1]))
    print(f"where the money went, {dates[0].date()}..{dates[-1].date()} "
          "(log price return; bonds = total yield decline):")
    for name, v in final.items():
        print(f"  {name:10s} {v:+6.2f}  ({cls[name]})")

    # sanity: standardised-PCA factor scores are mean-zero by construction, so they
    # carry NO drift. Project the RAW returns onto the loadings to keep the flow.
    fac_ret = R.to_numpy() @ loadings.T                  # (n, k) each col = factor P&L
    fac_cum = np.cumsum(fac_ret, axis=0)
    print("\nfactor flows — RAW returns projected onto each PCA spread "
          "(money short-leg -> long-leg):")
    order = np.argsort(-np.abs(fac_cum[-1]))[:5]
    fac = {}
    for i in order:
        L = loadings[i].copy(); c = fac_cum[:, i].copy()
        if c[-1] < 0:                                    # orient to the winning side
            L, c = -L, -c
        lo, sh = _legs(L, labels)
        fac[i] = (c, lo, sh)
        print(f"  PC{i+1} ({var[i]*100:.0f}% var): long [{lo}]  short [{sh}]   "
              f"net {c[-1]:+.2f}")

    _plot(dates, final, cls, fac, var)
    print("\nsaved phase5_flows.png")


def _plot(dates, final, cls, fac, var):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6))
    names = list(final)[::-1]; vals = [final[n] for n in names]
    colc = {"equity": "#0c757f", "rates": "#4a8ca8", "commodity": "#a8631a",
            "fx": "#6b8e23"}
    a1.barh(names, vals, color=[colc[cls[n]] for n in names])
    a1.axvline(0, color="k", lw=0.6)
    for n, v in zip(names, vals):
        a1.text(v + (0.05 if v >= 0 else -0.05), n, f"{v:+.1f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    a1.set_title("Where the money went, 1999–2026")
    a1.set_xlabel("cumulative return (log)  ·  teal=equity  blue=bonds  ochre=commodity  green=FX")
    a1.margins(x=0.15)

    cols = ["#0c757f", "#a8631a", "#6b8e23", "#4a8ca8", "#8a5115"]
    for (i, (c, lo, sh)), col in zip(fac.items(), cols):
        a2.plot(dates, c, lw=1.7, color=col,
                label=f"+{lo.split(',')[0]}  −{sh.split(',')[0]}  ({c[-1]:+.1f})")
    a2.axhline(0, color="k", lw=0.6); a2.margins(x=0)
    a2.set_title("The dominant flows — cumulative return of each factor spread")
    a2.set_ylabel("cumulative factor return (money short-leg → long-leg)")
    a2.legend(fontsize=8, loc="upper left", title="long (+) / short (−)")
    fig.tight_layout(); fig.savefig("phase5_flows.png", dpi=110)


if __name__ == "__main__":
    main()
