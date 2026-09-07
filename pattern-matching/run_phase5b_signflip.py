"""Phase 5b: is the detected structure DIRECTIONAL, or leftover volatility?

Phase 5 found real windows distinguishable from vol-matched FHS surrogates -- but
the FHS null left some residual volatility clustering (|resid| autocorr ~0.10),
so the classifier might be keying on second-moment texture, not direction.

The sign-flip surrogate settles it: it keeps every return magnitude in place
(volatility path byte-identical -- clustering, fat tails, trailing-vol scaling
all unchanged) and randomises only the signs. Any real-vs-sign-flip signal is
therefore purely directional. We run both nulls side by side, each with its own
same-null control.

  * real vs sign-flip strong AND ~= real vs FHS -> directional structure, robust.
  * real vs sign-flip ~ 0 while real vs FHS large -> the FHS signal was vol, not
    direction (idea dies as a first-moment claim).
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matching.surrogate import fhs_surrogate, sign_flip_surrogate
from matching.universe import load_universe
from run_phase5_detector import _collect, _fit_score, _labelled

RNG = np.random.default_rng(0)


def _compare(uni, L, transform, name):
    Wr, dr = _collect(uni, L)
    Wa, da = _collect(uni, L, transform=lambda lp: transform(lp, RNG))
    r = _fit_score(*_labelled(Wr, dr, Wa, da))
    Wb, db = _collect(uni, L, transform=lambda lp: transform(lp, RNG))
    c = _fit_score(*_labelled(Wa, da, Wb, db))       # same-null control
    print(f"  L={L:3d}  real-vs-{name:9s}  AUC {r['auc']:.3f}  enrich@1% {r['e1']:+.3f}"
          f"  @5% {r['e5']:+.3f}   | control AUC {c['auc']:.3f}  @5% {c['e5']:+.3f}")
    return r, c


def main():
    uni = load_universe()
    print(f"universe: {len(uni)} stocks\n")
    print("Disentangling nulls (FHS = beyond growth+vol-envelope; "
          "sign-flip = purely directional):")
    res = {}
    for L in (40, 80):
        res[("fhs", L)] = _compare(uni, L, fhs_surrogate, "FHS")
        res[("sign", L)] = _compare(uni, L, sign_flip_surrogate, "sign-flip")
        print()

    print("=== verdict ===")
    sign_e5 = max(res[("sign", L)][0]["e5"] for L in (40, 80))
    sign_ctrl = max(abs(res[("sign", L)][1]["e5"]) for L in (40, 80))
    fhs_e5 = max(res[("fhs", L)][0]["e5"] for L in (40, 80))
    print(f"sign-flip (pure direction): best enrich@5% {sign_e5:+.3f}  "
          f"(control {sign_ctrl:.3f})")
    print(f"FHS (beyond growth+vol):    best enrich@5% {fhs_e5:+.3f}")
    if sign_e5 > 0.03 and sign_e5 > 3 * sign_ctrl:
        print("=> DIRECTIONAL structure confirmed (volatility held exactly).")
    else:
        print("=> no directional structure; the FHS signal was residual volatility.")
    _plot(res)


def _plot(res):
    Ls = [40, 80]
    x = np.arange(len(Ls))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(x - 0.27, [res[("fhs", L)][0]["e5"] for L in Ls], 0.24,
           label="real vs FHS (beyond growth+vol)")
    ax.bar(x - 0.03, [res[("sign", L)][0]["e5"] for L in Ls], 0.24,
           label="real vs sign-flip (pure direction, vol held)")
    ax.bar(x + 0.21, [max(res[("fhs", L)][1]["e5"], res[("sign", L)][1]["e5"]) for L in Ls],
           0.24, color="0.7", label="same-null controls")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([f"L={L}" for L in Ls])
    ax.set_ylabel("tail enrichment @5% (top-scored windows)")
    ax.set_title("Directional structure exists beyond volatility\n"
                 "(real distinguishable from a vol-identical sign-flipped fake)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig("phase5b_signflip.png", dpi=110)
    print("saved phase5b_signflip.png")


if __name__ == "__main__":
    main()
