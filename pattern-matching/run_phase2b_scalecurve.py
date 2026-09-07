"""Phase 2b: the linear direction signal across the full scale grid.

Same test as phase 2, but swept over EVERY length L in 20..260 (step 10), so we
can see how the (reversion-flavoured) activation IC varies with scale rather than
at five sampled points. Reports, per scale: component-1 activation IC (the
trailing-drift / momentum-reversion probe), the best single-component |IC|, and
the fitted-combination IC (expected to underperform -- fitted weights fit noise
at low SNR).
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_phase2_direction import evaluate
from matching.universe import load_universe

SCALES = list(range(20, 261, 10))
H = 20


def main():
    uni = load_universe()
    print(f"universe: {len(uni)} stocks   scales {SCALES[0]}..{SCALES[-1]} step 10\n")
    c1, best, comb, z = [], [], [], []
    for L in SCALES:
        r = evaluate(uni, L, H)
        c1.append(r["per_comp"][0])
        best.append(max(r["per_comp"], key=abs))
        comb.append(r["ic"]); z.append(r["z"])
        print(f"  L={L:3d}  c1 {r['per_comp'][0]:+.4f}  best|c| {best[-1]:+.4f}  "
              f"combined {r['ic']:+.4f}  z={r['z']:+.2f}")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhline(0, color="k", lw=0.6)
    ax.plot(SCALES, c1, "o-", label="component-1 activation IC (drift/reversion)")
    ax.plot(SCALES, best, "s-", label="best single-component |IC| (signed)")
    ax.plot(SCALES, comb, "^--", color="0.5", label="fitted-combination IC")
    ax.set_xlabel("window length L (days)"); ax.set_ylabel(f"OOS rank-IC (H={H})")
    ax.set_title("Linear direction signal vs scale (every L, 20..260 step 10)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig("phase2b_scalecurve.png", dpi=110)
    print("\nsaved phase2b_scalecurve.png")


if __name__ == "__main__":
    main()
