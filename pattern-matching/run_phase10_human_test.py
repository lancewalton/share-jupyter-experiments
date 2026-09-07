"""Phase 10 (Path 1): blind human-discrimination test.

Puts the human eye on the same footing as the detector. Each trial is a
2-alternative forced choice: two 1-year price charts, one REAL, one a vol-matched
FHS surrogate of the SAME segment (identical drift, identical volatility envelope
and clustering, only the directional order of the moves scrambled). The viewer
picks the real chart. Score well above chance => the eye sees genuine directional
structure; at chance => visual conviction is not separable from matched noise.

The answer key is written to the scratchpad (NOT printed), so the test is blind.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matching.surrogate import fhs_surrogate
from matching.universe import load_universe

NPAIRS, SEG = 12, 252
KEY_PATH = ("/private/tmp/claude-501/-Users-lance-Projects-share-jupyter-experiments-"
            "pattern-matching/0cafc70d-50b0-4444-9ce8-4474c05658c4/scratchpad/"
            "human_test_key.json")


def _norm(logP_seg):
    return np.exp(logP_seg - logP_seg[0]) * 100.0


def main(seed=20260821):
    rng = np.random.default_rng(seed)
    uni = [s for s in load_universe() if len(s[1]) > SEG + 200]
    picks = rng.choice(len(uni), NPAIRS, replace=False)

    fig, axes = plt.subplots(NPAIRS, 2, figsize=(11, 2.1 * NPAIRS))
    key = {}
    for i, sidx in enumerate(picks):
        _, logP, _ = uni[sidx]
        a = int(rng.integers(120, len(logP) - SEG - 1))
        seg = logP[a:a + SEG]
        real = _norm(seg)
        sur = _norm(fhs_surrogate(seg, rng))
        real_is_A = bool(rng.integers(0, 2))
        left, right = (real, sur) if real_is_A else (sur, real)
        key[str(i + 1)] = "A" if real_is_A else "B"
        for ax, series, tag in ((axes[i, 0], left, "A"), (axes[i, 1], right, "B")):
            ax.plot(series, lw=1.1, color="#14181b")
            ax.set_title(f"Pair {i+1} — {tag}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#bbbbbb")
    fig.suptitle("Which is the REAL chart in each pair?  (the other is a "
                 "vol-matched surrogate)", y=1.002, fontsize=12)
    fig.tight_layout()
    fig.savefig("phase10_human_test.png", dpi=120, bbox_inches="tight")

    with open(KEY_PATH, "w") as f:
        json.dump({"seed": seed, "key": key}, f)
    print(f"generated {NPAIRS} pairs -> phase10_human_test.png")
    print("answer key written (hidden) to scratchpad")


if __name__ == "__main__":
    main()
