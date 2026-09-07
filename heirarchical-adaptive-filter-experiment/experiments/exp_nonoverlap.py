"""Do Levels 2 and 3 beat Level 1 on a NON-OVERLAPPING realized-vol target?

The overlapping rolling-RV target inflated skill (shuffled control still scored
~0.84). Non-overlapping blocks share no observations, so any surviving skill is
real. We test on the synthetic ground-truth rig (with the step-size tracking
check) and on 32 years of real FTSE.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd
from haf import *
from haf.models import run_idbd

NL = 20


def block_series(r, k):
    return realized_vol_blocks(r, k)


def step_tracking(r, m, fast, k, theta=0.1):
    """On the synthetic block series, does the IDBD step size track true |dm/dt|?"""
    v = block_series(r, k)
    split = len(v) // 2
    # aggregate the ground-truth rate of change to the same block grid
    roc = true_rate_of_change(m)
    nb = len(v)
    roc_b = roc[:nb * k].reshape(nb, k).mean(axis=1)
    fast_b = fast[:nb * k].reshape(nb, k).mean(axis=1) > 0.5
    _, step, _ = run_idbd(v, lambda K: IDBD(K, theta=theta), NL, split)
    mask = np.isfinite(step)
    c = np.corrcoef(step[mask], roc_b[mask])[0, 1]
    return c, step[mask][~fast_b[mask]].mean(), step[mask][fast_b[mask]].mean(), nb


def run(label, r, m=None, fast=None):
    print(f"\n########## {label} ##########")
    for k in (5, 10):
        v = block_series(r, k)
        print(f"\n--- non-overlapping RV, block k={k}  ({len(v)} points) ---")
        tbl = compare(v, n_lags=NL, roll=10)
        print(tbl.round(4).to_string())
        # shuffled control
        vs = block_series(shuffled_returns(r), k)
        s = compare(vs, n_lags=NL, roll=10)["Skill_vs_base"]
        print("shuffled-control Skill_vs_base:",
              {i: round(x, 4) for i, x in s.items()})
        if m is not None:
            c, calm, turb, _ = step_tracking(r, m, fast, k)
            print(f"ground-truth: corr(IDBD step, true |dm/dt|)={c:+.3f} "
                  f"(step calm={calm:.4f} turbulent={turb:.4f})")


if __name__ == "__main__":
    r_s, m_s, fast_s = synth_vol_series()
    run("SYNTHETIC (ground truth)", r_s, m_s, fast_s)

    df = load_ftse()
    print(f"\nFTSE: {len(df)} returns {df.Date.min().date()}..{df.Date.max().date()}")
    run("REAL FTSE", df.ret.to_numpy())
