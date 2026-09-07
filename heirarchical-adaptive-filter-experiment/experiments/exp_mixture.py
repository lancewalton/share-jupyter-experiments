"""Does a structurally-different hierarchy (mix diverse timescale/structure
experts) beat the best single expert -- where nested learning rates could not?

Target: non-overlapping block realized VARIANCE on real FTSE (artefact-free).
Baselines: each single expert; equal-weight (fixed) mix. Then Level 2 (adaptive
Hedge mixer) and Level 3 (adaptive eta). Shuffled control for honesty.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from haf import load_ftse, realized_vol_blocks, rolling_mean_baseline
from haf.mixture import build_experts, Mixer, MetaMixer, run_mixer


def block_inputs(r, k):
    r = np.asarray(r, float)
    nb = len(r) // k
    rr = r[:nb * k].reshape(nb, k)
    rv_var = (rr ** 2).mean(axis=1)
    block_sign = np.sign(rr.sum(axis=1))
    return rv_var, block_sign


def skill_vs_base(actual, pred, base, idx):
    a, p, b = actual[idx], pred[idx], base[idx]
    m = np.isfinite(a) & np.isfinite(p) & np.isfinite(b)
    a, p, b = a[m], p[m], b[m]
    return 1 - np.sum((a - p) ** 2) / np.sum((a - b) ** 2)


def evaluate(r, k, eta=5.0, meta=0.1, label=""):
    v, sign = block_inputs(r, k)
    n = len(v)
    split = n // 2
    idx = slice(split, n)
    base = rolling_mean_baseline(v, 8)
    experts = build_experts(v, sign)

    rows = {}
    for name, p in experts.items():
        rows["expert:" + name] = skill_vs_base(v, p, base, idx)
    # fixed equal-weight mix
    eq = np.nanmean(np.array([experts[k_] for k_ in experts]), axis=0)
    rows["equal_mix(L0)"] = skill_vs_base(v, eq, base, idx)
    # Level 2 adaptive mixer
    p2, w2, names = run_mixer(experts, v, Mixer(len(experts), eta=eta))
    rows["mixer(L2)"] = skill_vs_base(v, p2, base, idx)
    # Level 3 adaptive-eta mixer
    p3, _, _ = run_mixer(experts, v, MetaMixer(len(experts), eta=eta, meta=meta))
    rows["metamixer(L3)"] = skill_vs_base(v, p3, base, idx)
    print(f"\n=== {label} block k={k} ({n} blocks) ===")
    for kk, vv in rows.items():
        print(f"  {kk:18s} {vv:+.4f}")
    # held-out average mixer weights (which experts does L2 rely on?)
    wmean = np.nanmean(w2[split:], axis=0)
    print("  L2 mean weights:", {names[i]: round(float(wmean[i]), 2) for i in range(len(names))})
    return rows


if __name__ == "__main__":
    df = load_ftse()
    r = df.ret.to_numpy()
    for k in (5, 10):
        evaluate(r, k, label="REAL FTSE")
    # shuffled control
    rng = np.random.default_rng(0)
    rs = r.copy(); rng.shuffle(rs)
    evaluate(rs, 5, label="SHUFFLED control")
