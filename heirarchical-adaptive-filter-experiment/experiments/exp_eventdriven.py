"""Event-driven higher level: does RE-FITTING the winning stack help at all?

Our best model is a fixed signed stack of diverse experts. Continuous adaptation
never beat it. Here we test the remaining option: re-estimating the stack (either
expanding walk-forward, or on a trailing window that FORGETS old data -- the thing
a change-point re-fit ultimately does). If no trailing window beats the single
static fit, there is no exploitable structural break and event-driven is dead too.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from haf import load_ftse, rolling_mean_baseline
from haf.mixture import build_experts
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
from exp_mixture import block_inputs, skill_vs_base


def fit_predict(P, v, fit_idx, at):
    """Least-squares stack on rows fit_idx, predict row `at`."""
    A = np.column_stack([np.ones(fit_idx.sum()), P[fit_idx]])
    coef, _, _, _ = np.linalg.lstsq(A, v[fit_idx], rcond=None)
    return coef[0] + P[at] @ coef[1:]


def run(r, k, label):
    v, sign = block_inputs(r, k)
    n = len(v)
    split = n // 2
    idx = slice(split, n)
    base = rolling_mean_baseline(v, 8)
    experts = build_experts(v, sign)
    names = list(experts)
    P = np.array([experts[nm] for nm in names]).T
    valid = np.all(np.isfinite(P), axis=1) & np.isfinite(v)

    # static: fit once on the training half
    tr = valid.copy(); tr[split:] = False
    coef, _, _, _ = np.linalg.lstsq(
        np.column_stack([np.ones(tr.sum()), P[tr]]), v[tr], rcond=None)
    static = np.full(n, np.nan)
    te = valid.copy(); te[:split] = False
    static[te] = coef[0] + P[te] @ coef[1:]

    print(f"\n=== {label} block k={k} ({n} blocks) ===")
    print(f"  static (fit once)          {skill_vs_base(v, static, base, idx):+.4f}")

    # walk-forward re-fit on a trailing window of length W (W=inf -> expanding)
    for W in (np.inf, 400, 200, 100, 50):
        pred = np.full(n, np.nan)
        for j in range(split, n):
            if not valid[j]:
                continue
            lo = 0 if np.isinf(W) else max(0, j - int(W))
            fit_idx = valid.copy()
            fit_idx[:lo] = False
            fit_idx[j:] = False
            if fit_idx.sum() < len(names) + 2:
                continue
            pred[j] = fit_predict(P, v, fit_idx, j)
        wlabel = "expanding" if np.isinf(W) else f"trailing {int(W)}"
        print(f"  refit {wlabel:14s}       {skill_vs_base(v, pred, base, idx):+.4f}")


if __name__ == "__main__":
    df = load_ftse()
    r = df.ret.to_numpy()
    for k in (5, 10):
        run(r, k, "REAL FTSE")
