"""Phase 0: the non-starter gate.

The whole programme rests on one assumption: a mechanism that decomposes recent
price paths into recurring components should discover, *on its own and first*,
the fundamental pattern -- the constant-annualised-growth drift ramp. If it
can't even do that, the idea is dead.

We do not pre-subtract the trend and we do not mean-centre the pooled windows;
instead we run an UNCENTRED SVD (the linear, deflationary base case of the
growing autoencoder to come) and ask:

  1. Is the leading component a monotone upward ramp?
  2. Does its activation reproduce each window's realised drift?
  3. Does the same ramp emerge at every time scale (20..260 days)?

A "yes" to all three clears the gate. What the *next* components look like is
reported too -- those are the "most distinct from the fundamental pattern"
shapes we actually came for.
"""
from __future__ import annotations

import glob

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.data import load_close
from mc.returns import log_returns
from matching.represent import vol_scaled_windows

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
SCALES = [20, 40, 80, 160, 260]
P, STRIDE, LOOKBACK = 32, 5, 60
MIN_ROWS, GLITCH = 1500, 0.6
REP = 80  # representative scale for the detailed panels


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _load_universe():
    series = []
    for pattern in DIRS:
        for path in sorted(glob.glob(pattern)):
            try:
                s = load_close(path)
            except Exception:
                continue
            r = log_returns(s.to_numpy())
            if len(r) < MIN_ROWS or np.abs(r).max() > GLITCH:
                continue
            series.append(np.log(s.to_numpy()))
    return series


def _pool(series, L):
    Ws, drifts = [], []
    for logP in series:
        W, ends, starts, drift = vol_scaled_windows(
            logP, L=L, P=P, stride=STRIDE, lookback=LOOKBACK)
        ok = np.isfinite(W).all(axis=1) & np.isfinite(drift)
        if ok.any():
            Ws.append(W[ok]); drifts.append(drift[ok])
    return np.vstack(Ws), np.concatenate(drifts)


def _svd_components(W, k=8):
    """Uncentred SVD. Returns (shapes, activations, var_explained).

    Each component is sign-oriented so its shape rises left-to-right (upward),
    matching the convention that a positive activation means an upward move.
    """
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    shapes = Vt[:k].copy()
    acts = (U[:, :k] * S[:k])
    for i in range(k):
        if shapes[i, -1] < shapes[i, 0]:
            shapes[i] *= -1; acts[:, i] *= -1
    total = (S ** 2).sum()
    var = (S[:k] ** 2) / total
    return shapes, acts, var


def main():
    series = _load_universe()
    print(f"universe: {len(series)} stocks\n")

    # --- per-scale gate: does component 1 = drift ramp at every scale? ---
    comp1_by_scale, gate_rows = {}, []
    for L in SCALES:
        W, drift = _pool(series, L)
        shapes, acts, var = _svd_components(W)
        rampness = _corr(shapes[0], np.linspace(0, 1, P))  # monotone-up similarity
        act_vs_drift = _corr(acts[:, 0], drift)
        comp1_by_scale[L] = shapes[0]
        gate_rows.append((L, len(W), var[0], rampness, act_vs_drift))
        print(f"L={L:3d}  windows={len(W):>7d}  var1={var[0]:.2f}  "
              f"ramp(comp1)={rampness:+.3f}  corr(act1,drift)={act_vs_drift:+.3f}")

    print()
    # --- detailed decomposition at the representative scale ---
    W, drift = _pool(series, REP)
    shapes, acts, var = _svd_components(W)
    print(f"representative L={REP}: variance explained by first 6 components:")
    print("  " + "  ".join(f"c{i+1}={v:.3f}" for i, v in enumerate(var[:6])))
    print(f"  corr(activation_1, realised drift) = {_corr(acts[:, 0], drift):+.3f}")

    _plot(comp1_by_scale, shapes, var, acts[:, 0], drift, gate_rows)
    print("\nsaved phase0_gate.png")


def _plot(comp1_by_scale, shapes, var, act1, drift, gate_rows):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    xg = np.linspace(0, 1, P)

    a = ax[0, 0]
    for L, sh in comp1_by_scale.items():
        a.plot(xg, sh, label=f"L={L}")
    a.axhline(0, color="k", lw=0.5, ls=":")
    a.set_title("Component 1 shape at each scale\n(should be a rising ramp everywhere)")
    a.set_xlabel("normalised time through window"); a.legend(fontsize=8)

    a = ax[0, 1]
    for i in range(4):
        a.plot(xg, shapes[i], label=f"c{i+1} ({var[i]*100:.0f}%)")
    a.axhline(0, color="k", lw=0.5, ls=":")
    a.set_title(f"Components 1-4 at L={REP}\n(c2+ = 'most distinct from the fundamental')")
    a.set_xlabel("normalised time through window"); a.legend(fontsize=8)

    a = ax[1, 0]
    idx = np.random.default_rng(0).choice(len(act1), size=min(4000, len(act1)), replace=False)
    a.scatter(drift[idx], act1[idx], s=4, alpha=0.25)
    a.set_xlabel("realised drift (vol units)"); a.set_ylabel("component-1 activation")
    a.set_title(f"Activation 1 vs realised drift  (corr {_corr(act1, drift):+.3f})")

    a = ax[1, 1]
    Ls = [r[0] for r in gate_rows]
    a.plot(Ls, [r[3] for r in gate_rows], "o-", label="ramp(comp1)")
    a.plot(Ls, [r[4] for r in gate_rows], "s-", label="corr(act1, drift)")
    a.plot(Ls, [r[2] for r in gate_rows], "^-", label="variance in comp1")
    a.set_ylim(0, 1.02); a.set_xlabel("scale L (days)")
    a.set_title("Gate metrics vs scale"); a.legend(fontsize=8)

    fig.tight_layout(); fig.savefig("phase0_gate.png", dpi=110)


if __name__ == "__main__":
    main()
