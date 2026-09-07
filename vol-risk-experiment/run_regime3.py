"""Task 1: 3-state HMM (calm / normal / crisis) + regime-conditional envelope.

The 2-state model left the turbulent-regime 95% breach at 6.5% (nominal 5%).
Does a finer 3-state split, with its own crisis-state width correction, reach
nominal in the worst state without spoiling the calm one? FTSE, causal filter.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
from hmmlearn.hmm import GaussianHMM

from mc.data import load_close
from mc.returns import log_returns
from mc.forecasters import _ewma_vol_series

SCALE, ANN = 100.0, 252
_N = NormalDist()
LEVELS = [0.90, 0.95, 0.99]


def _causal_filter(x, means, vars_, trans, start):
    n, k = len(x), len(means)
    lp = -0.5 * (np.log(2 * np.pi * vars_) + (x[:, None] - means) ** 2 / vars_)
    pdf = np.exp(lp - lp.max(axis=1, keepdims=True))
    a = np.empty((n, k)); a[0] = start * pdf[0]; a[0] /= a[0].sum()
    for t in range(1, n):
        a[t] = pdf[t] * (a[t - 1] @ trans); a[t] /= a[t].sum()
    return a


def _spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    ra, rb = np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    r = log_returns(load_close().to_numpy()); r = r[np.isfinite(r)]
    split = len(r) // 2
    x = (r * SCALE).reshape(-1, 1)
    hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=300,
                      random_state=0).fit(x[:split])
    vars_ = hmm.covars_.ravel(); order = np.argsort(vars_)   # low->high vol states
    names3 = ["calm", "normal", "crisis"]
    vol_ann = np.sqrt(vars_) / SCALE * np.sqrt(ANN)
    P = _causal_filter(r * SCALE, hmm.means_.ravel(), vars_, hmm.transmat_, hmm.startprob_)
    dur = 1 / (1 - np.diag(hmm.transmat_))
    print("3-state HMM (FTSE, fit on first half):")
    for st, nm in zip(order, names3):
        print(f"  {nm:<7} ann.vol {vol_ann[st]:.1%}  duration {dur[st]:>4.0f}d  "
              f"mean {hmm.means_.ravel()[st]/SCALE:+.2e}")
    crisis = order[-1]
    # forward vol IC of P(crisis)
    H = 20; frv = np.full(len(r), np.nan)
    for t in range(len(r) - H):
        frv[t] = np.sqrt(np.mean(r[t + 1:t + 1 + H] ** 2))
    te = np.arange(split, len(r) - H)
    print(f"  P(crisis) forward-{H}d vol IC (OOS): {_spearman(P[te, crisis], frv[te]):+.3f}")

    # regime-conditional envelope: per-state residual-std correction, blended by P
    sig = np.sqrt(_ewma_vol_series(r, lam=0.94, burn=60)); z = r / sig
    tr = np.arange(60, split)
    st_tr = P[tr].argmax(1)
    s_k = np.array([z[tr][st_tr == k].std() if (st_tr == k).sum() > 30 else 1.0
                    for k in range(3)])
    print(f"  train residual std per state (calm/normal/crisis): "
          f"{s_k[order[0]]:.3f} / {s_k[order[1]]:.3f} / {s_k[order[2]]:.3f}")
    # hard state-assignment for the factor (soft blend is poisoned by the degenerate
    # transient state's huge residual std); clip factor to a sane range
    factor = np.clip(s_k[P.argmax(1)], 0.8, 2.0)
    sig_adj = sig * factor

    idx = np.arange(split, len(r)); st = P[idx].argmax(1)
    print("\nOOS breach rate (1-coverage) by state, baseline -> regime-adjusted:")
    print(f"  {'band':>6}{'state':>8}{'n':>6}{'baseline':>11}{'regime-adj':>12}  nominal")
    for p in LEVELS:
        for k, nm in zip(order, names3):
            m = idx[st == k]
            if len(m) < 20:
                continue
            b = 1 - np.mean(np.abs(r[m]) <= _N.inv_cdf(0.5 + p / 2) * sig[m])
            a = 1 - np.mean(np.abs(r[m]) <= _N.inv_cdf(0.5 + p / 2) * sig_adj[m])
            print(f"  {p:>6.0%}{nm:>8}{len(m):>6}{b:>11.3f}{a:>12.3f}  {1-p:.3f}")
        print()


if __name__ == "__main__":
    main()
