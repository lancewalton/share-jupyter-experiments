"""Fold the regime correction into full FHS (multi-day), by regime, OOS.

Scale the whole FHS envelope by a regime factor f_t = s_calm + (s_turb-s_calm)*
P_turb(causal), fit on train. Does it fix FHS's multi-day tail under-coverage,
and is the fix concentrated in the turbulent regime (leaving calm intact)?
"""
from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM

from mc.data import load_close
from mc.returns import log_returns, realised_cumulative
from mc.forecasters import fhs_ewma, _ewma_vol_series
from mc.scoring import interval_hit
from mc.walkforward import DENSE_LEVELS

SCALE, T, STRIDE = 100.0, 10, 5
LEVELS = [0.90, 0.95, 0.99]
HORIZONS = [1, 5, 10]


def _causal_filter(x, means, vars_, trans, start):
    n, k = len(x), len(means)
    logpdf = -0.5 * (np.log(2 * np.pi * vars_) + (x[:, None] - means) ** 2 / vars_)
    pdf = np.exp(logpdf - logpdf.max(axis=1, keepdims=True))
    a = np.empty((n, k)); a[0] = start * pdf[0]; a[0] /= a[0].sum()
    for t in range(1, n):
        a[t] = pdf[t] * (a[t - 1] @ trans); a[t] /= a[t].sum()
    return a


def _idx(p, side):
    lv = (1 - p) / 2 if side == "lo" else 1 - (1 - p) / 2
    return int(np.argmin(np.abs(DENSE_LEVELS - lv)))


def main():
    r = log_returns(load_close().to_numpy())
    r = r[np.isfinite(r)]
    split = len(r) // 2
    hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=200,
                      random_state=0).fit((r[:split] * SCALE).reshape(-1, 1))
    vars_ = hmm.covars_.ravel(); turb = int(np.argmax(vars_))
    p_turb = _causal_filter(r * SCALE, hmm.means_.ravel(), vars_,
                            hmm.transmat_, hmm.startprob_)[:, turb]
    sig = np.sqrt(_ewma_vol_series(r, lam=0.94, burn=60)); z = r / sig
    tr = np.arange(60, split); hot_tr = p_turb[tr] >= 0.5
    s_calm, s_turb = z[tr][~hot_tr].std(), z[tr][hot_tr].std()
    f = s_calm + (s_turb - s_calm) * p_turb
    print(f"regime factors: calm {s_calm:.3f}  turb {s_turb:.3f}\n")

    lev = DENSE_LEVELS
    fc = fhs_ewma(lam=0.94, n_sims=10_000, seed=0)
    bounds = {p: (_idx(p, "lo"), _idx(p, "hi")) for p in LEVELS}
    # hits[model][regime][p][h] -> [n_hit, n_total]
    acc = {m: {g: {p: np.zeros((len(HORIZONS), 2)) for p in LEVELS}
               for g in ("calm", "turb")} for m in ("fhs", "fhs_regime")}

    origins = range(split, len(r) - T - 1, STRIDE)
    for t in origins:
        y = realised_cumulative(r, t, T)
        q = fc(r[: t + 1], T, lev)
        qr = f[t] * q
        g = "turb" if p_turb[t] >= 0.5 else "calm"
        for m, qq in (("fhs", q), ("fhs_regime", qr)):
            for p, (lo, hi) in bounds.items():
                for hi_i, h in enumerate(HORIZONS):
                    hit = interval_hit(qq[lo, h - 1], qq[hi, h - 1], y[h - 1])
                    acc[m][g][p][hi_i] += [1.0 if hit else 0.0, 1.0]

    print("OOS breach rate (1-coverage) by regime, h=1/5/10  [fhs -> fhs_regime]:")
    for p in LEVELS:
        print(f"  {p:.0%} band (nominal breach {1-p:.3f}):")
        for g in ("calm", "turb"):
            b = 1 - acc["fhs"][g][p][:, 0] / acc["fhs"][g][p][:, 1]
            a = 1 - acc["fhs_regime"][g][p][:, 0] / acc["fhs_regime"][g][p][:, 1]
            nt = int(acc["fhs"][g][p][0, 1])
            print(f"    {g:<5} (n={nt:>4}): "
                  + "  ".join(f"h{h}:{bb:.3f}->{aa:.3f}" for h, bb, aa in zip(HORIZONS, b, a)))


if __name__ == "__main__":
    main()
