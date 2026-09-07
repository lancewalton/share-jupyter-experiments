"""Part 3: a REGIME-CONDITIONAL envelope -- fix calibration where it fails.

Part 2 showed the 95% envelope breaches concentrate in the turbulent regime
(10% vs 5%). Here we widen the envelope by a regime-dependent factor: on the
train half, measure how much the EWMA vol under-forecasts in each regime
(std of standardised residuals per state), then scale the OOS envelope by a
smooth blend of those factors weighted by the causal P(turbulent). Test whether
the turbulent-regime breach rate returns to nominal without spoiling the calm one.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

from mc.data import load_close
from mc.returns import log_returns
from mc.forecasters import _ewma_vol_series

SCALE = 100.0
_N = NormalDist()


def _causal_filter(x, means, vars_, trans, start):
    n, k = len(x), len(means)
    logpdf = -0.5 * (np.log(2 * np.pi * vars_) + (x[:, None] - means) ** 2 / vars_)
    pdf = np.exp(logpdf - logpdf.max(axis=1, keepdims=True))
    a = np.empty((n, k))
    a[0] = start * pdf[0]; a[0] /= a[0].sum()
    for t in range(1, n):
        a[t] = pdf[t] * (a[t - 1] @ trans); a[t] /= a[t].sum()
    return a


def _coverage(r, sig, mask, p):
    z = _N.inv_cdf(0.5 + p / 2)
    inside = np.abs(r[mask]) <= z * sig[mask]
    return float(inside.mean())


def main():
    r = log_returns(load_close().to_numpy())
    r = r[np.isfinite(r)]
    split = len(r) // 2
    x = (r * SCALE).reshape(-1, 1)
    hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=200,
                      random_state=0).fit(x[:split])
    vars_ = hmm.covars_.ravel(); turb = int(np.argmax(vars_))
    p_turb = _causal_filter(r * SCALE, hmm.means_.ravel(), vars_,
                            hmm.transmat_, hmm.startprob_)[:, turb]

    sig = np.sqrt(_ewma_vol_series(r, lam=0.94, burn=60))       # baseline 1-day vol
    z = r / sig                                                 # standardised residuals

    # fit regime correction factors on TRAIN (how much EWMA under-forecasts vol)
    tr = np.arange(60, split)
    hot_tr = p_turb[tr] >= 0.5
    s_calm = z[tr][~hot_tr].std()
    s_turb = z[tr][hot_tr].std()
    print(f"train residual std:  calm {s_calm:.3f}   turbulent {s_turb:.3f}   "
          f"(>1 => EWMA under-forecasts vol)")

    factor = s_calm + (s_turb - s_calm) * p_turb                # smooth blend by P(turb)
    sig_adj = sig * factor

    te = np.arange(split, len(r))
    hot = p_turb >= 0.5
    m_turb = te[hot[te]]; m_calm = te[~hot[te]]
    print(f"\nOOS breach rate (1 - coverage), baseline vs regime-adjusted:")
    print(f"  {'band':>6}{'regime':>8}{'baseline':>11}{'regime-adj':>12}  nominal")
    for p in (0.95, 0.99):
        for lab, m in [("all", te), ("calm", m_calm), ("turb", m_turb)]:
            b = 1 - _coverage(r, sig, m, p)
            a = 1 - _coverage(r, sig_adj, m, p)
            print(f"  {p:>6.0%}{lab:>8}{b:>11.3f}{a:>12.3f}  {1-p:.3f}")
        print()

    _plot(r, sig, sig_adj, p_turb, split)
    print("saved regime_calib.png")


def _plot(r, sig, sig_adj, p_turb, split):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    d = slice(split, len(r))
    ax.plot(np.abs(r)[d], color="0.7", lw=0.4, label="|return|")
    ax.plot((1.96 * sig)[d], color="tab:blue", lw=0.8, label="95% band (baseline)")
    ax.plot((1.96 * sig_adj)[d], color="tab:red", lw=0.8, label="95% band (regime-adj)")
    ax.set_ylabel("daily move"); ax.set_xlabel("OOS day"); ax.legend(fontsize=8)
    ax.set_title("Regime-conditional envelope widens in turbulence (FTSE OOS)")
    ax.set_ylim(0, np.percentile(np.abs(r)[d], 99.5))
    fig.tight_layout(); fig.savefig("regime_calib.png", dpi=110)


if __name__ == "__main__":
    main()
