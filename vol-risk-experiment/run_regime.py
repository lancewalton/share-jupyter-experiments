"""Part 2: volatility REGIME detection as a first-class object.

A 2-state Gaussian HMM on daily returns finds a calm and a turbulent state
(different variances). We fit it on the train half, then run a CAUSAL forward
filter (uses only data up to t) to get P(turbulent | past) each day -- an honest,
online regime signal. We validate it four ways:
  1. state separation  -- turbulent vol / calm vol
  2. persistence       -- expected regime durations from the transition matrix
  3. predicts vol      -- IC of causal P(turbulent) vs forward realised vol
  4. explains breaches  -- do 95%-envelope breaches concentrate in the turbulent state?
Plus the sizing angle: is the turbulent state actually one to DE-RISK (are its
forward returns worse)?
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

from mc.data import load_close
from mc.returns import log_returns
from mc.forecasters import _ewma_vol_series

ANN, H, SCALE = 252, 20, 100.0


def _spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    ra, rb = np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def _causal_filter(x, means, vars_, trans, start):
    """Forward-filtered P(state | x[:t+1]) -- causal, no future leakage."""
    n, k = len(x), len(means)
    logpdf = -0.5 * (np.log(2 * np.pi * vars_) + (x[:, None] - means) ** 2 / vars_)
    pdf = np.exp(logpdf - logpdf.max(axis=1, keepdims=True))
    a = np.empty((n, k))
    a[0] = start * pdf[0]; a[0] /= a[0].sum()
    for t in range(1, n):
        a[t] = pdf[t] * (a[t - 1] @ trans)
        a[t] /= a[t].sum()
    return a


def main():
    r = log_returns(load_close().to_numpy())
    r = r[np.isfinite(r)]
    x = (r * SCALE).reshape(-1, 1)
    split = len(r) // 2

    hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=200,
                      random_state=0).fit(x[:split])
    means = hmm.means_.ravel(); vars_ = hmm.covars_.ravel()
    turb = int(np.argmax(vars_)); calm = 1 - turb
    vol_state = np.sqrt(vars_) / SCALE * np.sqrt(ANN)  # annualised per-state vol

    print(f"FTSE {len(r)} returns; HMM fit on first {split}.")
    print(f"  calm  state: ann.vol {vol_state[calm]:.1%}  daily mean {means[calm]/SCALE:+.2e}")
    print(f"  turb  state: ann.vol {vol_state[turb]:.1%}  daily mean {means[turb]/SCALE:+.2e}")
    print(f"  1. separation  turb/calm vol ratio: {vol_state[turb]/vol_state[calm]:.2f}")
    A = hmm.transmat_
    dur = 1 / (1 - np.diag(A))
    print(f"  2. persistence  expected duration  calm {dur[calm]:.0f}d  turb {dur[turb]:.0f}d")

    p_turb = _causal_filter(r * SCALE, means, vars_, A, hmm.startprob_)[:, turb]

    # 3. does causal P(turbulent) predict forward realised vol?
    frv = np.full(len(r), np.nan)
    for t in range(len(r) - H):
        frv[t] = np.sqrt(np.mean(r[t + 1:t + 1 + H] ** 2))
    te = np.arange(split, len(r) - H)
    print(f"  3. predicts vol  IC(P_turb, forward {H}d realised vol) [OOS]: "
          f"{_spearman(p_turb[te], frv[te]):+.3f}")

    # 4. do 95% one-day envelope breaches concentrate in the turbulent state?
    sig = np.sqrt(_ewma_vol_series(r, lam=0.94, burn=60))
    breach = np.abs(r) > 1.96 * sig
    hot = p_turb > 0.5
    idx = np.arange(split, len(r))
    br_turb = breach[idx][hot[idx]].mean(); br_calm = breach[idx][~hot[idx]].mean()
    print(f"  4. breaches      95%-band breach rate  turb {br_turb:.1%}  calm {br_calm:.1%}  "
          f"(nominal 5%)")

    # sizing angle: are turbulent-state forward returns worse (worth de-risking)?
    fwd = np.full(len(r), np.nan)
    for t in range(len(r) - H):
        fwd[t] = r[t + 1:t + 1 + H].sum()
    fr_turb = np.nanmean(fwd[idx][hot[idx]]); fr_calm = np.nanmean(fwd[idx][~hot[idx]])
    print(f"\n  sizing: mean forward {H}d return  turb {fr_turb:+.4f}  calm {fr_calm:+.4f}  "
          f"-> {'de-risk turbulent' if fr_turb < fr_calm else 'turbulent not worse'}")

    _plot(r, p_turb, split)
    print("\nsaved regime.png")


def _plot(r, p_turb, split):
    cum = np.cumsum(r)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(cum, color="black", lw=0.8)
    hot = p_turb > 0.5
    a1.fill_between(np.arange(len(r)), cum.min(), cum.max(), where=hot,
                    color="tab:red", alpha=0.15, step="mid", label="turbulent regime")
    a1.axvline(split, color="tab:blue", ls="--", lw=0.8, label="train/test split")
    a1.set_ylabel("cumulative log return (FTSE)"); a1.legend(loc="upper left")
    a1.set_title("Detected volatility regimes (causal HMM filter)")
    a2.plot(p_turb, color="tab:red", lw=0.5)
    a2.set_ylabel("P(turbulent | past)"); a2.set_xlabel("day"); a2.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig("regime.png", dpi=110)


if __name__ == "__main__":
    main()
