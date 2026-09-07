"""Task 3: vol-targeting + regime on the S&P 500 (higher-premium market).

Vol-targeting HURT on FTSE partly because the UK premium was weak over the period.
Does it help on the S&P 500 (strong premium, 2016-2026 via FRED)? And does the
market regime here look like FTSE (turbulent -> rebound) or Moreira-Muir
(turbulent -> negative)? Single index; ~10y (no 2008). Strictly this index only.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from mc.forecasters import _ewma_vol_series

ANN, SCALE, TARGET, MAXLEV, H = 252, 100.0, 0.15, 3.0, 20


def _sp500():
    df = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500")
    df.columns = ["Date", "Close"]
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return np.diff(np.log(df["Close"].to_numpy()))


def _stats(x):
    x = x[np.isfinite(x)]
    ar, av = x.mean() * ANN, x.std() * np.sqrt(ANN)
    g = np.cumsum(x); dd = float((g - np.maximum.accumulate(g)).min())
    return ar, av, (ar / av if av > 0 else 0.0), dd


def _filter(x, means, vars_, trans, start):
    n, k = len(x), len(means)
    lp = -0.5 * (np.log(2 * np.pi * vars_) + (x[:, None] - means) ** 2 / vars_)
    pdf = np.exp(lp - lp.max(axis=1, keepdims=True))
    a = np.empty((n, k)); a[0] = start * pdf[0]; a[0] /= a[0].sum()
    for t in range(1, n):
        a[t] = pdf[t] * (a[t - 1] @ trans); a[t] /= a[t].sum()
    return a


def main():
    r = _sp500(); r = r[np.isfinite(r)]
    print(f"S&P 500: {len(r)} daily returns\n")

    # --- vol-targeting ---
    sig_ann = np.sqrt(_ewma_vol_series(r, lam=0.94, burn=60)) * np.sqrt(ANN)
    w = np.nan_to_num(np.clip(TARGET / np.where(sig_ann > 0, sig_ann, np.nan), 0, MAXLEV))
    vm = w * r
    print(f"{'book':<22}{'ann.ret':>9}{'ann.vol':>9}{'Sharpe':>8}{'maxDD':>8}")
    for nm, x in [("buy & hold", r), ("vol-managed", vm)]:
        ar, av, sh, dd = _stats(x)
        print(f"{nm:<22}{ar:>+9.3f}{av:>9.3f}{sh:>+8.2f}{dd:>+8.2f}")

    # --- regime ---
    split = len(r) // 2
    hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=200,
                      random_state=0).fit((r[:split] * SCALE).reshape(-1, 1))
    turb = int(np.argmax(hmm.covars_.ravel()))
    vol_state = np.sqrt(hmm.covars_.ravel()) / SCALE * np.sqrt(ANN)
    P = _filter(r * SCALE, hmm.means_.ravel(), hmm.covars_.ravel(),
                hmm.transmat_, hmm.startprob_)[:, turb]
    fwd = np.full(len(r), np.nan)
    for t in range(len(r) - H):
        fwd[t] = r[t + 1:t + 1 + H].sum()
    frv = np.full(len(r), np.nan)
    for t in range(len(r) - H):
        frv[t] = np.sqrt(np.mean(r[t + 1:t + 1 + H] ** 2))
    te = np.arange(split, len(r) - H); hot = P[te] >= 0.5
    m = np.isfinite(frv[te])
    ic = np.corrcoef(np.argsort(np.argsort(P[te][m])),
                     np.argsort(np.argsort(frv[te][m])))[0, 1]
    print(f"\nregime: calm vol {vol_state[1-turb]:.1%}, turbulent vol {vol_state[turb]:.1%}"
          f"   P(turb) fwd-vol IC {ic:+.3f}")
    print(f"  forward-{H}d return  calm {np.nanmean(fwd[te][~hot]):+.4f}  "
          f"turbulent {np.nanmean(fwd[te][hot]):+.4f}  "
          f"-> {'de-risk helps' if np.nanmean(fwd[te][hot])<np.nanmean(fwd[te][~hot]) else 'turbulent rebounds (like FTSE)'}")

    # regime overlay (causal, lagged 1 day): cut exposure in turbulence
    wr = 1.0 - 0.7 * np.concatenate([[P[0]], P[:-1]])
    idx = np.arange(split, len(r))
    print("\nOOS Sharpe (second half):")
    for nm, x in [("buy & hold", r[idx]), ("vol-managed", vm[idx]),
                  ("regime-overlay", (wr * r)[idx]),
                  ("vol-managed + regime", (wr * vm)[idx])]:
        print(f"  {nm:<22}{_stats(x)[2]:+.2f}   maxDD {_stats(x)[3]:+.2f}")


if __name__ == "__main__":
    main()
