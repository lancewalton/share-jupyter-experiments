"""Volatility-matched (FHS) surrogates: the null for the structure detector.

The detector asks whether real price windows are distinguishable from a fake that
already contains everything we understand -- growth and the volatility envelope --
so that any distinguishing feature must be *directional path structure* beyond
volatility.

The fake is built by Filtered Historical Simulation:
  1. demean the returns (keep the drift aside);
  2. divide by a causal EWMA volatility, giving standardised residuals;
  3. SHUFFLE those residuals -- destroying any order / directional memory;
  4. re-impose the same volatility sequence and add the drift back.

So the surrogate keeps: the drift, the volatility clustering (same calm/turbulent
stretches, in the same places), and the residual distribution (fat tails). It
destroys: any structure in the *order* of moves -- momentum, reversion, a biased
exponential decay, rise/fall asymmetry. If a classifier can still tell real from
this, the difference is directional structure, not volatility.

Caveat the detector must check: if the EWMA leaves residual volatility clustering
in the standardised residuals, shuffling removes that too, and the classifier
could key on it (a second-moment tell). ``resid_abs_autocorr`` measures how much
is left, so a positive result can be attributed to direction, not leftover vol.
"""
from __future__ import annotations

import numpy as np


def ewma_vol(x: np.ndarray, halflife: float = 20.0, eps: float = 1e-8) -> np.ndarray:
    """Causal EWMA volatility of ``x`` (same length), warm-started on the mean sq."""
    x = np.asarray(x, dtype=float)
    lam = 0.5 ** (1.0 / halflife)
    acc = float(np.mean(x[: min(len(x), 20)] ** 2)) + eps
    out = np.empty(len(x))
    for i, xi in enumerate(x):
        acc = lam * acc + (1 - lam) * xi * xi
        out[i] = np.sqrt(acc)
    return out


def fhs_surrogate(logP: np.ndarray, rng: np.random.Generator,
                  halflife: float = 20.0) -> np.ndarray:
    """Vol-matched surrogate log-price path (see module docstring)."""
    logP = np.asarray(logP, dtype=float)
    r = np.diff(logP)
    m = r.mean()
    d = r - m
    sigma = ewma_vol(d, halflife)
    z = d / sigma
    z_shuf = rng.permutation(z)
    r_star = m + sigma * z_shuf
    return np.concatenate([[logP[0]], logP[0] + np.cumsum(r_star)])


def sign_flip_surrogate(logP: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomise only the SIGN of each return, keeping magnitudes in place.

    This preserves the volatility path *exactly* -- |return| sequence, clustering,
    fat tails, and hence the trailing-vol scaling are byte-identical to real --
    while destroying all directional structure (momentum, reversion, biased
    decay, up/down asymmetry). It is the clean first-moment null: if a classifier
    tells real from this, the difference can only be directional, since every
    second-moment quantity is unchanged.
    """
    logP = np.asarray(logP, dtype=float)
    r = np.diff(logP)
    m = r.mean()
    d = r - m
    d_star = rng.choice([-1.0, 1.0], size=len(d)) * np.abs(d)
    d_star *= d.std() / (d_star.std() + 1e-12)        # match variance exactly
    return np.concatenate([[logP[0]], logP[0] + np.cumsum(m + d_star)])


def resid_abs_autocorr(logP: np.ndarray, halflife: float = 20.0, lags=(1, 5)):
    """Autocorrelation of |standardised residuals| -- leftover vol clustering.

    Near zero means the EWMA removed the volatility structure, so a real-vs-
    surrogate classifier cannot be keying on second-moment clustering.
    """
    r = np.diff(np.asarray(logP, dtype=float))
    d = r - r.mean()
    z = np.abs(d / ewma_vol(d, halflife))
    z = z - z.mean()
    denom = np.sum(z * z) + 1e-12
    return {lag: float(np.sum(z[:-lag] * z[lag:]) / denom) for lag in lags}
