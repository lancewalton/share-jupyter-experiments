"""PCA of the cross-asset return panel -- deriving and scoring the segments.

Standardise each instrument (z-score) and decompose the correlation structure by
SVD. Components are the market's *segments*: PC1 is typically a risk-on/off factor,
later PCs the rates / dollar / idiosyncratic factors. Loadings say which
instruments define a segment; scores say how strongly it was active each day.
"""
from __future__ import annotations

import numpy as np


def standardize(R: np.ndarray, mu=None, sd=None):
    """Z-score columns; returns (Z, mu, sd). Pass mu/sd to apply a fixed scaling."""
    R = np.asarray(R, float)
    mu = R.mean(0) if mu is None else mu
    sd = (R.std(0) if sd is None else sd) + 1e-12
    return (R - mu) / sd, mu, sd


def pca(Z: np.ndarray):
    """SVD of standardised returns.

    Returns ``(var, loadings, scores)``: ``var[i]`` fraction of variance in
    component i; ``loadings`` (k, p) each row a unit eigenvector over instruments;
    ``scores`` (n, k) the daily activation of each component. Signs are oriented
    so the largest-magnitude loading in each component is positive.
    """
    Z = np.asarray(Z, float)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    loadings = Vt.copy()
    scores = U * S
    for i in range(len(loadings)):
        j = np.argmax(np.abs(loadings[i]))
        if loadings[i, j] < 0:
            loadings[i] *= -1
            scores[:, i] *= -1
    return var, loadings, scores
