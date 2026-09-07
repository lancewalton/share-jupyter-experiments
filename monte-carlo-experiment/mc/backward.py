"""Backward simulation reframed as a per-origin trust signal.

The forward forecaster resamples the last-N returns under an IID assumption.
Because IID resampling is time-symmetric, the *same* drift-zero quantiles
describe the distribution of the B-day cumulative return whether we look
forward or backward. So we can ask: where does the *actual* recent B-day
trajectory (the ordered returns that led up to the origin) sit within that
ensemble?

Under genuine exchangeability the actual backward path is a central draw
(PIT ~ Uniform). If it is extreme, either returns are serially dependent
within the window or the recent volatility regime differs from the window
average -- both make the forward IID forecast untrustworthy at this origin.

``backward_distrust`` returns a scalar in [0, 1): the mean two-sided tail
level of the actual backward path across horizons 1..B (0 = perfectly
central, ->1 = extreme).
"""
from __future__ import annotations

import numpy as np

from .scoring import pit_value


def backward_realised(r: np.ndarray, t: int, B: int) -> np.ndarray:
    """Cumulative of the past ``B`` returns ending at origin ``t``.

    Element ``h-1`` = r[t] + r[t-1] + ... + r[t-h+1] (the h-day trajectory
    that led up to the origin). Requires ``t - B + 1 >= 0``.
    """
    if t - B + 1 < 0:
        raise ValueError(f"insufficient past data at origin {t} for backward horizon {B}")
    past = r[t - B + 1 : t + 1][::-1]  # [r[t], r[t-1], ..., r[t-B+1]]
    return np.cumsum(past)


def _mean_tail_level(q_grid: np.ndarray, path: np.ndarray, levels: np.ndarray) -> float:
    """Mean over horizons of the two-sided tail level 2*|PIT - 0.5|."""
    B = len(path)
    e = np.empty(B)
    for h in range(B):
        pit = pit_value(q_grid[:, h], float(path[h]), levels)
        e[h] = 2.0 * abs(pit - 0.5)
    return float(e.mean())


def backward_distrust(
    q_grid: np.ndarray, r: np.ndarray, t: int, B: int, levels: np.ndarray
) -> float:
    """Distrust score: extremeness of the actual backward path vs the ensemble."""
    b = backward_realised(r, t, B)
    return _mean_tail_level(q_grid, b, levels)


def forward_miscalibration(
    q_grid: np.ndarray, y_forward: np.ndarray, levels: np.ndarray
) -> float:
    """Same extremeness measure applied to the realised *forward* path.

    High values mean the forward outcome landed in the tails of the predictive
    distribution -- i.e. the forecast mis-covered at this origin. This is the
    quantity the distrust score is meant to predict.
    """
    return _mean_tail_level(q_grid, y_forward, levels)
