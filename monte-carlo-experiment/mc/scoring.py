"""Proper scoring rules and calibration diagnostics, defined on quantiles.

Everything here operates on predictive *quantiles* so that analytic baselines
and simulation-based forecasters are scored on identical footing. CRPS is
obtained from the quantile (pinball) representation:

    CRPS = 2 * integral_0^1 pinball_tau d(tau)  ~=  2 * mean(pinball over grid)

which needs no Monte-Carlo ensemble.
"""
from __future__ import annotations

import numpy as np


def pinball_loss(q_pred: np.ndarray, y_true: np.ndarray, level: float) -> np.ndarray:
    """Quantile (pinball) loss at quantile ``level`` in (0, 1).

    Elementwise: (y - q) * level      if y >= q
                 (q - y) * (1 - level) otherwise.
    Lower is better; the loss is minimised in expectation by the true
    ``level``-quantile.
    """
    q_pred = np.asarray(q_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    diff = y_true - q_pred
    return np.where(diff >= 0, level * diff, (level - 1.0) * diff)


def crps_from_quantiles(
    q_grid: np.ndarray, y_true: np.ndarray, levels: np.ndarray
) -> np.ndarray:
    """Approximate CRPS from a dense quantile grid.

    ``q_grid`` has shape ``(len(levels), ...)`` giving predictive quantiles at
    each ``level`` for one or more targets; ``y_true`` broadcasts against the
    trailing axes. Returns CRPS per target (the leading level axis is
    integrated out). Uses the mean of pinball losses over the (assumed roughly
    uniform) ``levels`` grid times 2.
    """
    levels = np.asarray(levels, dtype=float)
    losses = np.stack(
        [pinball_loss(q_grid[i], y_true, lv) for i, lv in enumerate(levels)]
    )
    return 2.0 * losses.mean(axis=0)


def interval_hit(
    q_lower: np.ndarray, q_upper: np.ndarray, y_true: np.ndarray
) -> np.ndarray:
    """Boolean: does ``y_true`` fall within the central interval [lower, upper]?"""
    y_true = np.asarray(y_true, dtype=float)
    return (y_true >= q_lower) & (y_true <= q_upper)


def pit_value(q_grid: np.ndarray, y_true: float, levels: np.ndarray) -> float:
    """Probability Integral Transform: the predictive CDF evaluated at ``y_true``.

    Estimated by interpolating the (monotone) mapping level -> quantile.
    A well-calibrated forecaster yields PIT values ~ Uniform(0, 1) across
    many origins. ``q_grid`` is 1-D (quantiles at each level for one target).
    """
    q_grid = np.asarray(q_grid, dtype=float)
    levels = np.asarray(levels, dtype=float)
    # np.interp needs increasing x (quantiles are non-decreasing in level).
    return float(np.interp(y_true, q_grid, levels, left=levels[0], right=levels[-1]))
