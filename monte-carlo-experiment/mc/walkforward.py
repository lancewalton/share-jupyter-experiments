"""Strict walk-forward evaluation of a forecaster's calibration.

For each origin ``t`` we build the predictive quantiles from history ``r[:t+1]``
only, then score them against the realised future ``r[t+1 : t+1+T]``. No future
information leaks into any forecast.

Overlapping-window caveat: consecutive origins share most of their future
window, so the per-origin scores are strongly dependent. ``stride`` thins the
origins to reduce this; even so, reported means are more precise than their
naive standard errors suggest. We report the origin count so effective sample
size can be reasoned about downstream.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .returns import realised_cumulative
from .scoring import crps_from_quantiles, interval_hit, pit_value

# Dense level grid for CRPS / PIT; central nominal levels tested separately.
DENSE_LEVELS = np.round(np.arange(0.005, 0.9951, 0.005), 4)
NOMINAL_INTERVALS = (0.50, 0.75, 0.90, 0.95, 0.99)


@dataclass
class WalkForwardResult:
    n_origins: int
    horizon: int
    mean_crps: np.ndarray  # (T,) mean CRPS per horizon
    coverage: dict  # nominal central prob -> (T,) empirical hit rate
    pit: np.ndarray  # (n_origins, T) PIT values at each horizon
    levels: np.ndarray

    def coverage_at(self, h: int) -> dict:
        """Nominal -> empirical coverage at horizon ``h`` (1-based)."""
        return {p: float(v[h - 1]) for p, v in self.coverage.items()}


def _origins(n_returns: int, T: int, min_history: int, stride: int) -> range:
    # need r[:t+1] with t+1 >= min_history, and r[t+1 : t+1+T] to exist.
    first = min_history - 1
    last = n_returns - T - 1
    return range(first, last + 1, stride)


def evaluate(
    r: np.ndarray,
    forecaster,
    T: int,
    min_history: int,
    stride: int = 1,
    levels: np.ndarray = DENSE_LEVELS,
    intervals=NOMINAL_INTERVALS,
) -> WalkForwardResult:
    """Run walk-forward scoring of ``forecaster`` over log-return series ``r``."""
    r = np.asarray(r, dtype=float)
    origins = list(_origins(len(r), T, min_history, stride))
    if not origins:
        raise ValueError("no valid origins; check min_history/T/stride vs data length")

    lev = np.asarray(levels, dtype=float)
    # Precompute interval bound indices within the dense grid.
    bounds = {}
    for p in intervals:
        lo, hi = (1 - p) / 2, 1 - (1 - p) / 2
        bounds[p] = (int(np.argmin(np.abs(lev - lo))), int(np.argmin(np.abs(lev - hi))))

    crps_acc = np.zeros(T)
    cover_acc = {p: np.zeros(T) for p in intervals}
    pit = np.empty((len(origins), T))

    for k, t in enumerate(origins):
        y = realised_cumulative(r, t, T)  # (T,)
        q = forecaster(r[: t + 1], T, lev)  # (L, T)
        crps_acc += crps_from_quantiles(q, y, lev)
        for p, (i_lo, i_hi) in bounds.items():
            cover_acc[p] += interval_hit(q[i_lo], q[i_hi], y)
        for h in range(T):
            pit[k, h] = pit_value(q[:, h], y[h], lev)

    n = len(origins)
    return WalkForwardResult(
        n_origins=n,
        horizon=T,
        mean_crps=crps_acc / n,
        coverage={p: v / n for p, v in cover_acc.items()},
        pit=pit,
        levels=lev,
    )
