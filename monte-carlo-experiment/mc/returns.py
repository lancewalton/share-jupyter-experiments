"""Log returns and realised cumulative-return targets.

Indexing convention: ``r`` is the array of daily log returns, where ``r[i]``
is the return realised from day ``i`` to day ``i+1``. A forecast *origin*
``t`` means we know ``r[:t+1]`` (returns up to and including the step landing
on day ``t+1``... more precisely, all returns observed by the close we
forecast from). The realised cumulative target at horizon ``h`` is
``r[t+1] + ... + r[t+h]``.
"""
from __future__ import annotations

import numpy as np


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Daily log returns ln(P_t / P_{t-1}); length ``len(prices) - 1``."""
    prices = np.asarray(prices, dtype=float)
    return np.diff(np.log(prices))


def realised_cumulative(r: np.ndarray, t: int, T: int) -> np.ndarray:
    """Realised cumulative log return from origin ``t`` for horizons 1..T.

    Returns an array of length ``T`` where element ``h-1`` is the sum of the
    next ``h`` returns after origin ``t``. Requires ``t + T < len(r) + 1``,
    i.e. the future window ``r[t+1 : t+1+T]`` must exist.
    """
    future = r[t + 1 : t + 1 + T]
    if len(future) != T:
        raise ValueError(f"insufficient future data at origin {t} for horizon {T}")
    return np.cumsum(future)


def history_upto(r: np.ndarray, t: int) -> np.ndarray:
    """Return history available at origin ``t``: ``r[:t+1]``."""
    return r[: t + 1]
