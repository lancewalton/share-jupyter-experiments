"""Strength-preserving window representation for pattern matching.

The premise (see the sibling ``pattern-discovery`` project) is that z-normalising
each window destroys the very thing that carried signal there: *amplitude*. Here
we deliberately keep it, but express it in a scale-comparable unit.

A *window* is the length-``L`` segment of the LOG-price path. We turn it into a
feature vector by three operations, in order:

1. **anchor at the start** -- subtract the window's first value, so the path
   begins at 0. What remains is the cumulative log-return *from here forward*;
   the absolute price level (a pure nuisance) is gone, but the trend is NOT.
2. **scale by ``trailing_vol * sqrt(L)``** -- the expected diffusive range over
   ``L`` days, estimated from the ``lookback`` days *preceding* the window
   (strictly causal, never the window's own volatility). This sets the unit to
   "multiples of the range you'd expect at this scale", so a move is measured as
   its *strength* rather than divided away. A window ending at +2.0 in these
   units drifted twice its expected range. Crucially this touches only the unit,
   not the shape: the window's own slope (a local acceleration/deceleration
   relative to the global drift) is preserved for later components to catch.
3. **resample to ``P`` points** -- linear interpolation to a common length, so
   windows of different ``L`` share one feature space.

We do NOT mean-centre across the dataset anywhere: the whole point of the gate
test is to let the first extracted component absorb the mean drift ramp.
"""
from __future__ import annotations

import numpy as np


def log_price(prices: np.ndarray) -> np.ndarray:
    """Log-price path from a positive price array."""
    return np.log(np.asarray(prices, dtype=float))


def _resample_rows(W: np.ndarray, P: int) -> np.ndarray:
    """Linearly resample each row to length ``P`` on a common [0,1] grid."""
    W = np.atleast_2d(np.asarray(W, dtype=float))
    xold = np.linspace(0.0, 1.0, W.shape[1])
    xnew = np.linspace(0.0, 1.0, P)
    return np.array([np.interp(xnew, xold, w) for w in W])


def trailing_vol(logP: np.ndarray, starts: np.ndarray, lookback: int) -> np.ndarray:
    """Std of daily log-returns over the ``lookback`` days *before* each start.

    Strictly causal: for a window starting at index ``s`` it uses returns from
    ``[s-lookback, s)`` only, so it can never see inside (or after) the window.
    Returns NaN where there is not enough preceding history.
    """
    logP = np.asarray(logP, dtype=float)
    dr = np.diff(logP)  # daily log-returns; dr[i] spans logP[i]->logP[i+1]
    out = np.full(len(starts), np.nan)
    for k, s in enumerate(np.asarray(starts)):
        lo = s - lookback
        if lo < 0:
            continue
        seg = dr[lo:s]  # returns strictly before index s
        if len(seg) >= 2:
            out[k] = seg.std()
    return out


def forward_return_std(
    logP: np.ndarray, ends: np.ndarray, H: int, lookback: int = 60
) -> np.ndarray:
    """Vol-standardised strictly-future return over ``H`` steps after each window.

    ``(logP[end+H] - logP[end]) / (trailing_vol_at_end * sqrt(H))`` where the
    trailing vol uses the ``lookback`` returns ending at ``end`` -- all known at
    prediction time, so strictly causal. Standardising puts every stock/regime
    on one scale, which is what a *pooled* cross-sectional IC needs; the label
    then lives in the same expected-range units as the window features. NaN
    where the future or the vol estimate is unavailable.
    """
    logP = np.asarray(logP, dtype=float)
    ends = np.asarray(ends)
    out = np.full(len(ends), np.nan)
    tv = trailing_vol(logP, ends, lookback)  # returns in [end-lookback, end)
    ok = (ends + H < len(logP)) & np.isfinite(tv) & (tv > 0)
    raw = logP[ends[ok] + H] - logP[ends[ok]]
    out[ok] = raw / (tv[ok] * np.sqrt(H))
    return out


def vol_scaled_windows(
    logP: np.ndarray,
    L: int,
    P: int = 32,
    stride: int = 5,
    lookback: int = 60,
    eps: float = 1e-9,
):
    """Anchor / vol-scale / resample every length-``L`` window of ``logP``.

    Returns ``(W, ends, starts, drift)`` keeping only windows with a valid,
    positive trailing volatility:

      * ``W``      -- ``(m, P)`` strength-preserving feature rows (see module doc)
      * ``ends``   -- price index of each window's last point (completion time)
      * ``starts`` -- price index of each window's first point
      * ``drift``  -- realised net log-return across the window, in the SAME
                      vol units as ``W`` (this equals ``W[:, -1]``); a per-window
                      scalar "how far did it travel, in expected-range units".
    """
    logP = np.asarray(logP, dtype=float)
    starts = np.arange(0, len(logP) - L + 1, stride)
    if len(starts) == 0:
        return (np.empty((0, P)), np.empty(0, int), np.empty(0, int), np.empty(0))
    raw = np.stack([logP[s : s + L] for s in starts]).astype(float)
    anchored = raw - raw[:, :1]  # start at 0; level removed, trend kept
    tv = trailing_vol(logP, starts, lookback)
    scale = tv * np.sqrt(L)  # expected diffusive range over L days
    ok = np.isfinite(scale) & (scale > eps)
    anchored = anchored[ok] / scale[ok, None]
    starts = starts[ok]
    ends = starts + L - 1
    W = _resample_rows(anchored, P)
    drift = W[:, -1].copy()
    return W, ends, starts, drift
