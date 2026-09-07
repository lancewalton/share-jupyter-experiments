"""Sliding-window extraction of z-normalised price shapes, with future labels.

A *window* is a length-L segment of the log-price path. We z-normalise each
window (subtract its mean, divide by its std) so only the SHAPE remains -- level
and scale are removed, which is what makes a 'triangle' a triangle regardless of
price or volatility. Each window 'completes' at the price index of its last
point (time t); its label is the strictly-future return over the next H steps.
"""
from __future__ import annotations

import numpy as np


def log_price(prices: np.ndarray) -> np.ndarray:
    """Log-price path from a positive price array."""
    return np.log(np.asarray(prices, dtype=float))


def extract_windows(
    logP: np.ndarray, L: int, stride: int = 1, znorm: bool = True, eps: float = 1e-9
):
    """Sliding windows of the log-price path.

    Returns ``(W, ends, starts)`` where ``W`` has shape ``(m, L)``, ``ends[k]`` is
    the price index of the last point of window ``k`` (the 'completion' time), and
    ``starts[k]`` its first. With ``znorm`` each row is shape-normalised.
    """
    logP = np.asarray(logP, dtype=float)
    starts = np.arange(0, len(logP) - L + 1, stride)
    W = np.stack([logP[s : s + L] for s in starts]).astype(float)
    if znorm:
        mu = W.mean(axis=1, keepdims=True)
        sd = W.std(axis=1, keepdims=True)
        W = (W - mu) / (sd + eps)
    ends = starts + L - 1
    return W, ends, starts


def resample_to(W: np.ndarray, P: int) -> np.ndarray:
    """Linearly resample each length-L window to a common length ``P``.

    Makes shapes from different time scales directly comparable: a 5-day and a
    60-day 'triangle' both become a P-point normalised shape, so they can share
    a common feature space / clustering. Call before z-normalising.
    """
    W = np.atleast_2d(np.asarray(W, dtype=float))
    Lin = W.shape[1]
    xold = np.linspace(0.0, 1.0, Lin)
    xnew = np.linspace(0.0, 1.0, P)
    return np.array([np.interp(xnew, xold, w) for w in W])


def forward_return(logP: np.ndarray, ends: np.ndarray, H: int) -> np.ndarray:
    """Strictly-future log return over ``H`` steps after each window completion.

    ``label[k] = logP[end+H] - logP[end]``; NaN where the future is unavailable.
    """
    logP = np.asarray(logP, dtype=float)
    ends = np.asarray(ends)
    out = np.full(len(ends), np.nan)
    ok = ends + H < len(logP)
    out[ok] = logP[ends[ok] + H] - logP[ends[ok]]
    return out


def temporal_split(ends: np.ndarray, frac: float = 0.6):
    """Boolean (train, test) masks splitting windows by completion time.

    Train = earliest ``frac`` of windows by completion index; test = the rest.
    A strict temporal split -- no future leaks into training pattern selection.
    """
    ends = np.asarray(ends)
    cut = np.quantile(ends, frac)
    train = ends <= cut
    return train, ~train
