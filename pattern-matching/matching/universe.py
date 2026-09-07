"""Load the equity universe and build matched surrogates for null tests.

Shared by the phase scripts so they all see the same filtered universe as the
sibling ``pattern-discovery`` project (MIN_ROWS=1500, glitch filter at 0.6).
"""
from __future__ import annotations

import glob

import numpy as np

from mc.data import load_close
from mc.returns import log_returns

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
MIN_ROWS, GLITCH = 1500, 0.6


def load_universe(min_rows: int = MIN_ROWS, glitch: float = GLITCH):
    """Return a list of ``(name, logP, dates)`` for each usable series."""
    out = []
    for pattern in DIRS:
        for path in sorted(glob.glob(pattern)):
            try:
                s = load_close(path)
            except Exception:
                continue
            r = log_returns(s.to_numpy())
            if len(r) < min_rows or np.abs(r).max() > glitch:
                continue
            name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            out.append((name, np.log(s.to_numpy()), s.index.to_numpy()))
    return out


INDEX_TICKERS = {"SPX", "FTSE", "UKX", "NDX", "DJI", "GSPC", "VIX", "SX5E"}


def _bad_prints(logP: np.ndarray, thr: float = 0.15, resid: float = 0.08) -> int:
    """Count spike-and-reversal events -- a large move that ~cancels next day, the
    signature of a single bad price (real moves do not reverse next day)."""
    r = np.diff(np.asarray(logP, float))
    return int(sum(1 for t in range(len(r) - 1)
                   if abs(r[t]) > thr and abs(r[t] + r[t + 1]) < resid))


def _iso_steps(logP: np.ndarray, thr: float = 0.20, calm: float = 0.10) -> int:
    """Count isolated large non-reversing steps -- the signature of an unadjusted
    split / spin-off / special dividend (a lone big jump with calm on both sides).
    Over-flags: a real takeover pop looks the same, so this is a conservative cut."""
    r = np.diff(np.asarray(logP, float))
    return int(sum(1 for t in range(1, len(r) - 1)
                   if abs(r[t]) > thr and abs(r[t - 1]) < calm and abs(r[t + 1]) < calm))


def clean_universe(max_bad: int = 1, max_steps: int | None = None,
                   min_rows: int = MIN_ROWS, glitch: float = GLITCH):
    """``load_universe`` minus index tickers, series with > ``max_bad`` bad prints,
    and (if ``max_steps`` set) series with > ``max_steps`` isolated split-like steps.
    The corruption-audited universe for robustness checks."""
    out, dropped = [], []
    for name, logP, dates in load_universe(min_rows, glitch):
        bad = (name.upper() in INDEX_TICKERS or _bad_prints(logP) > max_bad
               or (max_steps is not None and _iso_steps(logP) > max_steps))
        (dropped if bad else out).append(name if bad else (name, logP, dates))
    return out, dropped


def iid_surrogate(logP: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian random walk matching this series' drift and volatility.

    Destroys ALL temporal structure but keeps the first two moments of the
    returns -- a pure random walk. If real components match this surrogate's,
    the components are artefacts of drift+diffusion, not the market.
    """
    r = np.diff(np.asarray(logP, float))
    sim = rng.normal(r.mean(), r.std(), size=len(r))
    return np.concatenate([[logP[0]], logP[0] + np.cumsum(sim)])


def shuffle_surrogate(logP: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute the real returns: keeps the exact return distribution, destroys
    order. A stricter null than iid (matches every moment, not just two)."""
    r = np.diff(np.asarray(logP, float))
    sim = rng.permutation(r)
    return np.concatenate([[logP[0]], logP[0] + np.cumsum(sim)])
