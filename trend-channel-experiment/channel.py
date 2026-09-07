"""Regression trend channel: a rising OLS line on ln(price) with parallel bands.

A channel over a window is the least-squares line through ln(price) plus/minus a
multiple of the residual standard deviation. It is parallel by construction, its
gradient is the trend, and its width (band separation) is a volatility measure.
Everything is causal: fit on a trailing window, then trade forward.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Channel:
    gradient: float    # ln-price per bar
    intercept: float   # ln-price at x = 0 (window start)
    sigma: float       # residual standard deviation (ln)
    r2: float
    n: int

    def mid(self, x: float) -> float:
        return float(np.exp(self.intercept + self.gradient * x))

    def upper(self, x: float, k: float = 2.0) -> float:
        return float(np.exp(self.intercept + self.gradient * x + k * self.sigma))

    def lower(self, x: float, k: float = 2.0) -> float:
        return float(np.exp(self.intercept + self.gradient * x - k * self.sigma))

    def annual_gradient(self, bars_per_year: int = 252) -> float:
        """Compound annual growth implied by the ln-slope."""
        return float(np.exp(self.gradient * bars_per_year) - 1.0)


def fit_channel(logy: np.ndarray) -> Channel:
    """OLS fit of ``logy`` against bar index 0..n-1."""
    n = len(logy)
    x = np.arange(n, dtype=float)
    g, c = np.polyfit(x, logy, 1)
    fitted = c + g * x
    resid = logy - fitted
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((logy - logy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    sigma = float(np.std(resid))
    return Channel(gradient=float(g), intercept=float(c), sigma=sigma, r2=r2, n=n)
