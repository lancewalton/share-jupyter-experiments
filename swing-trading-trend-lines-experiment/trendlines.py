"""Deterministic trend-line structure for swing trading.

The support/resistance lines described in the experiment README are exactly the
edges of a convex hull:

* Support lines = lower convex hull of the LOW prices, taken from the global
  minimum low rightward. Their gradients increase monotonically.
* Resistance lines = upper convex hull of the HIGH prices, taken from the global
  maximum high rightward. Their gradients decrease monotonically.

There are no free parameters in this layer; it is fully determined by the data.
Work in bar-index space (one unit per bar), which is the usual convention for
trend lines and side-steps calendar gaps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Line:
    """A trend line anchored at bars ``a`` and ``b`` with values ``ya``, ``yb``."""

    a: int
    b: int
    ya: float
    yb: float

    @property
    def slope(self) -> float:
        return (self.yb - self.ya) / (self.b - self.a)

    def value_at(self, x: float) -> float:
        """Value of the ray (line extended without bound) at bar index ``x``."""
        return self.ya + self.slope * (x - self.a)


def _lines_from_vertices(vertices: list[int], y: np.ndarray) -> list[Line]:
    return [
        Line(a, b, float(y[a]), float(y[b]))
        for a, b in zip(vertices, vertices[1:])
    ]


def support_lines(lows: np.ndarray) -> list[Line]:
    """Support trend lines (rising lower hull) from the global minimum rightward."""
    return _lines_from_vertices(lower_hull_from_min(lows), lows)


def upper_hull_from_max(y: np.ndarray) -> list[int]:
    """Indices of the upper convex hull vertices, from the global maximum rightward."""
    return lower_hull_from_min(-y)


def resistance_lines(highs: np.ndarray) -> list[Line]:
    """Resistance trend lines (falling upper hull) from the global maximum rightward."""
    return _lines_from_vertices(upper_hull_from_max(highs), highs)


def lower_hull_from_min(y: np.ndarray) -> list[int]:
    """Indices of the lower convex hull vertices, from the global minimum rightward."""
    start = int(np.argmin(y))
    hull: list[int] = []
    for i in range(start, len(y)):
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], i, y) <= 0:
            hull.pop()
        hull.append(i)
    return hull


def _cross(o: int, a: int, b: int, y: np.ndarray) -> float:
    """Cross product of oa x ob in (index, value) space; > 0 is a left turn."""
    return (a - o) * (y[b] - y[o]) - (y[a] - y[o]) * (b - o)
