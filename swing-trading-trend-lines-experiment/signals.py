"""Trading-signal layer for the swing trend-line method.

Structure (support/resistance convex hulls) comes from ``trendlines``; this
module adds the volatility unit and the touch/breach logic that turn structure
into entry/exit signals. The single pre-registered parameter is ``k``: the
"closeness" band is ``k * ATR`` (14-period Wilder), evaluated in price space
against the exp() of the ln-price ray.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trendlines import Line, resistance_lines, support_lines


@dataclass(frozen=True)
class Signal:
    """An entry signalled by the latest bar breaking its action line."""

    direction: str  # "LONG" or "SHORT"
    action_line: Line
    safety_line: Line
    entry_price: float
    touches: int


def count_touches(value: np.ndarray, ray: np.ndarray, band: np.ndarray) -> int:
    """Number of distinct episodes where ``value`` sits within ``band`` of ``ray``.

    A touch episode is a maximal run of bars inside the band; the price must
    leave the band for a new touch to be counted (hysteresis).
    """
    in_band = np.abs(value - ray) <= band
    # Count rising edges: an in-band bar whose predecessor was out of band.
    starts = in_band & ~np.concatenate(([False], in_band[:-1]))
    return int(starts.sum())


def select_line(lines: list[Line], min_span: int) -> Line | None:
    """Most recent hull line spanning at least ``min_span`` bars, else None.

    With ``min_span`` 0 this is just the last (steepest, most recent) line;
    a positive floor steps back to a longer, less-steep, more-established line.
    """
    for line in reversed(lines):
        if line.b - line.a >= min_span:
            return line
    return None


def _ray_prices(line: Line, xs: np.ndarray) -> np.ndarray:
    return np.exp(line.value_at(xs))


def _qualified_touches(line: Line, values: np.ndarray, band: np.ndarray) -> int:
    """Touch count of ``values`` against ``line``'s ray over [line.a, end]."""
    xs = np.arange(line.a, len(values))
    return count_touches(values[line.a:], _ray_prices(line, xs), band[line.a:])


def latest_signal(df: pd.DataFrame, k: float = 0.5, atr_period: int = 14,
                  min_touches: int = 3, min_span: int = 0,
                  atr_values: np.ndarray | None = None) -> Signal | None:
    """Signal implied by the latest bar, or None.

    Structure is built on all bars except the latest (the signal bar). A LONG is
    signalled when the latest HIGH clears the last resistance line by more than
    the k*ATR band and that line has >= ``min_touches`` touches; SHORT mirrors it
    with the last support line. Both a support and a resistance line must exist
    (one is the action line, the other the safety line).
    """
    if len(df) < atr_period + 2:
        return None

    structure = df.iloc[:-1]
    latest = df.iloc[-1]
    supports = support_lines(np.log(structure["low"].to_numpy()))
    resistances = resistance_lines(np.log(structure["high"].to_numpy()))
    up_line = select_line(resistances, min_span)
    down_line = select_line(supports, min_span)
    safety_up = select_line(supports, min_span)      # safety for a LONG
    safety_down = select_line(resistances, min_span)  # safety for a SHORT
    if up_line is None or down_line is None or safety_up is None or safety_down is None:
        return None

    av = atr(df, atr_period).to_numpy() if atr_values is None else atr_values
    band_full = pd.Series(k * av).bfill().to_numpy()
    band = band_full[:-1]
    band_latest = band_full[-1]
    x_latest = len(df) - 1
    highs = structure["high"].to_numpy()
    lows = structure["low"].to_numpy()

    up_touches = _qualified_touches(up_line, highs, band)
    if up_touches >= min_touches:
        ray = float(np.exp(up_line.value_at(x_latest)))
        if latest["high"] > ray + band_latest:
            return Signal("LONG", up_line, safety_up, float(latest["close"]), up_touches)

    down_touches = _qualified_touches(down_line, lows, band)
    if down_touches >= min_touches:
        ray = float(np.exp(down_line.value_at(x_latest)))
        if latest["low"] < ray - band_latest:
            return Signal("SHORT", down_line, safety_down, float(latest["close"]), down_touches)

    return None


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range, seeded with the SMA of the first ``period`` TRs."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    out = np.full(len(df), np.nan)
    if len(df) >= period:
        seed = true_range.iloc[:period].mean()
        out[period - 1] = seed
        tr = true_range.to_numpy()
        for i in range(period, len(df)):
            out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return pd.Series(out, index=df.index)
