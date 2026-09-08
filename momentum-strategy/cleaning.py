"""Repair gross OHLC data errors (single-bar wick spikes) before analysis.

A daily bar whose low sits far below its own open/close body — or whose high
sits far above it — is a data error, not a real price. Such ticks are fatal to
a method that anchors trend lines on the global extreme, so we clamp the
offending wick back to the bar's own body. Real gaps and crashes (where the
body itself moves) are left untouched.
"""
from __future__ import annotations

import pandas as pd


def repair_bad_ticks(df: pd.DataFrame, max_wick_ratio: float = 0.5) -> pd.DataFrame:
    """Clamp implausible low/high wicks to the bar's own body.

    ``max_wick_ratio`` is the furthest a wick may extend from the body before it
    is treated as an error: a low below ``max_wick_ratio * min(open, close)`` or
    a high above ``max(open, close) / max_wick_ratio``.
    """
    out = df.copy()
    body_lo = out[["open", "close"]].min(axis=1)
    body_hi = out[["open", "close"]].max(axis=1)

    bad_low = out["low"] < max_wick_ratio * body_lo
    bad_high = out["high"] > body_hi / max_wick_ratio

    out.loc[bad_low, "low"] = body_lo[bad_low]
    out.loc[bad_high, "high"] = body_hi[bad_high]
    return out
