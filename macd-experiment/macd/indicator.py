"""Standard MACD indicator and its causal long-only position signal.

MACD line = EMA_fast(close) - EMA_slow(close); signal line = EMA_signal(MACD line);
histogram = MACD line - signal line. Classic lengths are 12 / 26 / 9. The long-only
trade rule holds a position whenever the MACD line is above its signal line.
"""
from __future__ import annotations

import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = _ema(close, fast) - _ema(close, slow)
    sig = _ema(line, signal)
    hist = line - sig
    return line, sig, hist


def long_state(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """The long-only position to hold each day, lagged one day so it is causal.

    A day's position is decided from the crossover observed on the *previous*
    close, so it never depends on the same day's price.
    """
    line, sig, _ = macd(close, fast=fast, slow=slow, signal=signal)
    desired = line > sig
    return desired.shift(1, fill_value=False)
