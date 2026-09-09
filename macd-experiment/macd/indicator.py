"""Standard MACD indicator and its causal long-only position signal.

MACD line = EMA_fast(close) - EMA_slow(close); signal line = EMA_signal(MACD line);
histogram = MACD line - signal line. Classic lengths are 12 / 26 / 9. The long-only
trade rule holds a position whenever the MACD line is above its signal line.
"""
from __future__ import annotations

import numpy as np
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


def macd_cross_state(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """Raw (unshifted) long signal: MACD line above its signal line."""
    line, sig, _ = macd(close, fast=fast, slow=slow, signal=signal)
    return line > sig


def long_state(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """The long-only position to hold each day, lagged one day so it is causal.

    A day's position is decided from the crossover observed on the *previous*
    close, so it never depends on the same day's price.
    """
    return macd_cross_state(close, fast=fast, slow=slow, signal=signal).shift(
        1, fill_value=False
    )


def hysteresis_state(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Hold from when ``entry`` turns True until ``exit_`` turns False.

    Enables asymmetric long-only rules (e.g. enter on a low-price MACD cross, exit on
    a high-price one). When entry and exit are the same signal it reduces to that
    signal exactly.
    """
    e = entry.to_numpy(dtype=bool)
    x = exit_.to_numpy(dtype=bool)
    out = np.empty(len(e), dtype=bool)
    held = False
    for i in range(len(e)):
        held = x[i] if held else e[i]
        out[i] = held
    return pd.Series(out, index=entry.index)


def vol_gate(
    returns: pd.DataFrame, window: int = 20, q: float = 0.5, q_window: int = 252
) -> pd.DataFrame:
    """Regime gate: True when trailing realised vol clears its own trailing q-quantile.

    The premise (README) is that trending markets are more volatile than flat
    consolidation, so requiring above-threshold volatility should skip false crosses
    in quiet markets. The threshold is each name's own rolling q-quantile of vol, so
    the gate is point-in-time and self-scaling. Closed until a threshold exists.
    """
    vol = returns.rolling(window).std()
    thr = vol.rolling(q_window, min_periods=window).quantile(q)
    return (vol >= thr) & thr.notna()


def gated_long_state(
    close: pd.DataFrame,
    returns: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    vol_window: int = 20,
    vol_q: float = 0.5,
    q_window: int = 252,
) -> pd.DataFrame:
    """MACD long signal AND the volatility-regime gate, lagged one day (causal)."""
    desired = macd_cross_state(close, fast=fast, slow=slow, signal=signal)
    gate = vol_gate(returns, window=vol_window, q=vol_q, q_window=q_window)
    return (desired & gate).shift(1, fill_value=False)
