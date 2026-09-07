"""Full trade simulator for the swing trend-line method (spread-bet wrapper).

Spread betting: returns are on notional as a percentage, there is no UK stamp
duty and no CGT, but each side pays a spread and open positions accrue daily
financing. One position at a time, per ticker, walked forward with no look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signals import atr, latest_signal, line_touch_bounds, select_line
from trendlines import Line, resistance_lines, support_lines


@dataclass(frozen=True)
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    days: int
    ret: float
    # Entry geometry (from the signal's action/safety lines), for the setup study.
    action_slope: float = 0.0   # ln-price per bar
    safety_slope: float = 0.0
    action_span: int = 0        # bars the action line spans (b - a)
    safety_span: int = 0
    separation: float = 0.0     # |action ray - safety ray| / entry, at the signal bar
    touches: int = 0
    rel_volume: float = float("nan")  # signal-bar volume / trailing 20-bar median
    # Line-duration study: raw endpoints and touch positions (bar indices) of the
    # action and safety lines at the breakout, for computing relative durations.
    signal_bar: int = -1
    action_a: int = -1
    action_b: int = -1
    safety_a: int = -1
    safety_b: int = -1
    action_touches: int = 0
    action_touch_first: int = -1
    action_touch_last: int = -1
    safety_touches: int = 0
    safety_touch_first: int = -1
    safety_touch_last: int = -1


def _line_duration(line: Line, series: np.ndarray, band: np.ndarray,
                   signal_bar: int) -> tuple[int, int, int, int, int]:
    """Endpoints (a, b) and touch bounds (count, first, last) of ``line`` against
    ``series`` over the structure bars [line.a, signal_bar)."""
    xs = np.arange(line.a, signal_bar)
    ray = np.exp(line.value_at(xs))
    n, first, last = line_touch_bounds(series[line.a:signal_bar], ray,
                                       band[line.a:signal_bar], line.a)
    return line.a, line.b, n, first, last


@dataclass(frozen=True)
class CostModel:
    spread_bps_per_side: float = 10.0   # basis points paid on entry and on exit
    financing_annual: float = 0.05      # annual financing on notional, charged daily


def trail_stop(direction: str, prev_stop: float, safety_today: float) -> float:
    """Ratchet the stop toward the safety line, never loosening it."""
    if direction == "LONG":
        return max(prev_stop, safety_today)
    return min(prev_stop, safety_today)


def trade_return(direction: str, entry: float, exit: float, days: int,
                 costs: CostModel) -> float:
    """Net fractional return on notional for one closed trade."""
    gross = (exit - entry) / entry
    if direction == "SHORT":
        gross = -gross
    spread = 2 * costs.spread_bps_per_side / 1e4
    financing = costs.financing_annual * days / 365
    return gross - spread - financing


def _current_safety(struct: pd.DataFrame, direction: str, x: int,
                    min_span: int = 0) -> float | None:
    """Price-space value at bar ``x`` of the safety line (opposite the trade),
    from structure built on ``struct`` (bars strictly before ``x``)."""
    if direction == "LONG":
        lines = support_lines(np.log(struct["low"].to_numpy()))
    else:
        lines = resistance_lines(np.log(struct["high"].to_numpy()))
    line = select_line(lines, min_span)
    return None if line is None else float(np.exp(line.value_at(x)))


def simulate(df: pd.DataFrame, k: float = 0.5, atr_period: int = 14,
             min_touches: int = 3, min_span: int = 0, stop_atr: float | None = 2.0,
             costs: CostModel | None = None, warmup: int = 60) -> list[Trade]:
    """Walk the ticker forward, one position at a time, and return closed trades.

    ``stop_atr`` sets the initial disaster-floor stop at ``entry -/+ stop_atr*ATR``;
    pass ``None`` for the author's pure exit — the safety line alone is the stop.
    """
    costs = costs or CostModel()
    a = atr(df, atr_period).to_numpy()
    band = pd.Series(k * a).bfill().to_numpy()
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    vols = df["volume"].to_numpy() if "volume" in df.columns else None
    n = len(df)

    trades: list[Trade] = []
    i = warmup
    while i < n - 1:
        sig = latest_signal(df.iloc[: i + 1], k=k, atr_period=atr_period,
                            min_touches=min_touches, min_span=min_span,
                            atr_values=a[: i + 1])
        if sig is None:
            i += 1
            continue

        direction = sig.direction
        entry_idx = i + 1
        entry_price = float(opens[entry_idx])
        if stop_atr is None:  # pure safety-line exit: no floor until the line sets it
            stop = float("-inf") if direction == "LONG" else float("inf")
        else:
            stop = (entry_price - stop_atr * a[i] if direction == "LONG"
                    else entry_price + stop_atr * a[i])

        exit_idx = exit_price = None
        for j in range(entry_idx, n):
            safety = _current_safety(df.iloc[:j], direction, j, min_span)
            if safety is not None:
                stop = trail_stop(direction, stop, safety)
            if direction == "LONG" and lows[j] <= stop:
                exit_idx, exit_price = j, min(float(opens[j]), stop)
                break
            if direction == "SHORT" and highs[j] >= stop:
                exit_idx, exit_price = j, max(float(opens[j]), stop)
                break
        if exit_idx is None:  # ran to the end still open: mark out at last close
            exit_idx, exit_price = n - 1, float(closes[-1])

        days = max(1, (df.index[exit_idx] - df.index[entry_idx]).days)
        al, sl = sig.action_line, sig.safety_line
        action_ray = float(np.exp(al.value_at(i)))
        safety_ray = float(np.exp(sl.value_at(i)))
        rel_volume = float("nan")
        if vols is not None and i >= 20:
            base = np.median(vols[i - 20:i])
            if np.isfinite(base) and base > 0 and np.isfinite(vols[i]):
                rel_volume = float(vols[i] / base)
        act_series, saf_series = (highs, lows) if direction == "LONG" else (lows, highs)
        aa, ab, at_n, at_f, at_l = _line_duration(al, act_series, band, i)
        sa, sb, st_n, st_f, st_l = _line_duration(sl, saf_series, band, i)
        trades.append(Trade(
            df.index[entry_idx], df.index[exit_idx], direction,
            entry_price, exit_price, days,
            trade_return(direction, entry_price, exit_price, days, costs),
            action_slope=al.slope, safety_slope=sl.slope,
            action_span=al.b - al.a, safety_span=sl.b - sl.a,
            separation=abs(action_ray - safety_ray) / entry_price,
            touches=sig.touches, rel_volume=rel_volume,
            signal_bar=i, action_a=aa, action_b=ab, safety_a=sa, safety_b=sb,
            action_touches=at_n, action_touch_first=at_f, action_touch_last=at_l,
            safety_touches=st_n, safety_touch_first=st_f, safety_touch_last=st_l,
        ))
        i = exit_idx + 1
    return trades
