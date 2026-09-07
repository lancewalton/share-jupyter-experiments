"""Quick Flip Scalper — core logic (pure, unit-testable).

An opening-range *fade* strategy on 5-minute candles:

1. C1 = first three 5-min bars (09:30-09:45 NY). Its high/low is the box.
2. Liquidity filter: C1 range >= threshold x prior-day 14-day ATR
   (ATR computed from daily bars aggregated from the 5-min data).
3. Within the 75-min box after C1, look for a reversal *outside* C1's range
   and fade back towards the far side of C1:
   - bullish C1  -> price grabs liquidity ABOVE C1.high, we look for a bearish
     reversal and go SHORT, targeting C1.low.
   - bearish C1  -> mirror image: LONG, targeting C1.high.
   Reversal types (bullish-C1 wording; inverse for bearish):
   - shooting star: bearish candle, long upper wick, tiny lower wick.
     Entry at the NEXT bar's open; stop at the star's high.
   - bearish engulfing: bullish C2 then bearish C3 whose range fully covers
     C2's. Entry is a sell-STOP at C2's low: from C4 (the bar after C3) onward,
     if a bar opens below C2's low we enter at its open, otherwise we enter when
     price breaks down through C2's low. Stop at C3's high.

Exits: take-profit at the far side of C1; once price retraces >= `trail_frac`
into C1's range the stop is moved to the near side of C1 (lock gains); hard
time-stop at the end of the box.

All intrabar fills are CONSERVATIVE: when a single 5-min bar straddles both the
stop and the target we assume the adverse level (the stop) filled first, because
the bar hides the true path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class Candle:
    open: float
    high: float
    low: float
    close: float

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def rng(self) -> float:
        return self.high - self.low

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def bullish(self) -> bool:
        return self.close > self.open


@dataclass(frozen=True)
class Params:
    # liquidity filter is applied in post-processing via liq_ratio; kept here
    # only for reference / a hard floor:
    liq_threshold: float = 0.0
    # shooting star / hammer geometry
    wick_body_mult: float = 2.0     # long wick >= mult x body
    wick_min_frac: float = 0.5      # long wick >= frac x range ("lots")
    opp_wick_max_frac: float = 0.10  # short wick <= frac x range ("very little")
    stop_mult: float = 1.0          # widen stop distance from entry by this factor
    # exits
    trail_frac: float = 0.5         # retrace this far into C1 -> lock stop to near side
    box_bars: int = 15             # bars after C1 that make up the 75-min box
    c1_bars: int = 3


class Side(Enum):
    SHORT = "short"   # fade a bullish C1
    LONG = "long"     # fade a bearish C1


class Pattern(Enum):
    STAR = "star"
    ENGULF = "engulf"


class Stage(Enum):
    NO_C1 = "no_c1"
    NO_ATR = "no_atr"
    NO_SIGNAL = "no_signal"
    NO_FILL = "no_fill"
    TRADED = "traded"


class Exit(Enum):
    TARGET = "target"
    STOP = "stop"
    TRAIL = "trail"      # stop that had been moved to lock gains
    TIME = "time"


# --------------------------------------------------------------------------- #
# Candle geometry
# --------------------------------------------------------------------------- #
def is_shooting_star(c: Candle, p: Params) -> bool:
    """Bearish rejection candle: long upper wick, tiny lower wick."""
    if c.bullish or c.rng <= 0:
        return False
    long_wick = c.upper_wick >= p.wick_body_mult * c.body and c.upper_wick >= p.wick_min_frac * c.rng
    tiny_other = c.lower_wick <= p.opp_wick_max_frac * c.rng
    return long_wick and tiny_other


def is_hammer(c: Candle, p: Params) -> bool:
    """Bullish rejection candle: long lower wick, tiny upper wick."""
    if not c.bullish or c.rng <= 0:
        return False
    long_wick = c.lower_wick >= p.wick_body_mult * c.body and c.lower_wick >= p.wick_min_frac * c.rng
    tiny_other = c.upper_wick <= p.opp_wick_max_frac * c.rng
    return long_wick and tiny_other


def is_bearish_engulf(c2: Candle, c3: Candle) -> bool:
    """C2 bullish, C3 bearish, C3's range fully covers C2's."""
    return c2.bullish and not c3.bullish and c3.high >= c2.high and c3.low <= c2.low


def is_bullish_engulf(c2: Candle, c3: Candle) -> bool:
    """C2 bearish, C3 bullish, C3's range fully covers C2's."""
    return (not c2.bullish) and c3.bullish and c3.high >= c2.high and c3.low <= c2.low


# --------------------------------------------------------------------------- #
# Intrabar fills (conservative)
# --------------------------------------------------------------------------- #
def bar_exit(bar: Candle, side: Side, stop: float, target: float) -> tuple[Exit, float] | None:
    """Does this bar hit stop or target? Adverse (stop) assumed first if both."""
    if side is Side.SHORT:
        hit_stop = bar.high >= stop
        hit_tgt = bar.low <= target
        if hit_stop:
            return (Exit.STOP, stop)
        if hit_tgt:
            return (Exit.TARGET, target)
    else:
        hit_stop = bar.low <= stop
        hit_tgt = bar.high >= target
        if hit_stop:
            return (Exit.STOP, stop)
        if hit_tgt:
            return (Exit.TARGET, target)
    return None


def stop_entry_fill(bar: Candle, side: Side, level: float) -> float | None:
    """Fill a resting entry STOP (breakdown/breakout). SHORT enters on price
    breaking DOWN through `level`; LONG on breaking UP through it. A bar that
    gaps through the level at its open fills at the open."""
    if side is Side.SHORT:
        if bar.open <= level:
            return bar.open
        if bar.low <= level:
            return level
    else:
        if bar.open >= level:
            return bar.open
        if bar.high >= level:
            return level
    return None


# --------------------------------------------------------------------------- #
# Result record
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    stage: Stage
    liq_ratio: float | None = None        # C1 range / ATR
    side: Side | None = None
    pattern: Pattern | None = None
    entry_i: int | None = None            # index within session
    entry_px: float | None = None
    stop: float | None = None
    target: float | None = None
    exit_i: int | None = None
    exit_px: float | None = None
    exit_reason: Exit | None = None
    gross_ret: float | None = None        # signed, fraction (short = (entry-exit)/entry)
    r_multiple: float | None = None       # gross_ret / initial risk fraction
    mae: float | None = None              # worst adverse excursion (fraction, +ve)
    mfe: float | None = None              # best favourable excursion (fraction, +ve)
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Per-day simulation
# --------------------------------------------------------------------------- #
def _c1_box(candles: list[Candle], p: Params):
    c1 = candles[: p.c1_bars]
    c1_hi = max(c.high for c in c1)
    c1_lo = min(c.low for c in c1)
    bullish = c1[-1].close > c1[0].open
    return c1_hi, c1_lo, bullish


def _signed_ret(side: Side, entry: float, px: float) -> float:
    return (entry - px) / entry if side is Side.SHORT else (px - entry) / entry


def simulate_day(candles: list[Candle], atr: float | None, p: Params) -> Result:
    """Simulate one name-day. `candles` is the full regular-session list of
    5-min bars (index 0 == 09:30). Liquidity thresholding is left to the caller
    via Result.liq_ratio; here we always search so the ratio is recorded."""
    if len(candles) < p.c1_bars + 2:
        return Result(Stage.NO_C1)
    if atr is None or atr <= 0:
        return Result(Stage.NO_ATR)

    c1_hi, c1_lo, bullish = _c1_box(candles, p)
    liq_ratio = (c1_hi - c1_lo) / atr
    side = Side.SHORT if bullish else Side.LONG
    target = c1_lo if side is Side.SHORT else c1_hi
    mid = 0.5 * (c1_hi + c1_lo)

    box_start = p.c1_bars
    box_end = min(len(candles) - 1, p.c1_bars + p.box_bars - 1)  # inclusive index

    # ---- find first valid reversal signal in the box ----------------------- #
    signal = _find_signal(candles, p, side, c1_hi, c1_lo, box_start, box_end)
    if signal is None:
        return Result(Stage.NO_SIGNAL, liq_ratio=liq_ratio)
    pattern, entry_i, entry_px, stop = signal
    if entry_i is None:  # engulfing limit never filled
        return Result(Stage.NO_FILL, liq_ratio=liq_ratio, side=side, pattern=pattern)

    # ---- walk the trade to exit ------------------------------------------- #
    return _run_trade(candles, p, side, pattern, entry_i, entry_px, stop, target,
                      c1_hi, c1_lo, mid, box_end, liq_ratio)


def _outside(side: Side, c: Candle, c1_hi: float, c1_lo: float) -> bool:
    return c.high > c1_hi if side is Side.SHORT else c.low < c1_lo


def _find_signal(candles, p, side, c1_hi, c1_lo, box_start, box_end):
    """Return (pattern, entry_i, entry_px, stop) for the first valid signal, or
    None. entry_i is None if a valid engulfing signal never fills its limit."""
    for i in range(box_start, box_end + 1):
        c = candles[i]
        # --- shooting star / hammer (needs a next bar to enter on) ---------- #
        star = is_shooting_star(c, p) if side is Side.SHORT else is_hammer(c, p)
        if star and _outside(side, c, c1_hi, c1_lo) and i + 1 <= box_end:
            entry_px = candles[i + 1].open
            entry_outside = entry_px >= c1_hi if side is Side.SHORT else entry_px <= c1_lo
            raw_stop = c.high if side is Side.SHORT else c.low
            stop = entry_px + p.stop_mult * (raw_stop - entry_px)
            if entry_outside and _valid(side, entry_px, stop, c1_hi, c1_lo):
                return (Pattern.STAR, i + 1, entry_px, stop)
        # --- engulfing (C2 = i-1, C3 = i; entry scanned from C4 = i+1) ------- #
        if i - 1 >= box_start:
            c2, c3 = candles[i - 1], c
            eng = is_bearish_engulf(c2, c3) if side is Side.SHORT else is_bullish_engulf(c2, c3)
            if eng and _outside(side, c3, c1_hi, c1_lo):
                level = c2.low if side is Side.SHORT else c2.high
                lvl_outside = level >= c1_hi if side is Side.SHORT else level <= c1_lo
                raw_stop = c3.high if side is Side.SHORT else c3.low
                if lvl_outside:
                    for j in range(i + 1, box_end + 1):   # from C4 onward
                        fill = stop_entry_fill(candles[j], side, level)
                        if fill is not None:
                            stop = fill + p.stop_mult * (raw_stop - fill)
                            if _valid(side, fill, stop, c1_hi, c1_lo):
                                return (Pattern.ENGULF, j, fill, stop)
                            return None      # degenerate fill (e.g. gapped past target)
                    return (Pattern.ENGULF, None, None, raw_stop)  # signalled, never filled
    return None


def _valid(side: Side, entry: float, stop: float, c1_hi: float, c1_lo: float) -> bool:
    """Stop on the correct side of entry, and target beyond entry."""
    if side is Side.SHORT:
        return stop > entry and entry > c1_lo   # target = c1_lo
    return stop < entry and entry < c1_hi        # target = c1_hi


def _run_trade(candles, p, side, pattern, entry_i, entry_px, stop, target,
               c1_hi, c1_lo, mid, box_end, liq_ratio) -> Result:
    near = c1_hi if side is Side.SHORT else c1_lo  # trail-to level
    cur_stop = stop
    trailed = False
    mae = mfe = 0.0

    for k in range(entry_i, box_end + 1):
        bar = candles[k]
        # update excursions
        adverse_px = bar.high if side is Side.SHORT else bar.low
        favour_px = bar.low if side is Side.SHORT else bar.high
        mae = max(mae, -_signed_ret(side, entry_px, adverse_px))
        mfe = max(mfe, _signed_ret(side, entry_px, favour_px))

        ex = bar_exit(bar, side, cur_stop, target)
        if ex is not None:
            reason, px = ex
            if reason is Exit.STOP and trailed:
                reason = Exit.TRAIL
            return _finish(side, pattern, entry_i, entry_px, stop, target,
                           k, px, reason, mae, mfe, liq_ratio)

        # activate trail based on THIS completed bar (applies from next bar) —
        # no intrabar lookahead
        if not trailed:
            reached = (bar.low <= mid) if side is Side.SHORT else (bar.high >= mid)
            if reached:
                trailed = True
                cur_stop = near

    # time stop at box end
    last = candles[box_end]
    return _finish(side, pattern, entry_i, entry_px, stop, target,
                   box_end, last.close, Exit.TIME, mae, mfe, liq_ratio)


def _finish(side, pattern, entry_i, entry_px, stop, target, exit_i, exit_px,
            reason, mae, mfe, liq_ratio) -> Result:
    gross = _signed_ret(side, entry_px, exit_px)
    risk = abs(stop - entry_px) / entry_px
    r = gross / risk if risk > 0 else 0.0
    return Result(
        stage=Stage.TRADED, liq_ratio=liq_ratio, side=side, pattern=pattern,
        entry_i=entry_i, entry_px=entry_px, stop=stop, target=target,
        exit_i=exit_i, exit_px=exit_px, exit_reason=reason,
        gross_ret=gross, r_multiple=r, mae=mae, mfe=mfe,
    )
