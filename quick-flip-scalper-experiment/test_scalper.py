"""Unit tests for the quick-flip scalper core logic."""
from scalper import (
    Candle, Params, Side, Pattern, Stage, Exit,
    is_shooting_star, is_hammer, is_bearish_engulf, is_bullish_engulf,
    bar_exit, stop_entry_fill, simulate_day,
)

P = Params()


# ---- candle geometry ------------------------------------------------------ #
def test_shooting_star_true():
    # bearish, long upper wick, ~no lower wick
    c = Candle(open=100.0, high=110.0, low=99.8, close=99.9)
    assert is_shooting_star(c, P)


def test_shooting_star_rejects_bullish():
    c = Candle(open=99.9, high=110.0, low=99.8, close=100.0)  # bullish body
    assert not is_shooting_star(c, P)


def test_shooting_star_rejects_big_lower_wick():
    c = Candle(open=100.0, high=110.0, low=95.0, close=99.9)  # lower wick too big
    assert not is_shooting_star(c, P)


def test_hammer_is_mirror_of_star():
    c = Candle(open=100.1, high=100.2, low=90.0, close=100.2)  # bullish, long lower wick
    assert is_hammer(c, P)
    assert not is_shooting_star(c, P)


def test_bearish_engulf():
    c2 = Candle(open=100.0, high=101.0, low=99.9, close=100.9)   # bullish
    c3 = Candle(open=101.2, high=101.5, low=99.5, close=99.8)    # bearish, covers c2
    assert is_bearish_engulf(c2, c3)
    assert not is_bearish_engulf(c3, c2)


def test_bearish_engulf_needs_full_range_cover():
    c2 = Candle(open=100.0, high=101.6, low=99.9, close=100.9)   # c2.high above c3.high
    c3 = Candle(open=101.2, high=101.5, low=99.5, close=99.8)
    assert not is_bearish_engulf(c2, c3)


def test_bullish_engulf():
    c2 = Candle(open=100.9, high=101.0, low=99.9, close=100.0)   # bearish
    c3 = Candle(open=99.8, high=101.5, low=99.5, close=101.2)    # bullish, covers c2
    assert is_bullish_engulf(c2, c3)


# ---- conservative intrabar fills ------------------------------------------ #
def test_bar_exit_short_stop_first_when_bar_straddles():
    bar = Candle(open=100, high=105, low=90, close=95)  # hits both stop(104) & target(92)
    reason, px = bar_exit(bar, Side.SHORT, stop=104, target=92)
    assert reason is Exit.STOP and px == 104   # adverse assumed first


def test_bar_exit_short_target_only():
    bar = Candle(open=100, high=101, low=90, close=95)
    reason, px = bar_exit(bar, Side.SHORT, stop=104, target=92)
    assert reason is Exit.TARGET and px == 92


def test_bar_exit_none():
    bar = Candle(open=100, high=101, low=99, close=100)
    assert bar_exit(bar, Side.SHORT, stop=104, target=92) is None


def test_stop_entry_short_breakdown_vs_gap():
    # SHORT enters on price breaking DOWN through the level (level below price).
    # a bar whose low just reaches the level fills at the level
    touch = Candle(open=101, high=102, low=100, close=100.5)
    assert stop_entry_fill(touch, Side.SHORT, level=100) == 100
    # a bar that opens below the level (gap down) fills at that open
    gap = Candle(open=99, high=99.5, low=98, close=98.5)
    assert stop_entry_fill(gap, Side.SHORT, level=100) == 99
    # a bar that never drops to the level does not fill
    miss = Candle(open=101, high=102, low=100.5, close=101.5)
    assert stop_entry_fill(miss, Side.SHORT, level=100) is None


# ---- full-day simulation -------------------------------------------------- #
def _c1_bullish(hi=101.0, lo=100.0):
    # three bars whose max high = hi, min low = lo, closing up (bullish C1)
    return [
        Candle(100.0, 100.5, lo, 100.3),
        Candle(100.3, 100.8, 100.2, 100.6),
        Candle(100.6, hi, 100.5, 100.9),  # close(100.9) > open of bar0 (100.0)
    ]


def test_no_signal_when_price_stays_in_box():
    candles = _c1_bullish()
    # 15 quiet bars inside the range -> no reversal outside
    candles += [Candle(100.5, 100.9, 100.4, 100.6) for _ in range(15)]
    r = simulate_day(candles, 1.0, P)
    assert r.stage is Stage.NO_SIGNAL
    assert abs(r.liq_ratio - 1.0) < 1e-9   # C1 range 1.0 / atr 1.0


def test_shooting_star_short_hits_target():
    candles = _c1_bullish(hi=101.0, lo=100.0)  # box top 101, target 100
    # bar3: push above the box then a shooting star (bearish, long upper wick)
    candles.append(Candle(open=101.6, high=103.0, low=101.5, close=101.55))  # star, high 103
    # bar4: entry at this open (101.6 >= c1_hi). Then falls straight to target.
    candles.append(Candle(open=101.6, high=101.7, low=99.5, close=99.8))    # low 99.5 <= 100
    candles += [Candle(100.0, 100.1, 99.9, 100.0) for _ in range(13)]
    r = simulate_day(candles, 1.0, P)
    assert r.stage is Stage.TRADED
    assert r.side is Side.SHORT and r.pattern is Pattern.STAR
    assert r.entry_px == 101.6
    assert r.exit_reason is Exit.TARGET and r.exit_px == 100.0
    assert r.gross_ret > 0


def test_shooting_star_short_stopped_conservatively():
    candles = _c1_bullish(hi=101.0, lo=100.0)
    candles.append(Candle(open=101.6, high=103.0, low=101.5, close=101.55))  # star, stop=103
    # entry bar straddles both stop(103) and target(100) -> conservative STOP
    candles.append(Candle(open=101.6, high=103.5, low=99.0, close=100.0))
    candles += [Candle(100.0, 100.1, 99.9, 100.0) for _ in range(13)]
    r = simulate_day(candles, 1.0, P)
    assert r.stage is Stage.TRADED
    assert r.exit_reason is Exit.STOP and r.exit_px == 103.0
    assert r.gross_ret < 0


def test_engulfing_short_stop_fills_on_breakdown():
    candles = _c1_bullish(hi=101.0, lo=100.0)
    # C2 bullish above box (low >= c1_hi), C3 bearish engulfing C2
    c2 = Candle(open=101.2, high=102.0, low=101.1, close=101.9)   # level = c2.low 101.1
    c3 = Candle(open=101.9, high=102.1, low=100.8, close=101.0)   # engulfs, high 102.1
    candles += [c2, c3]
    # C4 opens above the level, then its low breaks down through 101.1 -> fill at level
    candles.append(Candle(open=101.3, high=101.4, low=101.0, close=101.05))
    candles.append(Candle(open=101.05, high=101.1, low=99.5, close=99.8))  # to target 100
    candles += [Candle(100.0, 100.1, 99.9, 100.0) for _ in range(11)]
    r = simulate_day(candles, 1.0, P)
    assert r.stage is Stage.TRADED
    assert r.pattern is Pattern.ENGULF
    assert r.entry_px == 101.1
    assert r.stop == 102.1
    assert r.exit_reason is Exit.TARGET


def test_engulfing_short_signalled_but_never_fills():
    candles = _c1_bullish(hi=101.0, lo=100.0)
    c2 = Candle(open=101.2, high=102.0, low=101.1, close=101.9)
    c3 = Candle(open=101.9, high=102.1, low=100.8, close=101.0)
    candles += [c2, c3]
    # price stays ABOVE the 101.1 breakdown level forever -> NO_FILL
    candles += [Candle(101.5, 101.9, 101.3, 101.6) for _ in range(11)]
    r = simulate_day(candles, 1.0, P)
    assert r.stage is Stage.NO_FILL
    assert r.pattern is Pattern.ENGULF


def test_time_stop_when_neither_hit():
    candles = _c1_bullish(hi=101.0, lo=100.0)
    candles.append(Candle(open=101.6, high=103.0, low=101.5, close=101.55))  # star
    candles.append(Candle(open=101.6, high=101.7, low=101.2, close=101.4))  # entry, drifts
    # drift sideways above target, below stop, until box end
    candles += [Candle(101.3, 101.5, 101.1, 101.3) for _ in range(20)]
    r = simulate_day(candles, 1.0, P)
    assert r.stage is Stage.TRADED
    assert r.exit_reason is Exit.TIME


def test_bearish_c1_takes_long_side():
    # bearish C1: closes down
    c1 = [
        Candle(101.0, 101.0, 100.5, 100.7),
        Candle(100.7, 100.8, 100.2, 100.4),
        Candle(100.4, 100.5, 100.0, 100.1),  # close 100.1 < open bar0 101.0
    ]
    # hammer below the box (bullish, long lower wick), then entry falls-in
    hammer = Candle(open=99.6, high=99.7, low=98.0, close=99.65)  # bullish, low 98 < c1_lo 100
    entry = Candle(open=99.6, high=101.2, low=99.5, close=101.0)  # to target c1_hi 101
    candles = c1 + [hammer, entry] + [Candle(101.0, 101.1, 100.9, 101.0) for _ in range(12)]
    r = simulate_day(candles, 1.0, P)
    assert r.stage is Stage.TRADED
    assert r.side is Side.LONG and r.pattern is Pattern.STAR
    assert r.exit_reason is Exit.TARGET
