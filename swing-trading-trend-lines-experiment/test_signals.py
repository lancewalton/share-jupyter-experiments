import numpy as np
import pandas as pd

from signals import atr, count_touches, latest_signal, select_line
from trendlines import Line


def test_wilder_atr_matches_hand_computation():
    # Arrange: 5 bars with TRs [2, 3, 2, 3, 2] for period 3.
    df = pd.DataFrame(
        {
            "high": [10.0, 12, 11, 13, 12],
            "low": [8.0, 9, 9, 10, 10],
            "close": [9.0, 11, 10, 12, 11],
        }
    )

    # Act
    a = atr(df, period=3)

    # Assert: seed = mean(TR[0:3]) = 7/3; then Wilder recursion.
    assert np.isnan(a.iloc[0]) and np.isnan(a.iloc[1])
    assert np.isclose(a.iloc[2], 7 / 3)
    assert np.isclose(a.iloc[3], ((7 / 3) * 2 + 3) / 3)
    assert np.isclose(a.iloc[4], (a.iloc[3] * 2 + 2) / 3)


def test_count_touches_counts_separated_in_band_episodes():
    # Arrange: value returns to the ray three times, leaving the band between.
    value = np.array([10.0, 10, 5, 10, 10, 5, 10])
    ray = np.full(7, 10.0)
    band = np.full(7, 1.0)

    # Act / Assert: three distinct touch episodes.
    assert count_touches(value, ray, band) == 3


def test_count_touches_treats_one_continuous_run_as_a_single_touch():
    value = np.array([10.0, 10, 10, 5, 5])
    ray = np.full(5, 10.0)
    band = np.full(5, 1.0)

    assert count_touches(value, ray, band) == 1


def test_select_line_takes_most_recent_line_meeting_min_span():
    lines = [Line(0, 30, 0, 0), Line(30, 35, 0, 0), Line(35, 40, 0, 0)]

    # Default (0) keeps the last, steepest line; a span floor skips short recents.
    assert select_line(lines, 0) is lines[-1]
    assert select_line(lines, 10) is lines[0]
    assert select_line(lines, 100) is None
    assert select_line([], 10) is None


def test_no_signal_when_no_resistance_structure_exists():
    # Arrange: a clean uptrend -> global max is the last bar -> no resistance line.
    n = 40
    base = np.linspace(100, 200, n)
    df = pd.DataFrame(
        {"open": base, "high": base + 1, "low": base - 1, "close": base}
    )

    # Act
    sig = latest_signal(df, k=0.5, atr_period=14, min_touches=3)

    # Assert
    assert sig is None


def _triangle(sig_high: float, sig_low: float, sig_close: float) -> pd.DataFrame:
    """Descending resistance (3 touches) over a slowly rising support, then a
    final bar whose high/low is supplied to trigger (or not) a breakout."""
    n = 30
    x = np.arange(n)
    ray_hi = 100 * 0.99**x
    high = ray_hi * 0.6
    high[[0, 14, 29]] = ray_hi[[0, 14, 29]]  # three touches on the resistance line
    low = 30 * 1.002**x
    close = (high + low) / 2
    return pd.DataFrame(
        {
            "open": np.r_[close, sig_close],
            "high": np.r_[high, sig_high],
            "low": np.r_[low, sig_low],
            "close": np.r_[close, sig_close],
        }
    )


def test_long_signal_on_breakout_of_qualified_resistance():
    # Arrange: final high (leftmost resistance ~74) clears it decisively.
    df = _triangle(sig_high=115.0, sig_low=60.0, sig_close=90.0)

    # Act
    sig = latest_signal(df, k=0.5, atr_period=14, min_touches=3)

    # Assert
    assert sig is not None
    assert sig.direction == "LONG"
    assert sig.touches >= 3
    assert (sig.action_line.a, sig.action_line.b) == (0, 29)
    assert sig.entry_price == 90.0


def test_no_signal_when_qualified_line_is_not_broken():
    # Arrange: identical structure, but the final high stays under the line+band.
    df = _triangle(sig_high=72.0, sig_low=60.0, sig_close=70.0)

    # Act
    sig = latest_signal(df, k=0.5, atr_period=14, min_touches=3)

    # Assert
    assert sig is None
