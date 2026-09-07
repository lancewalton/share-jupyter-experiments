import numpy as np

from trendlines import lower_hull_from_min, support_lines, resistance_lines


def test_collinear_rising_lows_keep_only_endpoints():
    # Arrange: lows rising in a straight line after the global minimum.
    lows = np.array([1.0, 2.0, 3.0])

    # Act
    vertices = lower_hull_from_min(lows)

    # Assert: interior collinear point is dropped.
    assert vertices == [0, 2]


def test_point_below_chord_is_kept_and_left_of_min_excluded():
    # Arrange: global min at index 1; index 2 dips below the 1->3 chord.
    lows = np.array([5.0, 1.0, 2.0, 4.0])

    # Act
    vertices = lower_hull_from_min(lows)

    # Assert: bar 0 (left of the minimum) is excluded; the dip is a vertex.
    assert vertices == [1, 2, 3]


def test_support_lines_chain_with_increasing_positive_gradient():
    # Arrange: hull vertices at 1, 2, 3 -> two chained segments.
    lows = np.array([5.0, 1.0, 2.0, 4.0])

    # Act
    lines = support_lines(lows)

    # Assert: segments chain end-to-start, slopes positive and strictly rising.
    assert [(ln.a, ln.b) for ln in lines] == [(1, 2), (2, 3)]
    assert lines[0].slope == 1.0
    assert lines[1].slope == 2.0


def test_resistance_lines_chain_with_decreasing_negative_gradient():
    # Arrange: global max at index 1; index 2 pokes above the 1->3 chord.
    highs = np.array([1.0, 4.0, 3.0, 0.0])

    # Act
    lines = resistance_lines(highs)

    # Assert: segments chain end-to-start, slopes negative and strictly falling.
    assert [(ln.a, ln.b) for ln in lines] == [(1, 2), (2, 3)]
    assert lines[0].slope == -1.0
    assert lines[1].slope == -3.0


def test_no_low_falls_below_any_support_ray():
    # The defining README constraint: from each support line's anchor to the end
    # of the series, no low may sit below the ray it generates.
    rng = np.random.default_rng(20260906)
    tol = 1e-9
    for _ in range(200):
        lows = rng.normal(size=rng.integers(5, 60)).cumsum()
        for ln in support_lines(lows):
            xs = np.arange(ln.a, len(lows))
            ray = ln.ya + ln.slope * (xs - ln.a)
            assert np.all(lows[ln.a:] >= ray - tol)


def test_no_high_rises_above_any_resistance_ray():
    rng = np.random.default_rng(20260906)
    tol = 1e-9
    for _ in range(200):
        highs = rng.normal(size=rng.integers(5, 60)).cumsum()
        for ln in resistance_lines(highs):
            xs = np.arange(ln.a, len(highs))
            ray = ln.ya + ln.slope * (xs - ln.a)
            assert np.all(highs[ln.a:] <= ray + tol)
