import pandas as pd

from cleaning import repair_bad_ticks


def _bar(o, h, l, c):
    return {"open": o, "high": h, "low": l, "close": c}


def test_absurd_low_wick_is_repaired_to_body_floor():
    # Arrange: middle bar has a 4.7 low while trading around 461 (a bad tick).
    df = pd.DataFrame(
        [_bar(460, 470, 460, 466), _bar(461, 471, 4.7, 461), _bar(460, 469, 460, 466)]
    )

    # Act
    out = repair_bad_ticks(df)

    # Assert: the spurious low becomes the bar's own body floor; nothing else moves.
    assert out.loc[1, "low"] == 461.0
    assert out.loc[0, "low"] == 460.0
    assert out.loc[2, "low"] == 460.0


def test_absurd_high_wick_is_repaired_and_wide_real_bar_kept():
    # Arrange: bar 0 has a 10000 high spike; bar 1 is wide but plausible.
    df = pd.DataFrame([_bar(100, 10000, 99, 101), _bar(100, 130, 70, 120)])

    # Act
    out = repair_bad_ticks(df)

    # Assert: the spike is clamped to the body; the wide-but-real bar is untouched.
    assert out.loc[0, "high"] == 101.0
    assert out.loc[1, "high"] == 130.0
    assert out.loc[1, "low"] == 70.0
