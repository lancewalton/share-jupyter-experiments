"""Tests for the frame-level hysteresis and the retrenchment re-entry scan."""
import numpy as np
import pandas as pd

from macd.indicator import hysteresis_state, hysteresis_frame, retrench_entry


def _bools(mat):
    idx = pd.date_range("2020-01-01", periods=len(mat), freq="B")
    return pd.DataFrame(np.asarray(mat, dtype=bool), index=idx,
                        columns=list("AB")[: len(mat[0])])


def _series(vals):
    idx = pd.date_range("2020-01-01", periods=len(vals), freq="B")
    return pd.Series(np.asarray(vals, dtype=bool), index=idx)


def test_hysteresis_frame_matches_columnwise_hysteresis_state():
    entry = _bools([[0, 1], [1, 0], [0, 0], [1, 1], [1, 0]])
    exit_ = _bools([[1, 1], [1, 1], [1, 0], [0, 1], [1, 1]])
    out = hysteresis_frame(entry, exit_)
    for col in entry.columns:
        assert out[col].equals(hysteresis_state(entry[col], exit_[col]))


def test_retrench_enters_on_the_second_up_cross_within_the_window():
    # up-cross at t=2 and t=6 (down-cross at t=4 between them)
    cross = _series([0, 0, 1, 1, 0, 0, 1, 1, 0])
    entry = retrench_entry(cross, window=10)
    assert not entry.iloc[2]        # first up-cross is skipped
    assert entry.iloc[6]           # second up-cross is the entry
    assert entry.sum() == 1


def test_retrench_rearms_when_the_second_up_cross_is_too_late():
    cross = _series([0, 0, 1, 1, 0, 0, 1, 1, 0])
    entry = retrench_entry(cross, window=2)   # 6 - 2 = 4 > 2 -> no entry
    assert entry.sum() == 0


def test_retrench_takes_only_the_second_then_resets():
    # up-crosses at t=2, t=5, t=8 -> enter at 5 (2nd), t=8 is a fresh first -> no entry
    cross = _series([0, 0, 1, 1, 0, 1, 1, 0, 1])
    entry = retrench_entry(cross, window=10)
    assert entry.iloc[5] and entry.sum() == 1
