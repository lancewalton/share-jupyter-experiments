"""Tests for low-for-long / high-for-short price sourcing: hysteresis + wick clamp."""
import numpy as np
import pandas as pd

from macd.indicator import hysteresis_state, long_state
from macd.data import clamp_wicks


def _series(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def _bools(values):
    return _series(values).astype(bool)


def test_hysteresis_enters_on_entry_and_holds_until_exit_drops():
    entry = _bools([0, 1, 0, 0, 1])
    exit_ = _bools([1, 1, 1, 0, 1])
    out = hysteresis_state(entry, exit_)
    # flat->enter at 1; hold while exit True; drop when exit False at 3; re-enter at 4
    assert list(out.astype(int)) == [0, 1, 1, 0, 1]


def test_hysteresis_reduces_to_the_plain_state_when_entry_equals_exit():
    s = _bools([0, 1, 1, 0, 1, 1, 0])
    assert hysteresis_state(s, s).equals(s)


def test_hysteresis_composes_with_macd_to_match_baseline_on_close():
    # entry-state == exit-state == MACD>signal must reproduce the ungated long_state
    px = _series(100 + np.cumsum(np.sin(np.linspace(0, 12, 120))))
    from macd.indicator import macd_cross_state
    s = macd_cross_state(px)
    assert hysteresis_state(s, s).shift(1, fill_value=False).equals(long_state(px))


def test_clamp_wicks_removes_gross_spikes_and_preserves_ordering():
    adj_close = _series([100.0, 100.0, 100.0])
    adj_low = _series([98.0, 5.0, 100.0])       # middle is a bad-tick spike
    adj_high = _series([101.0, 100.0, 500.0])   # last is a bad-tick spike
    lo, hi = clamp_wicks(adj_low, adj_high, adj_close)
    assert lo.iloc[0] == 98.0 and hi.iloc[0] == 101.0     # normal bars untouched
    assert lo.iloc[1] == 50.0                             # clamped up to 0.5*close
    assert hi.iloc[2] == 200.0                            # clamped down to 2*close
    assert (lo <= adj_close).all() and (hi >= adj_close).all()
