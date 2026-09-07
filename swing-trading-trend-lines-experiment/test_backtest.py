import numpy as np
import pandas as pd

import backtest
from backtest import CostModel, simulate, trade_return, trail_stop
from signals import Signal
from trendlines import Line


def test_long_net_return_subtracts_spread_and_financing():
    # Arrange: 10% gross gain over 10 days; 10bps/side spread, 5%/yr financing.
    costs = CostModel(spread_bps_per_side=10, financing_annual=0.05)

    # Act
    net = trade_return("LONG", entry=100.0, exit=110.0, days=10, costs=costs)

    # Assert
    expected = 0.10 - (0.001 * 2) - 0.05 * 10 / 365
    assert np.isclose(net, expected)


def test_short_net_return_profits_when_price_falls():
    costs = CostModel(spread_bps_per_side=10, financing_annual=0.05)

    net = trade_return("SHORT", entry=100.0, exit=90.0, days=5, costs=costs)

    expected = 0.10 - (0.001 * 2) - 0.05 * 5 / 365
    assert np.isclose(net, expected)


def test_long_stop_ratchets_up_and_never_loosens():
    assert trail_stop("LONG", prev_stop=95.0, safety_today=97.0) == 97.0
    assert trail_stop("LONG", prev_stop=97.0, safety_today=96.0) == 97.0


def test_short_stop_ratchets_down_and_never_loosens():
    assert trail_stop("SHORT", prev_stop=105.0, safety_today=103.0) == 103.0
    assert trail_stop("SHORT", prev_stop=103.0, safety_today=104.0) == 103.0


def test_simulate_enters_next_open_and_exits_at_floor_stop(monkeypatch):
    # Arrange: 40 flat bars (ATR=1), a sharp low on bar 30 that hits the floor.
    n = 40
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    high = np.full(n, 100.5)
    low = np.full(n, 99.5)
    op = np.full(n, 100.0)
    close = np.full(n, 100.0)
    low[30] = 90.0        # spikes through the floor stop at 98
    op[30] = 99.0         # opens above the stop, so the fill is exactly the stop
    df = pd.DataFrame({"open": op, "high": high, "low": low, "close": close}, index=idx)

    dummy = Signal("LONG", Line(0, 1, 0.0, 0.0), Line(0, 1, 0.0, 0.0), 100.0, 3)
    calls = {"n": 0}

    def fake_signal(_df, **_kw):
        calls["n"] += 1
        return dummy if calls["n"] == 1 else None

    monkeypatch.setattr(backtest, "latest_signal", fake_signal)
    monkeypatch.setattr(backtest, "_current_safety", lambda *a, **k: None)

    # Act
    trades = simulate(df, stop_atr=2.0, warmup=20, costs=CostModel())

    # Assert: one LONG, entered at bar-21 open, stopped out at the 98 floor.
    assert len(trades) == 1
    tr = trades[0]
    assert tr.direction == "LONG"
    assert tr.entry_price == 100.0 and tr.exit_price == 98.0
    assert tr.days == 9
    assert np.isclose(tr.ret, trade_return("LONG", 100.0, 98.0, 9, CostModel()))
    # Entry geometry is recorded from the signal's lines (dummy: span 1, slope 0).
    assert tr.action_span == 1 and tr.safety_span == 1
    assert tr.action_slope == 0.0 and tr.separation == 0.0
    assert np.isnan(tr.rel_volume)  # no volume column -> nan, not a crash


def test_simulate_records_line_duration_fields(monkeypatch):
    # Arrange: flat highs at 100.5, lows at 99.5, one floor stop on bar 30.
    n = 40
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    high = np.full(n, 100.5)
    low = np.full(n, 99.5)
    op = np.full(n, 100.0)
    close = np.full(n, 100.0)
    low[30] = 90.0
    op[30] = 99.0
    df = pd.DataFrame({"open": op, "high": high, "low": low, "close": close}, index=idx)

    # Action line rides the highs (100.5), safety line rides the lows (99.5), so
    # every structure bar is a touch: one continuous episode starting at bar 0.
    action = Line(0, 2, float(np.log(100.5)), float(np.log(100.5)))
    safety = Line(0, 5, float(np.log(99.5)), float(np.log(99.5)))
    dummy = Signal("LONG", action, safety, 100.0, 3)
    calls = {"n": 0}

    def fake_signal(_df, **_kw):
        calls["n"] += 1
        return dummy if calls["n"] == 1 else None

    monkeypatch.setattr(backtest, "latest_signal", fake_signal)
    monkeypatch.setattr(backtest, "_current_safety", lambda *a, **k: None)

    # Act
    tr = simulate(df, stop_atr=2.0, warmup=20, costs=CostModel())[0]

    # Assert: endpoints and signal bar recorded, touches detected on the right series.
    assert (tr.action_a, tr.action_b) == (0, 2)
    assert (tr.safety_a, tr.safety_b) == (0, 5)
    assert tr.signal_bar == 20
    assert tr.action_touches == 1 and tr.action_touch_first == 0 and tr.action_touch_last == 0
    assert tr.safety_touches == 1 and tr.safety_touch_first == 0 and tr.safety_touch_last == 0


def test_pure_safety_exit_ignores_atr_floor(monkeypatch):
    # Arrange: normal lows (97) sit ABOVE where a 2*ATR floor (~98) would exit,
    # but the safety line is fixed at 95; only a drop through 95 should exit.
    n = 40
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    high = np.full(n, 100.5)
    low = np.full(n, 97.0)
    op = np.full(n, 100.0)
    close = np.full(n, 100.0)
    low[30] = 94.0        # breaches the safety line at 95
    op[30] = 96.0         # opens above it -> fills at the line
    df = pd.DataFrame({"open": op, "high": high, "low": low, "close": close}, index=idx)

    dummy = Signal("LONG", Line(0, 1, 0.0, 0.0), Line(0, 1, 0.0, 0.0), 100.0, 3)
    calls = {"n": 0}

    def fake_signal(_df, **_kw):
        calls["n"] += 1
        return dummy if calls["n"] == 1 else None

    monkeypatch.setattr(backtest, "latest_signal", fake_signal)
    monkeypatch.setattr(backtest, "_current_safety", lambda *a, **k: 95.0)

    # Act: pure safety-line exit (no ATR floor).
    trades = simulate(df, stop_atr=None, warmup=20, costs=CostModel())

    # Assert: one trade, held until the 95 breach on bar 30, not stopped at ~98.
    assert len(trades) == 1
    assert trades[0].exit_price == 95.0
    assert trades[0].exit_date == idx[30]
