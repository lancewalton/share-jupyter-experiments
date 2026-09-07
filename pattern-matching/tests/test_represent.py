"""Tests for the strength-preserving window representation."""
import numpy as np

from matching.represent import (
    forward_return_std,
    trailing_vol,
    vol_scaled_windows,
)


def test_trailing_vol_is_strictly_causal():
    # returns before index s must be the only ones used
    logP = np.cumsum(np.r_[0.0, np.arange(1, 21) * 0.001])  # 21 points, 20 returns
    starts = np.array([10])
    tv = trailing_vol(logP, starts, lookback=5)
    dr = np.diff(logP)
    assert np.isclose(tv[0], dr[5:10].std())  # exactly [s-lookback, s)


def test_trailing_vol_nan_without_history():
    logP = np.log(np.linspace(100, 110, 30))
    tv = trailing_vol(logP, np.array([3]), lookback=60)
    assert np.isnan(tv[0])


def test_windows_anchor_at_zero_and_resample_length():
    rng = np.random.default_rng(0)
    logP = np.cumsum(rng.normal(0.0005, 0.01, 500)) + np.log(100)
    W, ends, starts, drift = vol_scaled_windows(logP, L=60, P=32, stride=5, lookback=60)
    assert W.shape[1] == 32
    assert np.allclose(W[:, 0], 0.0)  # every window starts anchored at 0
    assert len(ends) == len(starts) == len(drift) == len(W)
    assert (ends == starts + 59).all()


def test_drift_equals_last_column_in_vol_units():
    rng = np.random.default_rng(1)
    logP = np.cumsum(rng.normal(0.0003, 0.012, 400)) + np.log(50)
    W, ends, starts, drift = vol_scaled_windows(logP, L=40, P=24, lookback=60)
    assert np.allclose(drift, W[:, -1])


def test_pure_drift_scales_to_expected_range_units():
    # constant daily step d, constant vol sigma over a clean series: the anchored
    # end value is d*L, scaled by sigma*sqrt(L) -> (d/sigma)*sqrt(L). Check sign
    # and that a steeper drift yields a proportionally larger end value.
    n = 400
    base = np.random.default_rng(2).normal(0, 0.01, n)
    logP_slow = np.cumsum(base + 0.002) + 5.0
    logP_fast = np.cumsum(base + 0.006) + 5.0
    _, _, _, d_slow = vol_scaled_windows(logP_slow, L=60, lookback=60)
    _, _, _, d_fast = vol_scaled_windows(logP_fast, L=60, lookback=60)
    assert np.nanmedian(d_fast) > np.nanmedian(d_slow) > 0


def test_forward_return_std_is_causal_and_standardised():
    rng = np.random.default_rng(5)
    logP = np.cumsum(rng.normal(0.0004, 0.011, 600)) + np.log(80)
    ends = np.array([200, 300, 400])
    H = 20
    f = forward_return_std(logP, ends, H=H, lookback=60)
    # matches an explicit causal construction
    for k, e in enumerate(ends):
        raw = logP[e + H] - logP[e]
        tv = np.diff(logP)[e - 60:e].std()
        assert np.isclose(f[k], raw / (tv * np.sqrt(H)))


def test_forward_return_std_nan_when_future_missing():
    logP = np.log(np.linspace(10, 20, 300))
    f = forward_return_std(logP, np.array([295]), H=20, lookback=60)
    assert np.isnan(f[0])
