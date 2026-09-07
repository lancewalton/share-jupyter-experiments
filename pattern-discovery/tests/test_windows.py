"""Tests for window extraction, labelling and the temporal split."""
import numpy as np

from patterns.windows import (
    log_price,
    extract_windows,
    forward_return,
    temporal_split,
    resample_to,
)


def test_resample_preserves_shape_across_scales():
    # a straight ramp at two different lengths resamples to the same P-point shape
    short = np.linspace(0, 1, 6)[None, :]
    long = np.linspace(0, 1, 40)[None, :]
    rs, rl = resample_to(short, 16)[0], resample_to(long, 16)[0]
    assert np.allclose(rs, rl, atol=1e-9)
    # a V-shape resamples to a V of the target length
    v = np.array([[3.0, 2.0, 1.0, 2.0, 3.0]])
    out = resample_to(v, 9)[0]
    assert out.shape == (9,)
    assert out[0] == 3.0 and out[-1] == 3.0 and out.argmin() == 4


def test_extract_shapes_and_indices():
    logP = np.arange(10, dtype=float)  # straight ramp
    W, ends, starts = extract_windows(logP, L=4, stride=2, znorm=False)
    assert W.shape == (4, 4)                     # starts 0,2,4,6
    assert list(starts) == [0, 2, 4, 6]
    assert list(ends) == [3, 5, 7, 9]
    assert np.allclose(W[0], [0, 1, 2, 3])


def test_znorm_removes_level_and_scale():
    logP = np.array([100.0, 101.0, 102.0, 103.0])  # any affine ramp
    W, _, _ = extract_windows(logP, L=4, znorm=True)
    # z-normed straight ramp is fixed regardless of level/slope
    assert np.isclose(W[0].mean(), 0.0, atol=1e-6)
    assert np.isclose(W[0].std(), 1.0, atol=1e-6)
    W2, _, _ = extract_windows(np.array([0.0, 5.0, 10.0, 15.0]), L=4, znorm=True)
    assert np.allclose(W[0], W2[0])              # same SHAPE -> same window


def test_forward_return_is_strictly_future():
    logP = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5])
    _, ends, _ = extract_windows(logP, L=2, stride=1, znorm=False)
    # window ending at index e -> forward H=2 return = logP[e+2]-logP[e]
    f = forward_return(logP, ends, H=2)
    # ends = [1,2,3,4,5]; last two lack a full future -> NaN
    assert np.isclose(f[0], logP[3] - logP[1])
    assert np.isnan(f[-1]) and np.isnan(f[-2])


def test_temporal_split_no_overlap():
    ends = np.arange(100)
    tr, te = temporal_split(ends, frac=0.6)
    assert tr.sum() > 0 and te.sum() > 0
    assert not (tr & te).any()
    assert ends[tr].max() <= ends[te].min()      # strictly earlier
