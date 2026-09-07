"""Tests for the backward-simulation trust signal."""
import numpy as np

from mc.backward import backward_realised, backward_distrust
from mc.forecasters import bootstrap_iid
from mc.walkforward import DENSE_LEVELS


def test_backward_realised_indexing():
    r = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # origin t=4, B=3 -> r[4]+.. = 0.5 ; 0.5+0.4 ; 0.5+0.4+0.3
    out = backward_realised(r, t=4, B=3)
    assert np.allclose(out, [0.5, 0.9, 1.2])


def test_backward_realised_raises_without_past():
    r = np.arange(5.0)
    try:
        backward_realised(r, t=1, B=3)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_distrust_averages_near_half_over_many_homogeneous_origins():
    # For a homogeneous IID window the score is a noisy per-path quantity whose
    # mean over many origins sits near 0.5 (E[2|U-0.5|]), well below the ceiling.
    rng = np.random.default_rng(10)
    r = rng.normal(scale=0.01, size=2000)
    f = bootstrap_iid(N=500, n_sims=8_000, seed=1)
    scores = []
    for t in range(600, 1600, 25):
        q = f(r[: t + 1], T=10, levels=DENSE_LEVELS)
        scores.append(backward_distrust(q, r, t, B=10, levels=DENSE_LEVELS))
    assert 0.4 < np.mean(scores) < 0.65


def test_distrust_high_when_recent_regime_differs():
    # Calm window with a recent strong trend/turbulence -> extreme backward path.
    rng = np.random.default_rng(11)
    calm = rng.normal(scale=0.003, size=490)
    recent = np.full(10, 0.03)  # a sharp sustained rally, unlike the window
    r = np.concatenate([calm, recent])
    t = len(r) - 1
    q = bootstrap_iid(N=500, n_sims=20_000, seed=1)(r[: t + 1], T=10, levels=DENSE_LEVELS)
    score = backward_distrust(q, r, t, B=10, levels=DENSE_LEVELS)
    assert score > 0.9
