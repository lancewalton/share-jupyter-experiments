"""Tests for the synthetic injection generator."""
import numpy as np

from matching.synth import biased_exp_decay, make_universe


def test_shape_anchored_and_unit():
    g = biased_exp_decay(50, k=3.0)
    assert g[0] == 0.0
    assert np.isclose(np.max(np.abs(g)), 1.0)
    # monotone decay (all steps same sign)
    assert np.all(np.diff(g) < 0)


def test_injection_produces_the_pattern_shape():
    # difference between an injected universe and its noise-identical A=0 twin,
    # over a pattern window, must equal A*sigma*sqrt(L)*g (up to the onset sign).
    L, A, sigma = 80, 1.0, 0.012
    base, _ = make_universe(0, n_series=1, A=0.0, sigma=sigma, L=L, rho=0.0)
    inj, onsets = make_universe(0, n_series=1, A=A, sigma=sigma, L=L, rho=0.0)
    e, s = onsets[0][0]
    diff = (inj[0][1] - base[0][1])          # logP difference = pure injection
    window_diff = diff[e - L + 1: e + 1] - diff[e - L + 1]  # anchored
    g = biased_exp_decay(L)
    expected = s * A * sigma * np.sqrt(L) * g
    assert np.allclose(window_diff, expected, atol=1e-9)


def test_rho_biases_forward_return():
    # with rho>0, the mean forward move after a +sign onset exceeds that after a
    # -sign onset (the predictive coupling is real).
    L, H = 80, 20
    inj, onsets = make_universe(1, n_series=1, A=1.0, rho=0.4, L=L, H=H)
    logP = inj[0][1]
    pos = [logP[e + H] - logP[e] for e, s in onsets[0] if s > 0]
    neg = [logP[e + H] - logP[e] for e, s in onsets[0] if s < 0]
    assert np.mean(pos) > np.mean(neg)
