"""Tests for the PCA factor primitives."""
import numpy as np

from pca.factors import pca, standardize


def test_standardize_zero_mean_unit_var():
    rng = np.random.default_rng(0)
    R = rng.normal(3, 5, (500, 4))
    Z, mu, sd = standardize(R)
    assert np.allclose(Z.mean(0), 0, atol=1e-9)
    assert np.allclose(Z.std(0), 1, atol=1e-6)


def test_variance_fractions_sum_to_one():
    rng = np.random.default_rng(1)
    Z, *_ = standardize(rng.normal(0, 1, (800, 5)))
    var, loadings, scores = pca(Z)
    assert abs(var.sum() - 1.0) < 1e-9
    assert np.all(np.diff(var) <= 1e-12)          # descending


def test_pca_recovers_a_dominant_correlated_block():
    # two instruments driven by a common factor, two independent -> PC1 on the block
    rng = np.random.default_rng(2)
    n = 4000
    f = rng.normal(0, 1, n)
    R = np.column_stack([f + 0.1 * rng.normal(0, 1, n),
                         f + 0.1 * rng.normal(0, 1, n),
                         rng.normal(0, 1, n), rng.normal(0, 1, n)])
    Z, *_ = standardize(R)
    var, loadings, scores = pca(Z)
    assert var[0] > 0.35                            # the block dominates
    assert abs(loadings[0, 0]) > 0.6 and abs(loadings[0, 1]) > 0.6   # loads the block
    assert abs(loadings[0, 2]) < 0.3 and abs(loadings[0, 3]) < 0.3   # not the independents
