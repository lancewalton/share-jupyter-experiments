"""Tests for the numpy autoencoder: gradient correctness + the non-linear win."""
import numpy as np

from matching.autoencoder import (
    Autoencoder1D,
    MLPAutoencoder,
    activations,
    deflationary_fit,
    final_residual,
)


def test_mlp_gradients_match_finite_difference():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (6, 5))
    ae = MLPAutoencoder(P=5, enc_hidden=[7, 4], code_dim=3, l2=1e-3, seed=2)
    _, (gW, gb) = ae.loss_and_grads(X)
    eps = 1e-6
    for li in range(len(ae.W)):
        for arr, grad in ((ae.W[li], gW[li]), (ae.b[li], gb[li])):
            it = np.nditer(arr, flags=["multi_index"]); checked = 0
            while not it.finished and checked < 4:
                i = it.multi_index; orig = arr[i]
                arr[i] = orig + eps; lp, _ = ae.loss_and_grads(X)
                arr[i] = orig - eps; lm, _ = ae.loss_and_grads(X)
                arr[i] = orig
                assert abs((lp - lm) / (2 * eps) - grad[i]) < 1e-4
                checked += 1; it.iternext()


def test_mlp_encode_has_code_dim_and_is_bounded():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (20, 8))
    ae = MLPAutoencoder(P=8, enc_hidden=[16], code_dim=5, seed=0)
    Z = ae.encode(X)
    assert Z.shape == (20, 5)
    assert np.all(np.abs(Z) <= 1.0 + 1e-9)  # tanh bottleneck


def test_analytic_gradients_match_finite_difference():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (7, 6))
    ae = Autoencoder1D(P=6, h=4, l2=1e-3, seed=1)
    loss0, grads = ae.loss_and_grads(X)
    eps = 1e-6
    for name in ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"):
        M = getattr(ae, name)
        it = np.nditer(M, flags=["multi_index"])
        checked = 0
        while not it.finished and checked < 5:
            i = it.multi_index
            orig = M[i]
            M[i] = orig + eps; lp, _ = ae.loss_and_grads(X)
            M[i] = orig - eps; lm, _ = ae.loss_and_grads(X)
            M[i] = orig
            num = (lp - lm) / (2 * eps)
            assert abs(num - grads[name][i]) < 1e-4, (name, i, num, grads[name][i])
            checked += 1
            it.iternext()


def _dip_manifold(n=1500, P=32, seed=0):
    """Localised dips whose POSITION slides -- a curved 1-D manifold that a
    single linear PCA component cannot capture but one AE component should."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, P)
    pos = rng.uniform(0.2, 0.8, n)
    X = -np.exp(-((grid[None, :] - pos[:, None]) ** 2) / 0.03)
    return X, pos


def test_one_component_beats_pca_on_sliding_dip():
    X, pos = _dip_manifold()
    ae = deflationary_fit(X, k=1, h=24, l2=1e-6, epochs=250, seed=0)
    ae_mse = np.mean(np.sum(final_residual(ae, X) ** 2, axis=1))
    # PCA with one component (uncentred, to match the AE which has a bias term
    # so can absorb the mean): best rank-1 reconstruction via SVD.
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Xr = (U[:, :1] * S[:1]) @ Vt[:1]
    pca_mse = np.mean(np.sum((X - Xr) ** 2, axis=1))
    assert ae_mse < 0.5 * pca_mse  # non-linear component clearly wins
    # and the code recovers the latent position (monotone relationship)
    z = activations(ae, X)[:, 0]
    rank_corr = np.corrcoef(np.argsort(np.argsort(z)),
                            np.argsort(np.argsort(pos)))[0, 1]
    assert abs(rank_corr) > 0.9
