import numpy as np

from channel import Channel, fit_channel


def test_fit_channel_recovers_a_perfect_line():
    # Arrange: exact line ln y = 1 + 0.02 x over 100 bars.
    x = np.arange(100)
    logy = 1.0 + 0.02 * x

    # Act
    ch = fit_channel(logy)

    # Assert: slope/intercept recovered, no residual, perfect fit.
    assert np.isclose(ch.gradient, 0.02)
    assert np.isclose(ch.intercept, 1.0)
    assert np.isclose(ch.sigma, 0.0, atol=1e-9)
    assert np.isclose(ch.r2, 1.0)
    assert ch.n == 100


def test_channel_bands_sit_k_sigma_around_the_mid_line_in_price_space():
    # Arrange: flat log-line at 0 with residual std 0.1 (band = exp(+/-k*sigma)).
    ch = Channel(gradient=0.0, intercept=0.0, sigma=0.1, r2=0.9, n=50)

    # Act / Assert: at x=0 mid=exp(0)=1; upper/lower = exp(+/-2*0.1).
    assert np.isclose(ch.mid(0), 1.0)
    assert np.isclose(ch.upper(0, k=2.0), np.exp(0.2))
    assert np.isclose(ch.lower(0, k=2.0), np.exp(-0.2))


def test_fit_channel_r2_drops_with_noise_but_slope_survives():
    rng = np.random.default_rng(0)
    x = np.arange(300)
    logy = 0.5 + 0.01 * x + rng.normal(0, 0.05, size=300)

    ch = fit_channel(logy)

    assert 0.005 < ch.gradient < 0.015     # ~0.01 recovered
    assert 0.5 < ch.r2 < 1.0               # real but imperfect fit
    assert ch.sigma > 0
