"""Two-era process designed to REQUIRE an adaptive meta-rate (Level 3).

y_t = phi_t * y_{t-1} + noise, where phi_t is a random walk whose *drift speed*
switches between eras:
    era A: phi drifts slowly  -> optimal learning rate LOW,  optimal theta LOW
    era B: phi drifts fast     -> optimal learning rate HIGH, optimal theta HIGH
Optionally the eras alternate several times, so no single fixed learning rate
(and no single fixed theta) is optimal across the whole series. phi_t is the
ground truth: we know exactly when the dynamics speed up.
"""
import numpy as np


def two_era_ar(n=12000, n_eras=6, slow=0.002, fast=0.03, noise=0.3, seed=3):
    rng = np.random.default_rng(seed)
    era_len = n // n_eras
    phi = np.zeros(n)
    drift = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        era = (t // era_len) % 2          # 0 = slow, 1 = fast
        d = fast if era == 1 else slow
        drift[t] = d
        phi[t] = np.clip(phi[t - 1] + rng.normal(0, d), -0.95, 0.95)
        y[t] = phi[t] * y[t - 1] + rng.normal(0, noise)
    return y, phi, drift
