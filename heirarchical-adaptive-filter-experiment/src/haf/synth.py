"""Synthetic ground-truth volatility rig.

A latent log-volatility m_t moves in alternating CALM (slow drift) and TURBULENT
(fast drift) segments whose length SHRINKS over time -- the rate of change itself
accelerates. Returns are r_t = exp(m_t) * z_t. Because m_t is known, we know the
true local rate of change |dm/dt|, so we can ask whether a level's step size
actually tracks the changing dynamics -- something real data cannot tell us.
"""
import numpy as np


def synth_vol_series(n=8000, seed=7):
    rng = np.random.default_rng(seed)
    m = np.zeros(n)
    fast = np.zeros(n, bool)
    m[0] = np.log(0.01)
    t, turbulent = 1, False
    while t < n:
        base_len = max(20, int(400 * (1 - 0.7 * t / n)))
        length = rng.integers(base_len // 2, base_len + 1)
        drift_std = 0.05 if turbulent else 0.005
        for _ in range(length):
            if t >= n:
                break
            m[t] = np.clip(m[t - 1] + rng.normal(0, drift_std),
                           np.log(0.002), np.log(0.15))
            fast[t] = turbulent
            t += 1
        turbulent = not turbulent
    r = np.exp(m) * rng.standard_normal(n)
    return r, m, fast


def true_rate_of_change(m, window=20):
    dm = np.abs(np.diff(m, prepend=m[0]))
    return np.convolve(dm, np.ones(window) / window, mode="same")
