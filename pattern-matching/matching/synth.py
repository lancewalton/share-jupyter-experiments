"""Synthetic universes for the injection / recovery positive control.

We build series that are exactly what the premise supposes: a constant-growth
drift, plus matched diffusion noise, plus a *recurring* biased-exponential-decay
pattern layered on top. Optionally the pattern predicts the forward return, so we
can test the direction pipeline's power, not just reconstruction.

The point is a positive control: plant a known pattern of a chosen amplitude and
predictive strength, run the identical pipeline, and see (a) whether component 1
still comes out as the growth ramp, (b) whether the planted shape re-emerges as a
later component, and (c) at what effect size the direction test detects it. That
maps the method's detection floor for this specific shape -- turning "maybe you
missed it" into a measured number.
"""
from __future__ import annotations

import numpy as np


def biased_exp_decay(P: int, k: float = 3.0) -> np.ndarray:
    """Unit biased-exponential-decay path shape on P points, anchored at 0.

    ``g(u) = (exp(-k u) - 1)``, normalised to max |g| = 1. Drops fast then
    settles at a biased (non-zero) plateau -- the shape the user proposed.
    """
    u = np.linspace(0.0, 1.0, P)
    g = np.exp(-k * u) - 1.0
    return g / np.max(np.abs(g))


def make_universe(seed, n_series=122, length=3000, drift=0.0003, sigma=0.012,
                  L=80, H=20, A=0.0, rho=0.0, m=110, k=3.0):
    """Build a synthetic universe (list of ``(name, logP, dates)``) plus onsets.

    ``A``   -- pattern amplitude, in units of ``sigma*sqrt(L)`` (the expected
               diffusive range), so A=1 is a pattern as large as a typical window.
    ``rho`` -- predictive coupling: a pattern of sign s biases the next-H-day
               return by ``rho*s`` in std units (0 = present but non-predictive).
    ``m``   -- mean gap between pattern onsets (jittered); must exceed L so
               successive patterns do not overlap.
    Each onset gets a random sign, so patterns add no net drift and their
    covariance contribution is a clean rank-1 bump along the shape.
    """
    rng = np.random.default_rng(seed)
    g = biased_exp_decay(L, k)
    dg = np.diff(g)                       # L-1 daily increments
    inj = A * sigma * np.sqrt(L) * dg
    fwd_unit = rho * sigma * np.sqrt(H) / H
    dates = np.arange(length)
    series, onset_info = [], []
    for i in range(n_series):
        r = rng.normal(drift, sigma, length - 1)
        onsets = []
        e = L - 1 + int(rng.integers(0, m))          # first completion index
        while e + H < length:
            s = 1 if rng.random() < 0.5 else -1
            r[e - L + 1: e] += s * inj               # pattern in the window ending at e
            if rho:
                r[e: e + H] += s * fwd_unit          # predictive forward bias
            onsets.append((e, s))
            e += L + int(rng.integers(0, m))         # non-overlapping, jittered
        logP = np.concatenate([[0.0], np.cumsum(r)]) + np.log(100.0)
        series.append((f"synth{i:03d}", logP, dates.copy()))
        onset_info.append(onsets)
    return series, onset_info
