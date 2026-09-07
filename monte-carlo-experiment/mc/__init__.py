"""Monte-Carlo share-price experiment.

Anchor: distributional *calibration* of drift-zero cumulative-log-return
envelopes, evaluated strictly walk-forward. See README.md and the project
memory for the staged research path.

Universal currency: a *forecaster* maps (return history up to an origin,
horizon T, quantile levels) -> predictive quantiles of cumulative log-return
for each horizon 1..T, centred on zero. All scoring is defined on quantiles.
"""
