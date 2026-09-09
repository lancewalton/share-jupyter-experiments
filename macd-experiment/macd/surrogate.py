"""Surrogate returns for the MACD permutation null.

Permuting each name's daily returns in time destroys the serial structure that a
trend-follower exploits while preserving the return distribution and the listing
window (NaN positions). If MACD looks as good on the surrogate as on the real
series, its apparent edge is a fishing artefact, not a real trend signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def permute_within_columns(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        vals = out[col].to_numpy().copy()
        valid = np.flatnonzero(~np.isnan(vals))
        if valid.size > 1:
            vals[valid] = rng.permutation(vals[valid])
            out[col] = vals
    return out
