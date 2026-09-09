"""Walk-forward parameter selection — the honest version of 're-tuning until it works'.

Each candidate parameter set has a precomputed daily return series (its concentrated
book). At each rebalance we pick the set with the best trailing-window return and trade
it forward until the next rebalance, then roll. Selection uses only trailing data, so
the stitched forward series is genuinely out-of-sample; the adaptation necessarily lags
the regime it is chasing.
"""
from __future__ import annotations

import pandas as pd


def walk_forward(
    returns_by_key: dict, train: int, step: int
) -> tuple[pd.Series, list[tuple[pd.Timestamp, object]]]:
    R = pd.DataFrame(returns_by_key)
    n = len(R)
    out = pd.Series(index=R.index, dtype=float)
    picks: list[tuple[pd.Timestamp, object]] = []
    i = train
    while i < n:
        trailing = R.iloc[i - train:i]
        key = (1.0 + trailing).prod().idxmax()      # best trailing total return
        j = min(i + step, n)
        out.iloc[i:j] = R[key].iloc[i:j].to_numpy()
        picks.append((R.index[i], key))
        i = j
    return out.iloc[train:], picks
