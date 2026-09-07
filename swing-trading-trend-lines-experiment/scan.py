"""Walk-forward scan: replay a ticker day by day and record entry signals.

At each bar t only data up to and including t is passed to ``latest_signal``
(which itself excludes t from the structure), so there is no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals import Signal, latest_signal


def scan(df: pd.DataFrame, k: float = 0.5, atr_period: int = 14, min_touches: int = 3,
         warmup: int = 60, evaluate_last: int | None = None
         ) -> list[tuple[pd.Timestamp, Signal]]:
    """Each evaluated bar still sees the full history up to it (faithful all-history
    anchoring); ``evaluate_last`` only limits how many recent bars are tested."""
    start = warmup if evaluate_last is None else max(warmup, len(df) - evaluate_last)
    out: list[tuple[pd.Timestamp, Signal]] = []
    for t in range(start, len(df)):
        sig = latest_signal(df.iloc[: t + 1], k=k, atr_period=atr_period,
                            min_touches=min_touches)
        if sig is not None:
            out.append((df.index[t], sig))
    return out


if __name__ == "__main__":
    import time
    import load_ftse

    for tkr in ["BARC", "VOD", "AZN", "BP", "ULVR", "RIO"]:
        df = load_ftse.load(tkr)
        t0 = time.time()
        sigs = scan(df, evaluate_last=300)
        longs = sum(s.direction == "LONG" for _, s in sigs)
        shorts = sum(s.direction == "SHORT" for _, s in sigs)
        print(f"{tkr}: {len(sigs)} signals ({longs}L/{shorts}S) over {len(df)} bars"
              f"  [{time.time()-t0:.1f}s]")
