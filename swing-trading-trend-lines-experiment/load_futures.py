"""Load the downloaded continuous-futures daily bars (data_futures/*.parquet)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cleaning import repair_bad_ticks

FUT_DIR = Path(__file__).resolve().parent / "data_futures"


def universe() -> list[str]:
    return sorted(p.stem for p in FUT_DIR.glob("*.parquet"))


def load(ticker: str, clean: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(FUT_DIR / f"{ticker}.parquet").sort_index()
    return repair_bad_ticks(df) if clean else df


if __name__ == "__main__":
    for t in universe():
        d = load(t)
        print(f"{t:12} {len(d):5} rows  {d.index.min().date()} -> {d.index.max().date()}")
