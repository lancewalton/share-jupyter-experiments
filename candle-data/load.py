"""Helpers to load the S&P 500 5-minute candle data.

    from candle_data.load import load_ticker, load_panel, load_long, universe

Timestamps are UTC-aware. Use ``tz_convert('America/New_York')`` for
session-local wall-clock times.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"


def universe() -> list[str]:
    """Tickers that have a parquet file on disk."""
    return sorted(p.stem for p in DATA_DIR.glob("*.parquet") if not p.stem.startswith("_"))


def load_ticker(ticker: str) -> pd.DataFrame:
    """OHLCV frame for one ticker, indexed by UTC datetime."""
    return pd.read_parquet(DATA_DIR / f"{ticker}.parquet")


def load_long(tickers: list[str] | None = None) -> pd.DataFrame:
    """Tidy long-form frame: columns ticker, datetime, o/h/l/c, adj_close, volume."""
    if tickers is None and (DATA_DIR / "_combined.parquet").exists():
        return pd.read_parquet(DATA_DIR / "_combined.parquet")
    frames = []
    for t in tickers or universe():
        df = load_ticker(t).reset_index()
        df.insert(0, "ticker", t)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_panel(field: str = "close", tickers: list[str] | None = None) -> pd.DataFrame:
    """Wide panel of one field: rows = datetime, columns = ticker.

    Aligns all tickers on a common timestamp index (NaN where a bar is missing).
    """
    cols = {}
    for t in tickers or universe():
        cols[t] = load_ticker(t)[field]
    return pd.DataFrame(cols).sort_index()
