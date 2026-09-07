"""Download multi-year daily OHLCV bars for a small, varied set of tickers.

Mirrors the approach of ``candle-data/download_sp500_5m.py`` but at daily
resolution and over a long history, which is what the swing-trading
trend-line structure needs.

    python download_daily.py

Writes one parquet per ticker into ``data/`` with a tz-naive DatetimeIndex
named ``date`` and columns open/high/low/close/adj_close/volume.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent / "data"

# A deliberately varied handful: a steady compounder, a mega-cap tech pair,
# a high-volatility name, a broad-market ETF, and a long sideways/decliner.
TICKERS = ["AAPL", "MSFT", "KO", "TSLA", "SPY", "INTC"]

START = "2015-01-01"
END = "2025-09-01"


def download(ticker: str) -> pd.DataFrame:
    raw = yf.download(
        ticker, start=START, end=END, interval="1d",
        auto_adjust=False, progress=False, threads=False,
    )
    if raw.empty:
        raise RuntimeError(f"no data for {ticker}")
    # yfinance may return a MultiIndex column frame for a single ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(
        columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
        }
    )[["open", "high", "low", "close", "adj_close", "volume"]]
    df.index.name = "date"
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for t in TICKERS:
        df = download(t)
        df.to_parquet(DATA_DIR / f"{t}.parquet")
        print(f"{t}: {len(df)} rows  {df.index.min().date()} -> {df.index.max().date()}")


if __name__ == "__main__":
    main()
