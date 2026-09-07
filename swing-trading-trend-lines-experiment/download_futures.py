"""Download long-history daily continuous futures for a diversified basket.

Front-month continuous contracts via yfinance. NOTE: these are not roll-/back-
adjusted, so they carry roll gaps and (for WTI in 2020) even negative prints —
non-positive OHLC rows are dropped so the ln(price) structure is well-defined.
This is a first-pass fair-shot test, not a production futures dataset.

    python download_futures.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent / "data_futures"

# Diversified cross-asset set (the classic trend-following universe) + 2 indices.
TICKERS = {
    "CL=F": "WTI_crude", "NG=F": "natgas", "RB=F": "gasoline", "HO=F": "heating_oil",
    "GC=F": "gold", "SI=F": "silver", "HG=F": "copper",
    "ZC=F": "corn", "ZS=F": "soybeans", "ZW=F": "wheat", "KC=F": "coffee",
    "SB=F": "sugar", "CT=F": "cotton",
    "ZN=F": "us10y", "ZB=F": "us30y",
    "6E=F": "eur", "6J=F": "jpy", "6B=F": "gbp", "6A=F": "aud", "6C=F": "cad",
    "ES=F": "sp500", "NQ=F": "nasdaq",
}

START, END = "2000-01-01", "2025-09-01"


def download(sym: str) -> pd.DataFrame:
    raw = yf.download(sym, start=START, end=END, interval="1d",
                      auto_adjust=False, progress=False, threads=False)
    if raw.empty:
        raise RuntimeError(f"no data for {sym}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                             "Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
    df = df[["open", "high", "low", "close", "adj_close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for sym, name in TICKERS.items():
        try:
            df = download(sym)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"{name:12} ({sym}): FAILED {e}")
            continue
        df.to_parquet(DATA_DIR / f"{name}.parquet")
        print(f"{name:12} ({sym}): {len(df):5} rows  {df.index.min().date()} -> {df.index.max().date()}")


if __name__ == "__main__":
    main()
