"""Download ~30 long-history US large-cap adjusted closes for the breadth test
(does the momentum/trend decay hit US shares too, not just UK). Saves a gitignored
parquet of aligned closes.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u download_us.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
OUT = HERE / "us_closes.parquet"
TICKERS = ["AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "PEP", "JPM", "WMT", "HD",
           "MCD", "DIS", "CSCO", "INTC", "IBM", "MMM", "CAT", "BA", "GE", "PFE",
           "MRK", "VZ", "T", "CVX", "WFC", "AXP", "GS", "UNH", "ORCL", "TXN"]


def main() -> None:
    print(f"Downloading {len(TICKERS)} US names 2000-2026 (adjusted close)...", flush=True)
    data = yf.download(TICKERS, start="2000-01-01", auto_adjust=True, progress=False)
    close = data["Close"]
    close = close.dropna(how="all").sort_index()
    close.to_parquet(OUT)
    print(f"saved {OUT.name}: {close.shape[0]} days x {close.shape[1]} names, "
          f"{close.index.min().date()} -> {close.index.max().date()}", flush=True)
    print("per-name first valid dates:", flush=True)
    for t in close.columns:
        fv = close[t].first_valid_index()
        print(f"  {t:6} {fv.date() if fv is not None else 'none'}  ({int(close[t].notna().sum())} obs)")


if __name__ == "__main__":
    main()
