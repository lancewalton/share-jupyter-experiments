"""Download a few long-history bond ETFs (adjusted close = total return) for the
bond breadth test. Thin, highly-correlated cross-section -- treat as indicative.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u download_bonds.py
"""
from __future__ import annotations

from pathlib import Path

import yfinance as yf

HERE = Path(__file__).resolve().parent
OUT = HERE / "bond_closes.parquet"
# TLT 20y+, IEF 7-10y, SHY 1-3y Treasuries; LQD IG corp; AGG aggregate; TIP inflation
TICKERS = ["TLT", "IEF", "SHY", "LQD", "AGG", "TIP"]


def main() -> None:
    print(f"Downloading {len(TICKERS)} bond ETFs (adjusted close = total return)...", flush=True)
    data = yf.download(TICKERS, start="2002-01-01", auto_adjust=True, progress=False)
    close = data["Close"].dropna(how="all").sort_index()
    close.to_parquet(OUT)
    print(f"saved {OUT.name}: {close.shape[0]} days x {close.shape[1]} names, "
          f"{close.index.min().date()} -> {close.index.max().date()}", flush=True)
    for t in close.columns:
        fv = close[t].first_valid_index()
        print(f"  {t:5} {fv.date() if fv is not None else 'none'}  ({int(close[t].notna().sum())} obs)")


if __name__ == "__main__":
    main()
