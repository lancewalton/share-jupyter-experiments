"""Load the FTSE daily OHLCV CSVs in ~/Projects/shares/data/yfinance.

These are investing.com-style exports: newest-first, dd/mm/yyyy dates, prices in
pence with thousands-commas, volume like "1.32M". This normalises them to an
ascending tz-naive DatetimeIndex named ``date`` with float o/h/l/c and volume.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cleaning import repair_bad_ticks

FTSE_DIR = Path("/Users/lance/Projects/shares/data/yfinance")

_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9}


def universe() -> list[str]:
    return sorted(p.stem for p in FTSE_DIR.glob("*.csv"))


def _to_float(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_volume(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    mult = s.str[-1].map(_SUFFIX)
    num = pd.to_numeric(s.str[:-1].str.replace(",", "", regex=False), errors="coerce")
    plain = pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")
    return (num * mult).fillna(plain)


def load(ticker: str, clean: bool = True) -> pd.DataFrame:
    df = pd.read_csv(FTSE_DIR / f"{ticker}.csv", encoding="utf-8-sig")
    out = pd.DataFrame(
        {
            "open": _to_float(df["Open"]),
            "high": _to_float(df["High"]),
            "low": _to_float(df["Low"]),
            "close": _to_float(df["Price"]),
            "volume": _parse_volume(df["Vol."]) if "Vol." in df else pd.NA,
        }
    )
    out.index = pd.to_datetime(df["Date"], dayfirst=True)
    out.index.name = "date"
    out = out[out[["open", "high", "low", "close"]].notna().all(axis=1)]
    out = out.sort_index()
    return repair_bad_ticks(out) if clean else out


if __name__ == "__main__":
    for t in ["AZN", "BP", "BARC"]:
        d = load(t)
        print(f"{t}: {len(d)} rows  {d.index.min().date()} -> {d.index.max().date()}"
              f"  last close {d['close'].iloc[-1]:.1f}")
