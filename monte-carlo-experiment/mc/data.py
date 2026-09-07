"""Load price time-series into an ascending close-price series.

Handles two on-disk formats transparently:
  * FTSE ``all.csv``          -- Close column, MM/DD/YYYY, newest-first, commas.
  * investing.com equities    -- Price column, DD/MM/YYYY, newest-first, commas,
                                 BOM, extra Vol./Change% columns.
Everything downstream sees only an ascending float array, so onboarding a new
source only ever needs this module.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = Path.home() / "Projects" / "FTSEData" / "all.csv"

_PRICE_COLS = ("Close", "Price", "Adj Close")


def _pick_price_col(df: pd.DataFrame) -> str:
    for c in _PRICE_COLS:
        if c in df.columns:
            return c
    raise ValueError(f"no price column among {_PRICE_COLS}; got {list(df.columns)}")


def _infer_dayfirst(dates: pd.Series) -> bool:
    """Infer DD/MM vs MM/DD by finding an unambiguous token in the column."""
    parts = dates.dropna().astype(str).str.split("/", expand=True)
    if parts.shape[1] < 2:
        return False
    a = pd.to_numeric(parts[0], errors="coerce")
    b = pd.to_numeric(parts[1], errors="coerce")
    if (a > 12).any():  # first field exceeds 12 -> it must be the day
        return True
    if (b > 12).any():  # second field exceeds 12 -> it must be the day
        return False
    return False  # fully ambiguous: default to US-style (FTSE all.csv)


def load_close(
    path: str | Path = DEFAULT_CSV,
    price_col: str | None = None,
    dayfirst: bool | None = None,
) -> pd.Series:
    """Return the close/last price as a float Series indexed by ascending date.

    ``price_col`` and ``dayfirst`` are auto-detected but can be overridden.
    Rows with non-numeric prices are dropped; positivity is validated (log
    returns need P > 0); duplicate dates keep the last.
    """
    df = pd.read_csv(path, thousands=",", encoding="utf-8-sig")
    df.columns = [c.strip().strip('"') for c in df.columns]
    col = price_col or _pick_price_col(df)
    if dayfirst is None:
        dayfirst = _infer_dayfirst(df["Date"].astype(str))
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=dayfirst, errors="coerce")
    # strip any surviving thousands separators (present when a stray non-numeric
    # cell made pandas read the column as strings) before coercion.
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    df = df.dropna(subset=["Date", col]).sort_values("Date")
    df = df[~df["Date"].duplicated(keep="last")].reset_index(drop=True)
    close = df.set_index("Date")[col].astype(float)
    if len(close) == 0:
        raise ValueError(f"no usable rows in {path}")
    if (close <= 0).any():
        raise ValueError("close series contains non-positive prices")
    return close


def close_array(
    path: str | Path = DEFAULT_CSV,
    price_col: str | None = None,
    dayfirst: bool | None = None,
) -> np.ndarray:
    """Ascending close prices as a plain float array."""
    return load_close(path, price_col, dayfirst).to_numpy()
