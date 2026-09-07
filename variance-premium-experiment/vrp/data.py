"""Load the VIX + S&P 500 panel for the variance-risk-premium study.

VIX (30-day implied volatility of the S&P 500, in annualised vol points) comes
from FRED (cached in ``data/vix.csv``); the S&P 500 price comes from the shared
equity data via the ``mc`` loader. Everything downstream sees one aligned frame
of ``spx`` (price) and ``vix`` (implied vol %) on common trading days.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
VIX_CSV = HERE / "data" / "vix.csv"
VIX3M_CSV = HERE / "data" / "vix3m.csv"       # FRED VXVCLS (3-month VIX)
SPX_CSV = HERE / "data" / "sp500_fred.csv"   # FRED SP500 (clean); the local SPX.csv is corrupt


def _fred(path, col: str) -> pd.Series:
    """Read a FRED single-series CSV (observation_date, VALUE); '.' = missing."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df[col] = pd.to_numeric(df[df.columns[1]], errors="coerce")
    df = df.dropna(subset=["date", col]).sort_values("date")
    return df.set_index("date")[col]


def load_vix(path: str | Path = VIX_CSV) -> pd.Series:
    """FRED VIXCLS: implied vol in annualised points, ascending by date."""
    return _fred(path, "vix")


def load_spx(path: str | Path = SPX_CSV) -> pd.Series:
    """FRED SP500 close, ascending by date. (The local investing.com SPX.csv is
    corrupt — wrong historical levels — so we use FRED's clean series.)"""
    return _fred(path, "spx")


def load_vix3m(path: str | Path = VIX3M_CSV) -> pd.Series:
    """FRED VXVCLS: 3-month implied vol in annualised points."""
    return _fred(path, "vix3m")


def panel(spx_path: str | Path = SPX_CSV, vix_path: str | Path = VIX_CSV) -> pd.DataFrame:
    """Aligned ``spx`` / ``vix`` frame on their common trading days."""
    spx, vix = load_spx(spx_path), load_vix(vix_path)
    df = pd.DataFrame({"spx": spx, "vix": vix}).dropna()
    return df.sort_index()


def term_structure(vix_path: str | Path = VIX_CSV,
                   vix3m_path: str | Path = VIX3M_CSV) -> pd.DataFrame:
    """Aligned ``vix`` (1-month) / ``vix3m`` (3-month) frame -- the VIX curve."""
    df = pd.DataFrame({"vix": load_vix(vix_path), "vix3m": load_vix3m(vix3m_path)}).dropna()
    return df.sort_index()
