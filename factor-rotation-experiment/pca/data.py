"""Assemble a cross-asset daily return panel from cached FRED series.

Each instrument becomes a daily "return-like" series on a common scale for PCA:
  * price  (equity, commodity, FX) -> log return;
  * yield  (Treasuries)            -> -Δyield (bond return DIRECTION: yields down =
                                      bonds up); magnitude is standardised away;
  * vix                            -> Δlog(level) (the volatility factor).
Aligned on the intersection of trading days. Gold and credit spreads are absent —
FRED serves neither cleanly for free (dead IDs / licensed 3-year cap).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"

# (fred id, kind, label, asset class)
PANEL = [
    ("NASDAQCOM", "price", "US equity", "equity"),
    ("DGS2", "yield", "UST 2Y", "rates"),
    ("DGS10", "yield", "UST 10Y", "rates"),
    ("DGS30", "yield", "UST 30Y", "rates"),
    ("DCOILWTICO", "price", "WTI oil", "commodity"),
    ("DHHNGSP", "price", "Nat gas", "commodity"),
    ("DEXUSEU", "price", "EUR/USD", "fx"),
    ("DEXJPUS", "price", "USD/JPY", "fx"),
    ("DEXUSUK", "price", "GBP/USD", "fx"),
    ("VIXCLS", "vix", "VIX", "vol"),
]


def _fred_level(fid: str, data_dir: Path = DATA) -> pd.Series:
    df = pd.read_csv(data_dir / f"{fid}.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df["v"] = pd.to_numeric(df[df.columns[1]], errors="coerce")
    return df.dropna(subset=["date", "v"]).sort_values("date").set_index("date")["v"]


def _to_return(level: pd.Series, kind: str) -> pd.Series:
    if kind == "yield":
        return -level.diff()
    return np.log(level).diff()


def build_returns(panel=PANEL, data_dir: Path = DATA) -> pd.DataFrame:
    """Aligned daily return matrix (columns = instrument labels)."""
    cols = {label: _to_return(_fred_level(fid, data_dir), kind)
            for fid, kind, label, _ in panel}
    return pd.DataFrame(cols).dropna().sort_index()


def asset_classes(panel=PANEL) -> dict:
    return {label: cls for _, _, label, cls in panel}
