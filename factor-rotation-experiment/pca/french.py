"""Parsers for the Ken French data library files (multi-section CSVs).

Industry portfolios: value-weighted monthly TOTAL returns (dividends included) for
12 industries, 1926-2026. F-F factors: the market excess return and risk-free rate
for a proper buy-and-hold-market benchmark. Returns are converted to decimals;
French's -99.99 / -999 missing codes become NaN.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FRENCH = Path(__file__).resolve().parent.parent / "data" / "french"


def _section_rows(lines, header_key):
    i = next(k for k, l in enumerate(lines) if header_key in l)
    cl = i if lines[i].lstrip().startswith(",") else i + 1   # columns on this line or next
    cols = [c.strip() for c in lines[cl].split(",") if c.strip()]
    rows = []
    for l in lines[cl + 1:]:
        p = [x.strip() for x in l.split(",")]
        if not p or not p[0][:6].isdigit() or len(p[0]) != 6:
            break
        rows.append((p[0], [float(x) for x in p[1:1 + len(cols)]]))
    idx = pd.to_datetime([r[0] for r in rows], format="%Y%m")
    return pd.DataFrame([r[1] for r in rows], index=idx, columns=cols)


def load_industry_vw(path: Path = FRENCH / "12_Industry_Portfolios.csv") -> pd.DataFrame:
    lines = Path(path).read_text().splitlines()
    df = _section_rows(lines, "Average Value Weighted Returns -- Monthly")
    return df.replace([-99.99, -999.0], np.nan) / 100.0


def load_ff_market(path: Path = FRENCH / "F-F_Research_Data_Factors.csv") -> pd.DataFrame:
    lines = Path(path).read_text().splitlines()
    df = _section_rows(lines, ",Mkt-RF,SMB,HML,RF")
    out = pd.DataFrame({"MktRF": df["Mkt-RF"], "RF": df["RF"]}) / 100.0
    out["Mkt"] = out["MktRF"] + out["RF"]                # market total return
    return out


REGION_FILES = {
    "North America": "North_America_3_Factors", "Europe": "Europe_3_Factors",
    "Japan": "Japan_3_Factors", "Asia Pacific": "Asia_Pacific_ex_Japan_3_Factors",
    "Emerging": "Emerging_5_Factors",
}


def load_regions(files=REGION_FILES):
    """Regional equity monthly TOTAL returns (USD) + the common risk-free rate.

    Each Ken French regional factor file gives Mkt-RF and RF; the region's total
    return is Mkt-RF + RF. Returns (regions DataFrame, RF Series) aligned on the
    common months (developed since 1990-07, emerging since 1990-07)."""
    out, rf = {}, None
    for name, fid in files.items():
        df = _section_rows((FRENCH / f"{fid}.csv").read_text().splitlines(), ",Mkt-RF") / 100.0
        out[name] = df["Mkt-RF"] + df["RF"]
        if rf is None:
            rf = df["RF"]
    R = pd.DataFrame(out).dropna()
    return R, rf.reindex(R.index)
