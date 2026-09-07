"""Cross-currency return panel from FRED exchange rates.

Every currency is expressed as its return vs the US dollar (up = the foreign
currency appreciates). FRED quotes some pairs as US$-per-foreign (DEXUSxx) and
others as foreign-per-US$ (DEXxxUS), so the latter are sign-flipped. CNY is
excluded (pegged/managed for most of the sample -> artificial returns).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

# (fred id, sign, label)  sign +1: US$/foreign (up=foreign strong); -1: foreign/US$
PANEL = [
    ("DEXUSEU", +1, "EUR"), ("DEXUSUK", +1, "GBP"), ("DEXUSAL", +1, "AUD"),
    ("DEXUSNZ", +1, "NZD"), ("DEXJPUS", -1, "JPY"), ("DEXCAUS", -1, "CAD"),
    ("DEXSZUS", -1, "CHF"), ("DEXMXUS", -1, "MXN"), ("DEXBZUS", -1, "BRL"),
    ("DEXSFUS", -1, "ZAR"), ("DEXINUS", -1, "INR"), ("DEXKOUS", -1, "KRW"),
    ("DEXSIUS", -1, "SGD"), ("DEXSDUS", -1, "SEK"), ("DEXNOUS", -1, "NOK"),
    ("DEXDNUS", -1, "DKK"),
]


def _level(fid: str) -> pd.Series:
    df = pd.read_csv(DATA / f"{fid}.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    df["d"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df["v"] = pd.to_numeric(df[df.columns[1]], errors="coerce")
    return df.dropna(subset=["d", "v"]).sort_values("d").set_index("d")["v"]


def build_fx_returns(panel=PANEL) -> pd.DataFrame:
    """Monthly log returns of each currency vs USD (foreign appreciation = +)."""
    cols = {}
    for fid, sign, label in panel:
        r = sign * np.log(_level(fid)).diff()
        cols[label] = r.resample("ME").sum()
    return pd.DataFrame(cols).dropna().sort_index()


# currencies with a FRED 3-month interbank rate (OECD IR3TIB01<cc>M156N)
RATE_MAP = {"EUR": "EZ", "GBP": "GB", "AUD": "AU", "NZD": "NZ", "JPY": "JP",
            "CAD": "CA", "CHF": "CH", "MXN": "MX", "KRW": "KR", "SEK": "SE",
            "NOK": "NO", "DKK": "DK"}


def load_rates() -> pd.DataFrame:
    """Monthly 3-month interbank rates (percent) for US + the rated currencies."""
    def r(cc):
        return _level(f"rate_{cc}").resample("ME").last()
    cols = {"US": r("US")}
    for cur, cc in RATE_MAP.items():
        cols[cur] = r(cc)
    return pd.DataFrame(cols).sort_index()


def build_total_returns():
    """(total_return, rate_differential) monthly frames for the rated currencies.

    total_return = spot return + carry earned = spot + (r_foreign − r_US)/12 using
    the PRIOR month's rate (causal). Differential is the current carry signal.
    """
    curs = list(RATE_MAP)
    spot = build_fx_returns()[curs]
    rates = load_rates()
    diff = rates[curs].sub(rates["US"], axis=0) / 100.0        # annual, decimal
    idx = spot.index.intersection(diff.index)
    spot, diff = spot.loc[idx], diff.loc[idx]
    carry = diff.shift(1) / 12.0                                # earned this month
    total = (spot + carry).dropna()
    return total, diff.loc[total.index]
