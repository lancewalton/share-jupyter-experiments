"""Weekly sentiment / risk-appetite panel (the scoped Idea-3 dataset).

Retail positioning (AAII, put/call, BUX-style up/down bets) is not free, so we
test the honest core with aggregate fear / risk-appetite / stress gauges that
ARE free from FRED, aligned to a weekly (Friday) grid against long equity:

  * VIX     -- the equity fear gauge (implied vol);
  * NFCI    -- National Financial Conditions Index (higher = tighter/stressed);
  * ANFCI   -- the risk-appetite-adjusted NFCI (excess conditions);
  * STLFSI4 -- St Louis Fed financial stress index.
All are "stress up" in sign, so the contrarian hypothesis predicts POSITIVE
correlation with the forward equity return.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def _fred(fid: str) -> pd.Series:
    df = pd.read_csv(DATA / f"{fid}.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    df["d"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df["v"] = pd.to_numeric(df[df.columns[1]], errors="coerce")
    return df.dropna(subset=["d", "v"]).sort_values("d").set_index("d")["v"]


def weekly_panel() -> pd.DataFrame:
    eq = np.log(_fred("NASDAQCOM")).resample("W-FRI").last()
    ret = eq.diff().rename("ret")                        # weekly equity log return
    vix = _fred("VIXCLS").resample("W-FRI").last().rename("VIX")
    nfci = _fred("NFCI").resample("W-FRI").last().rename("NFCI")
    anfci = _fred("ANFCI").resample("W-FRI").last().rename("ANFCI")
    stlfsi = _fred("STLFSI4").resample("W-FRI").last().rename("STLFSI")
    df = pd.concat([ret, vix, nfci, anfci, stlfsi], axis=1).dropna()
    return df.sort_index()


SIGNALS = ["VIX", "NFCI", "ANFCI", "STLFSI"]
