"""Loader tests: FTSE and investing.com formats, and date-order inference."""
import numpy as np

from mc.data import load_close, _infer_dayfirst
import pandas as pd


def test_infer_dayfirst_uk_vs_us():
    assert _infer_dayfirst(pd.Series(["28/07/2009", "01/12/2023"]))  # 28 -> day
    assert not _infer_dayfirst(pd.Series(["11/26/2021", "01/02/1990"]))  # 26 -> US


def test_load_investing_format(tmp_path):
    p = tmp_path / "X.csv"
    p.write_text(
        "﻿Date,Price,Open,High,Low,Vol.,Change %\n"
        '09/01/2026,"14,216.0","14,010.0","14,332.0","13,972.0",1.51M,+0.28%\n'
        '08/01/2026,"14,176.0","14,030.0","14,292.0","14,022.0",2.49M,-0.34%\n'
        '07/01/2026,"14,224.0","14,050.0","14,312.0","14,038.0",2.01M,+1.32%\n'
    )
    s = load_close(p)
    # ascending by date, comma-thousands parsed, Price column picked
    assert list(s.values) == [14224.0, 14176.0, 14216.0]
    assert s.index.is_monotonic_increasing


def test_load_drops_nonnumeric_prices(tmp_path):
    p = tmp_path / "Y.csv"
    p.write_text(
        "Date,Price\n"
        "03/01/2020,-\n"
        '04/01/2020,"1,000.0"\n'
        '05/01/2020,"1,010.0"\n'
    )
    s = load_close(p)
    assert list(s.values) == [1000.0, 1010.0]
