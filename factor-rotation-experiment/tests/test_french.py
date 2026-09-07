"""Tests for the Ken French data-library parsers."""
import numpy as np

from pca.french import load_ff_market, load_industry_vw, load_regions


def test_industry_portfolios_shape_and_scale():
    df = load_industry_vw()
    assert df.shape[1] == 12
    assert "Enrgy" in df.columns and "BusEq" in df.columns
    assert df.index[0].year == 1926 and df.index[-1].year >= 2025
    # monthly total returns as decimals: mean ~ +0.5-1.5%/mo, rarely |>50%|
    mu = df.loc["1945":].mean().mean()
    assert 0.003 < mu < 0.02
    assert (df.loc["1945":].abs() > 0.6).mean().mean() < 0.001


def test_ff_market_is_total_return():
    ff = load_ff_market()
    assert {"MktRF", "RF", "Mkt"}.issubset(ff.columns)
    assert np.allclose(ff["Mkt"], ff["MktRF"] + ff["RF"])
    # US equity ~ 10-13%/yr since 1945
    assert 0.08 < ff["Mkt"].loc["1945":].mean() * 12 < 0.15


def test_regions_load_five_since_1990():
    R, rf = load_regions()
    assert R.shape[1] == 5
    assert {"North America", "Europe", "Japan", "Emerging"}.issubset(R.columns)
    assert R.index[0].year == 1990 and len(rf) == len(R)
    # regional equity ~ 5-14%/yr in USD; monthly returns sane
    assert (0.04 < R.mean() * 12).all() and (R.mean() * 12 < 0.16).all()
    assert (R.abs() > 0.6).mean().mean() < 0.001
