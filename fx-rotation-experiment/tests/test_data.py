"""Tests for the FX return / carry panel."""
import numpy as np

from fx.data import RATE_MAP, build_fx_returns, build_total_returns, load_rates


def test_fx_returns_shape_and_sanity():
    R = build_fx_returns()
    assert R.shape[1] == 16
    assert {"EUR", "JPY", "MXN", "GBP"}.issubset(R.columns)
    assert R.index[0].year == 1999                    # EUR-era common window
    # monthly FX log returns: small, rarely |>40%|
    assert R.abs().mean().mean() < 0.05
    assert (R.abs() > 0.4).mean().mean() < 0.002


def test_rates_cover_the_rated_currencies():
    rt = load_rates()
    assert "US" in rt.columns
    assert set(RATE_MAP).issubset(rt.columns)
    # 3-month rates in percent: mostly 0-20
    assert 0 < np.nanmedian(rt.to_numpy()) < 20


def test_total_return_adds_carry_causally():
    total, diff = build_total_returns()
    assert total.shape[1] == len(RATE_MAP)
    assert total.index.equals(diff.index)
    # total return = spot + lagged carry, wherever both are defined
    spot = build_fx_returns()[list(RATE_MAP)].reindex(total.index)
    lhs = (total - spot).to_numpy().ravel()
    rhs = (diff.shift(1) / 12.0).to_numpy().ravel()
    m = np.isfinite(lhs) & np.isfinite(rhs)
    assert m.sum() > 1000 and np.allclose(lhs[m], rhs[m], atol=1e-9)
