"""Load and clean the EODHD UK daily panel for the MACD backtest.

The panel is survivorship-free (active + delisted names). Bad ``adjusted_close``
ticks appear as huge single-day log-return spikes; capping the daily log return at
the programme's glitch threshold (|r| <= 0.6) removes them without touching real
gaps or crashes. Daily simple returns are taken directly from the capped log
returns (``expm1``), so they are bounded and never blow up to infinity. A clean
price series (for the MACD EMAs) is rebuilt from the same capped log returns.

The investable universe is the top-N names by trailing turnover, refreshed
point-in-time, so illiquid micro-caps (bid-ask bounce, stale prices) never enter.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
DATA = HERE.parent / "data" / "eodhd"
OHLCV = DATA / "eodhd_uk_ohlcv.parquet"

GLITCH_CAP = 0.6


def winsorise_log_returns(logret: pd.DataFrame, cap: float = GLITCH_CAP) -> pd.DataFrame:
    return logret.clip(lower=-cap, upper=cap)


def clamp_wicks(
    adj_low: pd.DataFrame, adj_high: pd.DataFrame, adj_close: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clamp adjusted low/high wicks to a sane band around the adjusted close.

    A low below half the close, or a high above twice it, is a bad tick; clamp it to
    that bound. Also enforce low <= close <= high so the MACD-on-lows / MACD-on-highs
    signals never see a spike.
    """
    lo = adj_low.clip(lower=0.5 * adj_close, upper=adj_close)
    hi = adj_high.clip(lower=adj_close, upper=2.0 * adj_close)
    return lo, hi


def top_n_mask(liquidity: pd.DataFrame, n: int) -> pd.DataFrame:
    """Boolean frame: True for the n most-liquid names each day (NaN never eligible)."""
    rank = liquidity.rank(axis=1, ascending=False, method="first")
    return (rank <= n) & liquidity.notna()


def _ccy_factor() -> dict:
    m: dict = {}
    for fn in ("lse_active.json", "lse_delisted.json"):
        path = DATA / fn
        if not path.exists():
            continue
        for x in json.load(open(path)):
            code = x.get("Code")
            if code and code not in m:
                m[code] = 0.01 if x.get("Currency") == "GBX" else 1.0
    return m


def load_panel(
    min_obs: int = 750, start: str = "1998-01-01"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (clean_price, simple_returns, turnover) wide panels, dates x codes.

    Prices are relative (seeded at 1.0 per name); MACD signals and returns are
    scale-invariant. Turnover is in GBP (GBX prices rescaled) for the liquidity screen.
    """
    df = pd.read_parquet(OHLCV, columns=["date", "code", "adjusted_close", "close", "volume"])
    df = df[df["date"] >= pd.Timestamp(start)]

    fac = _ccy_factor()
    df["turnover"] = df["close"] * df["volume"] * df["code"].map(fac).fillna(1.0)

    adj = df.pivot(index="date", columns="code", values="adjusted_close").sort_index()
    adj = adj.where(adj > 0)
    keep = adj.notna().sum() >= min_obs
    adj = adj.loc[:, keep]

    logret = winsorise_log_returns(np.log(adj).diff())
    returns = np.expm1(logret).where(adj.notna())
    price = np.exp(logret.fillna(0.0).cumsum()).where(adj.notna())

    turn = df.pivot(index="date", columns="code", values="turnover").sort_index()
    turn = turn.reindex(columns=adj.columns).loc[adj.index]
    return price, returns, turn


def load_ohlc(
    min_obs: int = 750, start: str = "1998-01-01"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (clean_close, clean_high, clean_low, returns, turnover) wide panels.

    Highs and lows ride the same winsorised clean-close path via their intraday ratio
    to the adjusted close (clean_high = clean_close * adj_high/adj_close), so absolute
    bad ticks cannot whipsaw the MACD-on-highs / MACD-on-lows signals. Wicks are then
    clamped to a sane band. Used by the low-for-long / high-for-short test.
    """
    df = pd.read_parquet(
        OHLCV, columns=["date", "code", "high", "low", "close", "adjusted_close", "volume"]
    )
    df = df[df["date"] >= pd.Timestamp(start)]

    fac = _ccy_factor()
    df["turnover"] = df["close"] * df["volume"] * df["code"].map(fac).fillna(1.0)

    adj = df.pivot(index="date", columns="code", values="adjusted_close").sort_index()
    adj = adj.where(adj > 0)
    keep = adj.notna().sum() >= min_obs
    adj = adj.loc[:, keep]
    cols, idx = adj.columns, adj.index

    raw_close = df.pivot(index="date", columns="code", values="close").reindex(index=idx, columns=cols)
    raw_high = df.pivot(index="date", columns="code", values="high").reindex(index=idx, columns=cols)
    raw_low = df.pivot(index="date", columns="code", values="low").reindex(index=idx, columns=cols)

    logret = winsorise_log_returns(np.log(adj).diff())
    returns = np.expm1(logret).where(adj.notna())
    clean_close = np.exp(logret.fillna(0.0).cumsum()).where(adj.notna())

    ratio_high = (raw_high / raw_close.where(raw_close > 0))
    ratio_low = (raw_low / raw_close.where(raw_close > 0))
    clean_high = (clean_close * ratio_high).where(adj.notna())
    clean_low = (clean_close * ratio_low).where(adj.notna())
    clean_low, clean_high = clamp_wicks(clean_low, clean_high, clean_close)

    turn = df.pivot(index="date", columns="code", values="turnover").reindex(index=idx, columns=cols)
    return clean_close, clean_high, clean_low, returns, turn
