"""Render the deterministic support/resistance structure on daily bars.

Hulls are computed in ln(price) space (constant %-growth lines), then drawn on a
log-scaled price axis so the rays are straight while the labels stay in dollars.
Each trend line's a->b segment is solid; its ray extension to the final bar is
dashed. The final ("signal") bar is excluded when building the structure, per
the README, and marked on the chart.

    python plot_structure.py                 # all tickers, full history + 1y
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trendlines import resistance_lines, support_lines

DATA_DIR = Path(__file__).resolve().parent / "data"
CHART_DIR = Path(__file__).resolve().parent / "charts"

SUPPORT_C = "#1a8a3a"
RESIST_C = "#c02020"


def _draw_lines(ax, lines, x_last, values, colour):
    """Draw each line's solid a->b segment and its dashed ray to x_last."""
    for ln in lines:
        seg_x = [ln.a, ln.b]
        ax.plot(seg_x, np.exp([ln.ya, ln.yb]), color=colour, lw=1.4, alpha=0.9)
        if ln.b < x_last:
            ray_x = np.array([ln.b, x_last])
            ax.plot(ray_x, np.exp(ln.value_at(ray_x)), color=colour,
                    lw=1.0, ls="--", alpha=0.55)


def plot_frame(label: str, df: pd.DataFrame, window: int | None, ax) -> None:
    if window is not None:
        df = df.iloc[-window:]
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    # Exclude the final (signal) bar when establishing structure.
    ln_low = np.log(low[:-1])
    ln_high = np.log(high[:-1])
    supports = support_lines(ln_low)
    resistances = resistance_lines(ln_high)

    x = np.arange(len(df))
    x_last = len(df) - 1
    ax.fill_between(x, low, high, color="#b8c4d0", alpha=0.5, lw=0)
    ax.plot(x, close, color="#33404d", lw=0.8)
    _draw_lines(ax, supports, x_last, ln_low, SUPPORT_C)
    _draw_lines(ax, resistances, x_last, ln_high, RESIST_C)
    ax.axvline(x_last, color="#888", lw=0.6, ls=":")

    ax.set_yscale("log")
    ax.set_title(f"{label}  ({len(df)} bars)  S:{len(supports)}  R:{len(resistances)}",
                 fontsize=10)
    ax.set_xlim(0, x_last)
    idx = df.index
    ticks = np.linspace(0, x_last, 5).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([idx[t].strftime("%Y-%m") for t in ticks], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)


def _parquet_loader(ticker: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{ticker}.parquet")


def make_grid(tickers: list[str], loader, window: int | None,
              fname: str, suptitle: str) -> Path:
    rows = (len(tickers) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 3.2 * rows))
    for ax, t in zip(axes.flat, tickers):
        plot_frame(t, loader(t), window, ax)
    for ax in axes.flat[len(tickers):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12, y=1.0)
    fig.tight_layout()
    out = CHART_DIR / fname
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    CHART_DIR.mkdir(exist_ok=True)
    import load_ftse
    ftse = ["AZN", "BP", "BARC", "VOD", "ULVR", "RIO"]
    print(make_grid(ftse, load_ftse.load, None, "ftse_structure_full.png",
                    "FTSE — Support (green) / Resistance (red), full history (~27y), ln-price hull"))
    print(make_grid(ftse, load_ftse.load, 252, "ftse_structure_1y.png",
                    "FTSE — Support (green) / Resistance (red), last ~1 year, ln-price hull"))


if __name__ == "__main__":
    main()
