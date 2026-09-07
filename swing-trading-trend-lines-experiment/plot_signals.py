"""Mark walk-forward entry signals on price, to eyeball whether they fire sensibly.

For each evaluated bar the full history up to it is used (faithful all-history
anchoring); we display only the recent window and drop a marker on each signal
day. The action/safety lines shown are the final bar's structure, for context.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import load_ftse
from scan import scan
from trendlines import resistance_lines, support_lines

CHART_DIR = Path(__file__).resolve().parent / "charts"
WINDOW = 300


def plot_ticker(tkr: str, k: float, ax) -> None:
    df = load_ftse.load(tkr)
    sigs = scan(df, k=k, evaluate_last=WINDOW)

    struct = df.iloc[:-1]
    supports = support_lines(np.log(struct["low"].to_numpy()))
    resistances = resistance_lines(np.log(struct["high"].to_numpy()))

    win = df.iloc[-WINDOW:]
    gx = np.arange(len(df) - WINDOW, len(df))
    ax.plot(win.index, win["close"], color="#33404d", lw=0.9)

    for ln in [supports[-1], resistances[-1]]:
        colour = "#1a8a3a" if ln in supports else "#c02020"
        ax.plot(win.index, np.exp(ln.value_at(gx)), color=colour, lw=1.2, alpha=0.8)

    for when, sig in sigs:
        up = sig.direction == "LONG"
        ax.scatter(when, df.loc[when, "low" if up else "high"],
                   marker="^" if up else "v",
                   color="#1a8a3a" if up else "#c02020", s=55, zorder=5)

    ax.set_yscale("log")
    longs = sum(s.direction == "LONG" for _, s in sigs)
    shorts = len(sigs) - longs
    ax.set_title(f"{tkr}  last {WINDOW} bars  —  {longs} LONG (^)  {shorts} SHORT (v)",
                 fontsize=10)
    ax.tick_params(labelsize=7)


def main() -> None:
    CHART_DIR.mkdir(exist_ok=True)
    tickers = ["BARC", "VOD", "BP", "RIO"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, t in zip(axes.flat, tickers):
        plot_ticker(t, k=0.5, ax=ax)
    fig.suptitle("Entry signals (k=0.5·ATR, ≥3 touches) — all-history structure, walk-forward",
                 fontsize=12)
    fig.tight_layout()
    out = CHART_DIR / "signals.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
