"""Out-of-sample validation of the futures min-span result.

Split every market's trades at 2013-01-01: train era 2000-2012, test era 2013-2025.
If the net-positive edge at larger S survives into the later (post-supercycle) era,
it is a live lead; if it only shows in the early era, it is a decayed premium.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import load_futures
from backtest import CostModel, simulate, trade_return

NET = CostModel(spread_bps_per_side=5, financing_annual=0.0)
SPLIT = pd.Timestamp("2013-01-01")
SWEEP_S = [0, 20, 40, 60]


def pf(p):
    neg = -p[p < 0].sum()
    return p[p > 0].sum() / neg if neg > 0 else float("inf")


def stats(g, n, days):
    g, n = np.array(g), np.array(n)
    if len(g) == 0:
        return None
    return (len(g), np.mean(days), 100 * (g > 0).mean(),
            100 * g.mean(), pf(g), 100 * n.mean(), pf(n))


def main() -> None:
    lines = ["=== FUTURES out-of-sample by era (k=0.5, 5bps/side) ===",
             "train = entries before 2013 ; test = 2013 onward",
             f"{'S':>4} {'era':>5} {'trades':>7} {'hold':>6} {'win%':>6} "
             f"{'gMean%':>8} {'gPF':>5} {'nMean%':>8} {'nPF':>5}"]
    for S in SWEEP_S:
        buckets = {"train": ([], [], []), "test": ([], [], [])}
        for t in load_futures.universe():
            df = load_futures.load(t)
            for tr in simulate(df, k=0.5, min_span=S, costs=CostModel(0.0, 0.0)):
                era = "train" if tr.entry_date < SPLIT else "test"
                g, n, d = buckets[era]
                g.append(tr.ret)
                n.append(trade_return(tr.direction, tr.entry_price, tr.exit_price, tr.days, NET))
                d.append(tr.days)
        for era in ("train", "test"):
            s = stats(*buckets[era])
            if s:
                lines.append(f"{S:>4} {era:>5} {s[0]:>7} {s[1]:>6.1f} {s[2]:>6.1f} "
                             f"{s[3]:>8.3f} {s[4]:>5.2f} {s[5]:>8.3f} {s[6]:>5.2f}")
        lines.append("")
    report = "\n".join(lines)
    print(report)
    from pathlib import Path
    (Path(__file__).resolve().parent / "backtest_futures_oos_results.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
