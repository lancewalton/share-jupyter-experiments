"""Futures basket: does a minimum line span (cutting the 1.2-day whipsaw) help
where the gross edge is already near break-even? Sweep S at k=0.5.

On equities S made things monotonically worse. Futures trend more, so longer,
less-steep lines *might* capture trends instead of killing them — the open question.
"""
from __future__ import annotations

import numpy as np

import load_futures
from backtest import CostModel, simulate, trade_return

NET = CostModel(spread_bps_per_side=5, financing_annual=0.0)
SWEEP_S = [0, 10, 20, 40, 60]
UNIVERSE = load_futures.universe()


def pf(p):
    neg = -p[p < 0].sum()
    return p[p > 0].sum() / neg if neg > 0 else float("inf")


def config(S: int):
    g_pool, n_pool, days, g_rows, n_rows = [], [], [], [], []
    for t in UNIVERSE:
        df = load_futures.load(t)
        tr = simulate(df, k=0.5, min_span=S, costs=CostModel(0.0, 0.0))
        g = np.array([x.ret for x in tr])
        n = np.array([trade_return(x.direction, x.entry_price, x.exit_price, x.days, NET) for x in tr])
        if len(g) == 0:
            continue
        g_pool.extend(g.tolist()); n_pool.extend(n.tolist()); days.extend(x.days for x in tr)
        g_rows.append(np.prod(1 + g) - 1); n_rows.append(np.prod(1 + n) - 1)
    g, n = np.array(g_pool), np.array(n_pool)
    return {
        "n": len(g), "hold": np.mean(days), "win": 100 * (g > 0).mean(),
        "gmean": 100 * g.mean(), "gpf": pf(g), "nmean": 100 * n.mean(), "npf": pf(n),
        "gprof": 100 * np.mean([r > 0 for r in g_rows]),
        "nprof": 100 * np.mean([r > 0 for r in n_rows]),
    }


def main() -> None:
    lines = ["=== FUTURES min-span sweep (k=0.5, 5bps/side, no financing) ===",
             f"{'S':>4} {'trades':>7} {'hold':>6} {'win%':>6} {'gMean%':>8} {'gPF':>5} "
             f"{'gProf%':>7} {'nMean%':>8} {'nPF':>5} {'nProf%':>7}"]
    for S in SWEEP_S:
        a = config(S)
        lines.append(f"{S:>4} {a['n']:>7} {a['hold']:>6.1f} {a['win']:>6.1f} {a['gmean']:>8.3f} "
                     f"{a['gpf']:>5.2f} {a['gprof']:>7.0f} {a['nmean']:>8.3f} {a['npf']:>5.2f} {a['nprof']:>7.0f}")
    report = "\n".join(lines)
    print(report)
    from pathlib import Path
    (Path(__file__).resolve().parent / "backtest_futures_span_results.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
