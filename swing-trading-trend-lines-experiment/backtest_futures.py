"""Run the method on the diversified futures basket.

Futures cost model: 5 bps/side (commission + slippage), NO financing (carry is
in the price). The decisive figure is the zero-cost GROSS edge — futures can only
push net toward gross, never past it.
"""
from __future__ import annotations

import numpy as np

import load_futures
from backtest import CostModel, simulate, trade_return

NET = CostModel(spread_bps_per_side=5, financing_annual=0.0)


def main() -> None:
    rows, g_pool, n_pool, days = [], [], [], []
    for t in load_futures.universe():
        df = load_futures.load(t)
        tr = simulate(df, k=0.5, min_span=0, costs=CostModel(0.0, 0.0))
        g = np.array([x.ret for x in tr])
        nret = np.array([trade_return(x.direction, x.entry_price, x.exit_price, x.days, NET) for x in tr])
        if len(g) == 0:
            continue
        g_pool.extend(g.tolist()); n_pool.extend(nret.tolist())
        days.extend(x.days for x in tr)
        rows.append((t, len(g), 100 * (g > 0).mean(), 100 * g.mean(),
                     100 * (np.prod(1 + g) - 1), 100 * (np.prod(1 + nret) - 1)))

    lines = ["=== FUTURES BASKET  k=0.5  (5bps/side, no financing) ===",
             f"{'market':12} {'n':>5} {'win%':>6} {'grossMean%':>11} {'grossTot%':>10} {'netTot%':>10}"]
    for r in sorted(rows, key=lambda x: -x[4]):
        lines.append(f"{r[0]:12} {r[1]:>5} {r[2]:>6.1f} {r[3]:>11.3f} {r[4]:>10.1f} {r[5]:>10.1f}")

    g, n = np.array(g_pool), np.array(n_pool)
    def pf(p):
        neg = -p[p < 0].sum()
        return p[p > 0].sum() / neg if neg > 0 else float("inf")
    lines += ["",
              f"AGG trades={len(g)}  hold={np.mean(days):.1f}d  win={100*(g>0).mean():.1f}%",
              f"  GROSS  mean/tr={100*g.mean():+.3f}%  profit factor={pf(g):.2f}  "
              f"% markets gross-profitable={100*np.mean([r[4]>0 for r in rows]):.0f}%",
              f"  NET    mean/tr={100*n.mean():+.3f}%  profit factor={pf(n):.2f}  "
              f"% markets net-profitable={100*np.mean([r[5]>0 for r in rows]):.0f}%"]
    report = "\n".join(lines)
    print(report)
    from pathlib import Path
    (Path(__file__).resolve().parent / "backtest_futures_results.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
