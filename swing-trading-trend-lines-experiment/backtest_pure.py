"""Author-faithful exit: the safety line IS the stop (no ATR disaster-floor).

Tests Lance's hypothesis that the 20-day min-span keeps the bare safety line's
whipsaw down. Compares, at k=0.5:
  - pure safety exit, S=0   (bare line, expect heavy whipsaw)
  - pure safety exit, S=20  (author-faithful, pre-registered)
  - 2*ATR floor + trail, S=20 (our earlier version, for reference)
"""
from __future__ import annotations

import numpy as np

import load_ftse
from backtest import CostModel, simulate, trade_return
from backtest_run import UNIVERSE

NET = CostModel(spread_bps_per_side=10, financing_annual=0.05)


def config(min_span: int, stop_atr):
    g_pool, n_pool, g_rows, n_rows, days = [], [], [], [], []
    for t in UNIVERSE:
        df = load_ftse.load(t)
        tr = simulate(df, k=0.5, min_span=min_span, stop_atr=stop_atr,
                      costs=CostModel(0.0, 0.0))
        g = np.array([x.ret for x in tr])
        nret = np.array([trade_return(x.direction, x.entry_price, x.exit_price, x.days, NET)
                         for x in tr])
        g_pool.extend(g.tolist()); n_pool.extend(nret.tolist())
        days.extend(x.days for x in tr)
        g_rows.append(float(np.prod(1 + g) - 1) if len(g) else 0.0)
        n_rows.append(float(np.prod(1 + nret) - 1) if len(nret) else 0.0)

    def stat(pool, rows):
        p = np.array(pool)
        pos, neg = p[p > 0].sum(), -p[p < 0].sum()
        return (len(p), 100 * (p > 0).mean(), 100 * p.mean(),
                pos / neg if neg > 0 else float("inf"),
                float(np.mean(days)) if days else 0.0,
                100 * np.median(rows), 100 * np.mean([r > 0 for r in rows]))

    return stat(g_pool, g_rows), stat(n_pool, n_rows)


def main() -> None:
    configs = [("pure  S=0 ", 0, None), ("pure  S=20", 20, None),
               ("floor S=20", 20, 2.0)]
    hdr = f"{'config':11} {'cost':5} {'trades':>7} {'win%':>6} {'mean%':>8} {'PF':>5} {'hold':>6} {'medTot%':>9} {'%prof':>6}"
    lines = ["=== AUTHOR-FAITHFUL PURE SAFETY-LINE EXIT (k=0.5) ===", hdr]
    for label, S, sa in configs:
        gross, net = config(S, sa)
        for cost_label, s in (("gross", gross), ("net", net)):
            lines.append(f"{label:11} {cost_label:5} {s[0]:>7} {s[1]:>6.1f} {s[2]:>8.3f} "
                         f"{s[3]:>5.2f} {s[4]:>6.1f} {s[5]:>9.1f} {s[6]:>6.0f}")
        lines.append("")
    report = "\n".join(lines)
    print(report)
    from pathlib import Path
    (Path(__file__).resolve().parent / "backtest_pure_results.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
