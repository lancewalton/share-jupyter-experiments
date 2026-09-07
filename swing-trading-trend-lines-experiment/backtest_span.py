"""Option (b): min bar-span on the action/safety lines.

Pre-registered point: k=0.5, S=20 trading days. Then an S-sensitivity sweep
(the verdict rests on S=20; the sweep only shows whether it sits on a plateau
or a spike). Reports NET (spread-bet costs) and GROSS (zero-cost) side by side,
so we can see whether any raw edge emerges at any span.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import load_ftse
from backtest import CostModel, simulate, trade_return
from backtest_run import UNIVERSE

CHART_DIR = Path(__file__).resolve().parent / "charts"
NET = CostModel(spread_bps_per_side=10, financing_annual=0.05)
GROSS = CostModel(0.0, 0.0)
SWEEP_S = [5, 10, 20, 40, 60]


def _stats(pooled: list[float], rows: list[float], days: list[int]) -> dict:
    p = np.array(pooled)
    pos, neg = p[p > 0].sum(), -p[p < 0].sum()
    return {
        "n": len(p), "win": float((p > 0).mean()), "mean": float(p.mean()),
        "pf": float(pos / neg) if neg > 0 else float("inf"),
        "med_total": float(np.median(rows)), "pct_prof": float(np.mean([r > 0 for r in rows])),
        "hold": float(np.mean(days)) if days else 0.0,
    }


def config(min_span: int):
    """Simulate once per ticker (gross); derive both gross and net stats."""
    g_pool, n_pool, g_rows, n_rows, days = [], [], [], [], []
    for t in UNIVERSE:
        df = load_ftse.load(t)
        tr = simulate(df, k=0.5, min_span=min_span, costs=GROSS)
        g = np.array([x.ret for x in tr])
        n = np.array([trade_return(x.direction, x.entry_price, x.exit_price, x.days, NET)
                      for x in tr])
        g_pool.extend(g.tolist()); n_pool.extend(n.tolist())
        days.extend(x.days for x in tr)
        g_rows.append(float(np.prod(1 + g) - 1) if len(g) else 0.0)
        n_rows.append(float(np.prod(1 + n) - 1) if len(n) else 0.0)
    return _stats(g_pool, g_rows, days), _stats(n_pool, n_rows, days)


def main() -> None:
    CHART_DIR.mkdir(exist_ok=True)
    out = ["=== OPTION B: min-span sweep (k=0.5).  Pre-registered point: S=20 ===",
           f"{'S':>4} {'cost':>5} {'trades':>7} {'win%':>6} {'mean%':>8} {'PF':>5} "
           f"{'hold_d':>7} {'medTot%':>9} {'%prof':>6}"]
    net_pf, gross_pf, net_mean, gross_mean = [], [], [], []
    for S in SWEEP_S:
        gross, net = config(S)
        gross_pf.append(gross["pf"]); gross_mean.append(gross["mean"])
        net_pf.append(net["pf"]); net_mean.append(net["mean"])
        for label, a in (("gross", gross), ("net", net)):
            out.append(f"{S:>4} {label:>5} {a['n']:>7} {100*a['win']:>6.1f} "
                       f"{100*a['mean']:>8.3f} {a['pf']:>5.2f} {a['hold']:>7.1f} "
                       f"{100*a['med_total']:>9.1f} {100*a['pct_prof']:>6.0f}")
        out.append("")

    report = "\n".join(out)
    print(report)
    (Path(__file__).resolve().parent / "backtest_span_results.txt").write_text(report + "\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(SWEEP_S, gross_pf, "o-", label="gross")
    ax1.plot(SWEEP_S, net_pf, "s--", label="net")
    ax1.axhline(1.0, color="k", lw=0.8, ls=":")
    ax1.axvline(20, color="#888", lw=0.8); ax1.set_xlabel("min span S (days)")
    ax1.set_ylabel("profit factor"); ax1.set_title("Profit factor vs span (1.0 = breakeven)")
    ax1.legend()
    ax2.plot(SWEEP_S, [100*m for m in gross_mean], "o-", label="gross")
    ax2.plot(SWEEP_S, [100*m for m in net_mean], "s--", label="net")
    ax2.axhline(0, color="k", lw=0.8, ls=":"); ax2.axvline(20, color="#888", lw=0.8)
    ax2.set_xlabel("min span S (days)"); ax2.set_ylabel("mean return/trade %")
    ax2.set_title("Edge per trade vs span"); ax2.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "backtest_span.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("chart:", CHART_DIR / "backtest_span.png")


if __name__ == "__main__":
    main()
