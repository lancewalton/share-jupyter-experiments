"""Run the full backtest over a representative FTSE universe: base case + k-sweep.

Costs: spread-bet wrapper — 10bps/side spread, 5%/yr financing, no stamp duty.
Pre-registered central parameters: k=0.5, 14-period ATR, >=3 touches, 2*ATR floor.
Outputs a text summary and per-ticker/k charts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import load_ftse
from backtest import CostModel, simulate

CHART_DIR = Path(__file__).resolve().parent / "charts"
UNIVERSE = ["AZN", "HSBA", "BP", "GSK", "DGE", "BATS", "ULVR", "RIO", "AAL", "BARC",
            "LLOY", "LGEN", "PRU", "NG", "SSE", "TSCO", "REL", "IMB", "BA", "RR"]
COSTS = CostModel(spread_bps_per_side=10, financing_annual=0.05)
SWEEP_K = [0.25, 0.5, 0.75, 1.0, 1.5]


def per_ticker(tkr: str, k: float):
    df = load_ftse.load(tkr)
    trades = simulate(df, k=k, costs=COSTS)
    rets = np.array([t.ret for t in trades])
    bh = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    total = float(np.prod(1 + rets) - 1) if len(rets) else 0.0
    row = {
        "tkr": tkr, "n": len(rets),
        "win": float((rets > 0).mean()) if len(rets) else float("nan"),
        "mean": float(rets.mean()) if len(rets) else float("nan"),
        "total": total, "bh": float(bh),
        "longs": sum(t.direction == "LONG" for t in trades),
    }
    return row, rets


def run_config(k: float):
    rows, pooled = [], []
    for t in UNIVERSE:
        row, rets = per_ticker(t, k)
        rows.append(row)
        pooled.extend(rets.tolist())
    return rows, np.array(pooled)


def aggregate(rows: list[dict], all_rets: np.ndarray) -> dict:
    pos = all_rets[all_rets > 0].sum()
    neg = -all_rets[all_rets < 0].sum()
    return {
        "n": int(sum(r["n"] for r in rows)),
        "win": float((all_rets > 0).mean()),
        "mean_ret": float(all_rets.mean()),
        "median_total": float(np.median([r["total"] for r in rows])),
        "pct_profitable": float(np.mean([r["total"] > 0 for r in rows])),
        "profit_factor": float(pos / neg) if neg > 0 else float("inf"),
        "median_bh": float(np.median([r["bh"] for r in rows])),
    }


def main() -> None:
    CHART_DIR.mkdir(exist_ok=True)
    lines: list[str] = []

    # Base case (k = 0.5), per-ticker detail.
    base_rows, pooled = run_config(0.5)

    lines.append("=== BASE CASE  k=0.5  (spread-bet costs) ===")
    lines.append(f"{'tkr':6} {'n':>4} {'L%':>4} {'win%':>5} {'mean%':>7} {'total%':>9} {'B&H%':>9}")
    for r in base_rows:
        lines.append(f"{r['tkr']:6} {r['n']:>4} {100*r['longs']/max(r['n'],1):>4.0f} "
                     f"{100*r['win']:>5.1f} {100*r['mean']:>7.2f} {100*r['total']:>9.1f} {100*r['bh']:>9.1f}")
    agg = aggregate(base_rows, pooled)
    lines.append("")
    lines.append(f"AGG: trades={agg['n']}  pooled win={100*agg['win']:.1f}%  "
                 f"mean ret/trade={100*agg['mean_ret']:.3f}%  profit factor={agg['profit_factor']:.2f}")
    lines.append(f"     median per-ticker total={100*agg['median_total']:.1f}%  "
                 f"% tickers profitable={100*agg['pct_profitable']:.0f}%  "
                 f"median B&H={100*agg['median_bh']:.1f}%")

    # Sweep.
    lines.append("")
    lines.append("=== k-SWEEP (aggregate) ===")
    lines.append(f"{'k':>5} {'trades':>7} {'win%':>6} {'mean/tr%':>9} {'medTot%':>9} {'%prof':>6} {'PF':>5}")
    sweep_summ = []
    for k in SWEEP_K:
        if k == 0.5:
            rows, pr = base_rows, pooled
        else:
            rows, pr = run_config(k)
        a = aggregate(rows, pr)
        sweep_summ.append((k, a))
        lines.append(f"{k:>5} {a['n']:>7} {100*a['win']:>6.1f} {100*a['mean_ret']:>9.3f} "
                     f"{100*a['median_total']:>9.1f} {100*a['pct_profitable']:>6.0f} {a['profit_factor']:>5.2f}")

    report = "\n".join(lines)
    print(report)
    (Path(__file__).resolve().parent / "backtest_results.txt").write_text(report + "\n")

    # Chart: base-case per-ticker total vs buy&hold.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(base_rows))
    ax1.bar(x - 0.2, [100 * r["total"] for r in base_rows], 0.4, label="strategy", color="#c0504d")
    ax1.bar(x + 0.2, [100 * r["bh"] for r in base_rows], 0.4, label="buy & hold", color="#4d79c0")
    ax1.set_xticks(x); ax1.set_xticklabels([r["tkr"] for r in base_rows], rotation=90, fontsize=7)
    ax1.axhline(0, color="k", lw=0.6); ax1.set_ylabel("total return %"); ax1.legend()
    ax1.set_title("k=0.5: strategy vs buy & hold (26y, per ticker)")

    ax2.plot([k for k, _ in sweep_summ], [100 * a["mean_ret"] for _, a in sweep_summ], "o-")
    ax2.axhline(0, color="k", lw=0.6); ax2.set_xlabel("k (ATR band)")
    ax2.set_ylabel("mean net return per trade %"); ax2.set_title("k-sweep: edge per trade")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "backtest.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("\nchart:", CHART_DIR / "backtest.png")


if __name__ == "__main__":
    main()
