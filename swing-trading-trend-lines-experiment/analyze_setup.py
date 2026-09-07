"""Does the entry geometry predict trade success?

Features at entry: direction-adjusted action & safety line gradients, their
convergence, the action/safety separation (channel width), and both line spans.
Outcome: gross trade return (raw edge, before the known cost drag).

Guard against data-snooping: fit/inspect on the EARLIER half of trades, then check
the SAME relationships on the LATER half. Anything that only shows in-sample is noise.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

import load_ftse
from backtest import CostModel, simulate, trade_return
from backtest_run import UNIVERSE

NET = CostModel(10, 0.05)
GEOM = ["action_slope", "safety_slope", "convergence", "separation",
        "action_span", "safety_span"]
FEATURES = GEOM + ["rel_volume"]
LABELS = {
    "action_slope": "action gradient (against-trade, ln/bar)",
    "safety_slope": "safety gradient (with-trade, ln/bar)",
    "convergence": "wedge convergence (ln/bar)",
    "separation": "action–safety separation (frac of price)",
    "action_span": "action line span (bars)",
    "safety_span": "safety line span (bars)",
    "rel_volume": "breakout volume / 20-bar median",
}


CACHE = Path(__file__).resolve().parent / "setup_trades_v2.parquet"


def collect() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    rows = []
    for t in UNIVERSE:
        df = load_ftse.load(t)
        for tr in simulate(df, k=0.5, min_span=0, costs=CostModel(0.0, 0.0)):
            d = 1.0 if tr.direction == "LONG" else -1.0
            a_sl, s_sl = d * tr.action_slope, d * tr.safety_slope
            rows.append({
                "ticker": t, "date": tr.entry_date, "gross": tr.ret,
                "net": trade_return(tr.direction, tr.entry_price, tr.exit_price, tr.days, NET),
                "action_slope": a_sl, "safety_slope": s_sl,
                "convergence": s_sl - a_sl, "separation": tr.separation,
                "action_span": tr.action_span, "safety_span": tr.safety_span,
                "rel_volume": tr.rel_volume,
            })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.to_parquet(CACHE)
    return df


def auc(y: np.ndarray, score: np.ndarray) -> float:
    order = score.argsort()
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def logistic_auc(feats, ins, oos):
    tr_X, te_X = ins[feats].to_numpy(), oos[feats].to_numpy()
    tr_y = (ins["gross"] > 0).to_numpy().astype(float)
    te_y = (oos["gross"] > 0).to_numpy().astype(float)
    mu, sd = tr_X.mean(0), tr_X.std(0) + 1e-9
    Xtr = np.c_[np.ones(len(tr_X)), (tr_X - mu) / sd]
    Xte = np.c_[np.ones(len(te_X)), (te_X - mu) / sd]

    def nll(w):
        z = Xtr @ w
        return np.mean(np.logaddexp(0, z) - tr_y * z) + 1e-3 * (w[1:] ** 2).sum()

    w = optimize.minimize(nll, np.zeros(Xtr.shape[1]), method="BFGS").x
    return auc(tr_y, Xtr @ w), auc(te_y, Xte @ w), dict(zip(["bias"] + list(feats), w)), Xte @ w


def main() -> None:
    df = collect().dropna(subset=FEATURES).reset_index(drop=True)
    mid = len(df) // 2
    ins, oos = df.iloc[:mid], df.iloc[mid:]
    base_in, base_out = (ins["gross"] > 0).mean(), (oos["gross"] > 0).mean()

    out = [f"Trades: {len(df)}  (in-sample {len(ins)} to {ins['date'].iloc[-1].date()}, "
           f"out-of-sample {len(oos)} from {oos['date'].iloc[0].date()})",
           f"Base gross win rate: in {100*base_in:.1f}%  out {100*base_out:.1f}%", "",
           "Spearman corr of feature vs GROSS return (rho, p):",
           f"{'feature':32} {'in-sample':>18} {'out-of-sample':>18}"]
    for f in FEATURES:
        ri, pi = stats.spearmanr(ins[f], ins["gross"])
        ro, po = stats.spearmanr(oos[f], oos["gross"])
        out.append(f"{f:32} {ri:>+8.3f} (p={pi:5.3f}) {ro:>+8.3f} (p={po:5.3f})")

    g_in, g_out, _, _ = logistic_auc(GEOM, ins, oos)
    a_in, a_out, coef, oos_score = logistic_auc(FEATURES, ins, oos)
    out += ["", f"Logistic AUC (out-of-sample):  geometry only {g_out:.3f}   "
            f"geometry + volume {a_out:.3f}   (marginal {a_out - g_out:+.3f})",
            "  (0.50 = no predictive information; in-sample geom+vol AUC "
            f"{a_in:.3f})",
            "  weights: " + "  ".join(f"{k}={v:+.2f}" for k, v in coef.items() if k != "bias")]

    # Volume on its own: does high breakout volume lift the RETURN (magnitude)?
    o = oos.copy()
    o["vdec"] = pd.qcut(o["rel_volume"], 5, labels=False, duplicates="drop")
    out += ["", "OUT-OF-SAMPLE by breakout-volume quintile (5 = highest volume):",
            f"{'quintile':>8} {'n':>5} {'win%':>6} {'gross/tr%':>10} {'net/tr%':>9} {'medVol×':>8}"]
    for d, g in o.groupby("vdec"):
        out.append(f"{int(d)+1:>8} {len(g):>5} {100*(g['gross']>0).mean():>6.1f} "
                   f"{100*g['gross'].mean():>10.3f} {100*g['net'].mean():>9.3f} {g['rel_volume'].median():>8.2f}")

    # The decisive test: does selecting favourable geometry make money OUT-OF-SAMPLE?
    o = oos.copy()
    o["dec"] = pd.qcut(oos_score, 10, labels=False, duplicates="drop")
    out += ["", "OUT-OF-SAMPLE return by predicted-score decile (10 = best geometry):",
            f"{'decile':>6} {'n':>5} {'win%':>6} {'gross/tr%':>10} {'net/tr%':>9}"]
    for d, g in o.groupby("dec"):
        out.append(f"{int(d)+1:>6} {len(g):>5} {100*(g['gross']>0).mean():>6.1f} "
                   f"{100*g['gross'].mean():>10.3f} {100*g['net'].mean():>9.3f}")
    top = o[oos_score >= np.quantile(oos_score, 0.9)]
    out += ["", f"Top-decile geometry, OOS: n={len(top)}  win={100*(top['gross']>0).mean():.1f}%  "
            f"gross/tr={100*top['gross'].mean():+.3f}%  net/tr={100*top['net'].mean():+.3f}%"]

    report = "\n".join(out)
    print(report)
    (Path(__file__).resolve().parent / "setup_analysis.txt").write_text(report + "\n")

    # Decile win-rate profiles, in vs out of sample, per feature.
    ncol = 3
    nrow = -(-len(FEATURES) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 4 * nrow))
    for ax in axes.flat[len(FEATURES):]:
        ax.axis("off")
    for ax, f in zip(axes.flat, FEATURES):
        edges = np.quantile(ins[f], np.linspace(0, 1, 11))
        edges[0], edges[-1] = -np.inf, np.inf
        for part, lab, style in ((ins, "in-sample", "o-"), (oos, "out-of-sample", "s--")):
            b = pd.cut(part[f], np.unique(edges))
            g = part.groupby(b, observed=True)
            centres = g[f].median().to_numpy()
            wr = 100 * (g["gross"].apply(lambda s: (s > 0).mean())).to_numpy()
            ax.plot(centres, wr, style, label=lab, ms=4)
        ax.axhline(100 * base_in, color="#888", lw=0.8, ls=":")
        ax.set_title(LABELS[f], fontsize=9)
        ax.set_ylabel("gross win rate %", fontsize=8); ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Does entry geometry predict a winning trade? (dotted = base win rate)", fontsize=12)
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parent / "charts" / "setup_analysis.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("chart: charts/setup_analysis.png")


if __name__ == "__main__":
    main()
