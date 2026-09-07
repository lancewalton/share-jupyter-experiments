"""If strong breakouts revert, two strategies follow: FADE strong breakouts, or
CONTINUE only weak ones. Both reduce to one question — the raw forward return in
the break direction as a function of breakout extent — so test it cleanly with a
fixed-horizon event study, free of the trend-line trailing-stop trade.

At each breakout (from the continuation cache) enter at the recorded entry price
and hold a fixed horizon H; forward-in-break-direction return is
  fwd = sign * (close[entry+H] / entry_px - 1)
Reversion (fade) return is -fwd. Then:
  - Spearman(break_extent, fwd): negative => strong breaks revert.
  - By extent tercile: is the weak bucket's continuation positive? the strong
    bucket's fade positive?
  - Build both books (continue-weak, fade-strong), net of spread-bet costs, and
    stress the better one.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u event_study.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import load_ftse
from backtest import CostModel, trade_return
from backtest_run import UNIVERSE
from continuation_study import collect
from regime_gate_ftse import (OOS_YEAR, market_over_hold, metrics,
                              per_ticker_totals, year_block_boot)

HERE = Path(__file__).resolve().parent
HORIZONS = [5, 10, 20]
NET = CostModel(10, 0.05)
OUT = HERE / "event_study_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def with_forward(tr: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    dfs = {t: load_ftse.load(t) for t in UNIVERSE}
    closes = {t: d["close"].to_numpy() for t, d in dfs.items()}
    dates = {t: d.index for t, d in dfs.items()}
    pos = {t: {d: i for i, d in enumerate(dfs[t].index)} for t in dfs}
    mkt_idx = (1 + pd.DataFrame({t: d["close"] for t, d in dfs.items()}).sort_index()
               .pct_change().mean(axis=1)).cumprod()
    bh = {t: float(d["close"].iloc[-1] / d["close"].iloc[0] - 1) for t, d in dfs.items()}

    sign = np.where(tr.direction.to_numpy() == "LONG", 1.0, -1.0)
    tr = tr.copy()
    tr["sign"] = sign
    for H in HORIZONS:
        fwd = np.full(len(tr), np.nan)
        exit_dt = np.full(len(tr), np.datetime64("NaT", "ns"))
        exit_px = np.full(len(tr), np.nan)
        for r, (t, ed, epx, s) in enumerate(zip(tr.ticker, tr.entry, tr.entry_px, sign)):
            i = pos[t].get(ed)
            if i is None or i + H >= len(closes[t]):
                continue
            cx = closes[t][i + H]
            fwd[r] = s * (cx / epx - 1.0)
            exit_dt[r] = dates[t][i + H].to_datetime64()
            exit_px[r] = cx
        tr[f"fwd_{H}"] = fwd
        tr[f"exit_{H}"] = exit_dt
        tr[f"exitpx_{H}"] = exit_px
    return tr, mkt_idx, bh


def book(tr: pd.DataFrame, H: int, mode: str) -> pd.DataFrame:
    """A fixed-horizon book. mode 'continue' trades the break direction, 'fade'
    the opposite. Costs applied via trade_return."""
    d = tr.dropna(subset=[f"fwd_{H}", f"exitpx_{H}"]).copy()
    flip = 1.0 if mode == "continue" else -1.0
    trade_dir = np.where((d["sign"].to_numpy() * flip) > 0, "LONG", "SHORT")
    days = (d[f"exit_{H}"] - d.entry).dt.days.clip(lower=1).to_numpy()
    gross = np.array([trade_return(td, e, x, dd, CostModel(0.0, 0.0))
                      for td, e, x, dd in zip(trade_dir, d.entry_px, d[f"exitpx_{H}"], days)])
    net = np.array([trade_return(td, e, x, dd, NET)
                    for td, e, x, dd in zip(trade_dir, d.entry_px, d[f"exitpx_{H}"], days)])
    return pd.DataFrame({
        "ticker": d.ticker.to_numpy(), "entry": d.entry.to_numpy(), "exit": d[f"exit_{H}"].to_numpy(),
        "direction": trade_dir, "gross": gross, "net_10": net,
        "year": d.year.to_numpy(), "break_extent": d.break_extent.to_numpy(),
        "vol_state": d.vol_state.to_numpy(),
    })


def stress(tr: pd.DataFrame, mkt_idx, bh, name: str) -> None:
    tr = tr.copy().reset_index(drop=True)
    if len(tr) < 50:
        say(f"\n===== STRESS: {name} -- {len(tr)} trades, skipped ====="); return
    say(f"\n===== STRESS: {name}  (n={len(tr)}, names={tr.ticker.nunique()}, "
        f"years {tr.year.min()}-{tr.year.max()}) =====")
    for col, lab in [("gross", "gross"), ("net_10", "net(10bps+5%)")]:
        m, lo, hi = year_block_boot(tr, col)
        star = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
        say(f"  year-block bootstrap {lab:16s}: mean/tr {100*m:+.3f}%  "
            f"95% CI [{100*lo:+.3f}%, {100*hi:+.3f}%]{star}")
    sign = np.where(tr.direction == "LONG", 1.0, -1.0)
    tr["alpha"] = tr.gross.to_numpy() - sign * market_over_hold(tr, mkt_idx)
    m, lo, hi = year_block_boot(tr, "alpha")
    star = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
    say(f"  beta-neutral alpha (gross)      : mean/tr {100*m:+.3f}%  "
        f"95% CI [{100*lo:+.3f}%, {100*hi:+.3f}%]{star}")
    say("  " + per_ticker_totals(tr, "net_10", bh))


def main() -> None:
    say("Loading cached breakouts and computing forward returns...")
    raw, _, _ = collect()
    tr, mkt_idx, bh = with_forward(raw)
    tr = tr.sort_values("entry").reset_index(drop=True)
    mid = len(tr) // 2
    ins, oos = tr.iloc[:mid], tr.iloc[mid:]

    say("\n" + "#" * 72)
    say("# Does break_extent predict the forward BREAK-direction return? (negative => revert)")
    say("#" * 72)
    for H in HORIZONS:
        def sp(part):
            m = part[["break_extent", f"fwd_{H}"]].dropna()
            r, p = stats.spearmanr(m.break_extent, m[f"fwd_{H}"])
            return r, p, len(m)
        ri, pi, _ = sp(ins); ro, po, no = sp(oos)
        say(f"  H={H:>2}d  IS rho {ri:+.3f}(p{pi:4.2f})  OOS rho {ro:+.3f}(p{po:4.2f})  n_OOS={no}")

    say("\n" + "#" * 72)
    say("# Forward break-direction return by extent tercile (gross %, mean)")
    say("#" * 72)
    for H in HORIZONS:
        d = tr.dropna(subset=[f"fwd_{H}"]).copy()
        d["terc"] = pd.qcut(d.break_extent, 3, labels=["weak", "mid", "strong"])
        say(f"  H={H:>2}d  " + "  ".join(
            f"{t}: {100*g[f'fwd_{H}'].mean():+.3f}% (win {100*(g[f'fwd_{H}']>0).mean():.0f}%, n={len(g)})"
            for t, g in d.groupby("terc", observed=True)))

    # Build the two candidate books at each horizon, net, and note the best.
    say("\n" + "#" * 72)
    say("# Candidate books (net of 10bps/side + 5%/yr), by horizon")
    say("#" * 72)
    best = None
    for H in HORIZONS:
        be_lo = tr.break_extent.quantile(1/3)
        be_hi = tr.break_extent.quantile(2/3)
        cont_weak = book(tr[tr.break_extent <= be_lo], H, "continue")
        fade_strong = book(tr[tr.break_extent >= be_hi], H, "fade")
        say(f"\n  H={H}d")
        say(f"    continue-weak   gross {metrics(cont_weak,'gross')} | net {metrics(cont_weak,'net_10')}")
        say(f"    fade-strong     gross {metrics(fade_strong,'gross')} | net {metrics(fade_strong,'net_10')}")
        for label, bk in [(f"continue-weak H{H}", cont_weak), (f"fade-strong H{H}", fade_strong)]:
            g = bk.net_10.mean()
            if best is None or g > best[0]:
                best = (g, label, bk)

    say(f"\nBest net book: {best[1]} (net mean/tr {100*best[0]:+.3f}%)")
    stress(best[2], mkt_idx, bh, best[1])

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
