"""Does the RELATIVE duration of the action vs safety line predict a valid entry?

Hypothesis (Lance): a breakout of a line that has been "in operation" much
longer/shorter than the opposite (safety) line may be more or less likely to be
genuine vs a false breakout that reverts. Test whether relative duration predicts
(a) win probability and - the real prize the geometry study missed - (b) PAYOFF.

"Duration" is ambiguous, so we compute several, per line (action & safety):
  span_ab           = b - a           (between the two defining hull vertices)
  age_from_a        = signal_bar - a  (inception -> breakout)
  age_from_b        = signal_bar - b  (since the most recent defining vertex)
  touch_span        = last_touch - first_touch  (all touches, incl. later ones)
  age_since_last    = signal_bar - last_touch
  n_touches
Relative feature = ln(action_X / safety_X). Sign of any effect is left to data.

Guarded in-sample(early)/out-of-sample(late) split, like analyze_setup.py. Any
payoff-predictive feature is gated and run through the FTSE stress harness.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u line_duration_study.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import load_ftse
from backtest import CostModel, simulate
from backtest_run import UNIVERSE
from regime_gate_ftse import (OOS_YEAR, add_costs, market_over_hold, metrics,
                              per_ticker_totals, vol_state, year_block_boot)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "duration_trades.parquet"
OUT = HERE / "line_duration_study_results.txt"
PRIMARY = "rel_age_from_a"      # pre-registered primary duration definition
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def collect() -> tuple[pd.DataFrame, pd.Series, dict]:
    dfs = {t: load_ftse.load(t) for t in UNIVERSE}
    closes = pd.DataFrame({t: d["close"] for t, d in dfs.items()}).sort_index()
    mkt_idx = (1 + closes.pct_change().mean(axis=1)).cumprod()
    bh = {t: float(d["close"].iloc[-1] / d["close"].iloc[0] - 1) for t, d in dfs.items()}
    if CACHE.exists():
        return pd.read_parquet(CACHE), mkt_idx, bh
    rows = []
    for t, df in dfs.items():
        vs = vol_state(df)
        for tr in simulate(df, k=0.5, min_span=0, costs=CostModel(0.0, 0.0)):
            rows.append({
                "ticker": t, "entry": tr.entry_date, "exit": tr.exit_date,
                "direction": tr.direction, "entry_px": tr.entry_price,
                "exit_px": tr.exit_price, "days": tr.days, "gross": tr.ret,
                "year": tr.entry_date.year, "signal_bar": tr.signal_bar,
                "action_a": tr.action_a, "action_b": tr.action_b,
                "safety_a": tr.safety_a, "safety_b": tr.safety_b,
                "action_touches": tr.action_touches,
                "action_touch_first": tr.action_touch_first,
                "action_touch_last": tr.action_touch_last,
                "safety_touches": tr.safety_touches,
                "safety_touch_first": tr.safety_touch_first,
                "safety_touch_last": tr.safety_touch_last,
                "vol_state": float(vs.get(tr.entry_date, np.nan)),
            })
    tr = pd.DataFrame(rows).sort_values("entry").reset_index(drop=True)
    tr.to_parquet(CACHE)
    return tr, mkt_idx, bh


def _rl(a: pd.Series, b: pd.Series) -> np.ndarray:
    a, b = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    return np.where((a > 0) & (b > 0), np.log(np.where(b > 0, a, 1) / np.where(b > 0, b, 1)), np.nan)


DEFS = ["span_ab", "age_from_a", "age_from_b", "touch_span", "age_since_last", "n_touches"]


def add_features(tr: pd.DataFrame) -> pd.DataFrame:
    sb = tr.signal_bar
    tr["span_ab_A"], tr["span_ab_S"] = tr.action_b - tr.action_a, tr.safety_b - tr.safety_a
    tr["age_from_a_A"], tr["age_from_a_S"] = sb - tr.action_a, sb - tr.safety_a
    tr["age_from_b_A"], tr["age_from_b_S"] = sb - tr.action_b, sb - tr.safety_b
    tr["touch_span_A"] = np.where(tr.action_touch_last >= 0, tr.action_touch_last - tr.action_touch_first, np.nan)
    tr["touch_span_S"] = np.where(tr.safety_touch_last >= 0, tr.safety_touch_last - tr.safety_touch_first, np.nan)
    tr["age_since_last_A"] = np.where(tr.action_touch_last >= 0, sb - tr.action_touch_last, np.nan)
    tr["age_since_last_S"] = np.where(tr.safety_touch_last >= 0, sb - tr.safety_touch_last, np.nan)
    tr["n_touches_A"], tr["n_touches_S"] = tr.action_touches, tr.safety_touches
    for d in DEFS:
        tr[f"rel_{d}"] = _rl(tr[f"{d}_A"], tr[f"{d}_S"])
    tr["win"] = (tr.gross > 0).astype(float)
    return tr


def corr_table(ins: pd.DataFrame, oos: pd.DataFrame, feats: list[str]) -> None:
    say(f"{'feature':16} {'IS rho(gross)':>14} {'OOS rho(gross)':>15} "
        f"{'IS rho(win)':>13} {'OOS rho(win)':>13}   n_OOS")
    for f in feats:
        def sp(part, y):
            m = part[[f, y]].dropna()
            if len(m) < 30:
                return np.nan, np.nan, 0
            r, p = stats.spearmanr(m[f], m[y])
            return r, p, len(m)
        ri, pi, _ = sp(ins, "gross")
        ro, po, no = sp(oos, "gross")
        rwi, _, _ = sp(ins, "win")
        rwo, _, _ = sp(oos, "win")
        say(f"{f:16} {ri:>+9.3f}(p{pi:4.2f}) {ro:>+9.3f}(p{po:4.2f}) "
            f"{rwi:>+13.3f} {rwo:>+13.3f}   {no}")


def deciles(oos: pd.DataFrame, feat: str) -> None:
    o = oos[[feat, "gross", "net_10", "win"]].dropna().copy()
    if len(o) < 200:
        say(f"  (too few OOS rows for deciles on {feat})")
        return
    o["dec"] = pd.qcut(o[feat], 10, labels=False, duplicates="drop")
    say(f"\nOOS payoff by {feat} decile (1=most negative rel, 10=most positive):")
    say(f"{'dec':>4} {'n':>5} {'win%':>6} {'gross/tr%':>10} {'net/tr%':>9} {'medRel':>8}")
    for d, g in o.groupby("dec"):
        say(f"{int(d)+1:>4} {len(g):>5} {100*g.win.mean():>6.1f} "
            f"{100*g.gross.mean():>10.3f} {100*g.net_10.mean():>9.3f} {g[feat].median():>8.2f}")


def stress(tr: pd.DataFrame, mkt_idx: pd.Series, bh: dict, name: str) -> None:
    tr = tr.copy().reset_index(drop=True)
    if len(tr) == 0:
        say(f"\n===== STRESS: {name} -- empty =====")
        return
    say(f"\n===== STRESS: {name}  (n={len(tr)}, names={tr.ticker.nunique()}, "
        f"years {tr.year.min()}-{tr.year.max()}) =====")
    for col, lab in [("gross", "gross"), ("net_10", "net(10bps+5%)")]:
        m, lo, hi = year_block_boot(tr, col)
        star = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
        say(f"  year-block bootstrap {lab:16s}: mean/tr {100*m:+.3f}%  "
            f"95% CI [{100*lo:+.3f}%, {100*hi:+.3f}%]{star}")
    sign = np.where(tr.direction.to_numpy() == "LONG", 1.0, -1.0)
    tr["alpha"] = tr.gross.to_numpy() - sign * market_over_hold(tr, mkt_idx)
    m, lo, hi = year_block_boot(tr, "alpha")
    star = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
    say(f"  beta-neutral alpha (gross)      : mean/tr {100*m:+.3f}%  "
        f"95% CI [{100*lo:+.3f}%, {100*hi:+.3f}%]{star}")
    pre, post = tr[tr.year < OOS_YEAR], tr[tr.year >= OOS_YEAR]
    say(f"  OOS split @ {OOS_YEAR}: pre  net10 {metrics(pre,'net_10')}")
    say(f"  {'':16s}      post net10 {metrics(post,'net_10')}")
    say("  " + per_ticker_totals(tr, "net_10", bh))


def main() -> None:
    say("Collecting duration-instrumented trades (20 names)...")
    tr, mkt_idx, bh = collect()
    tr = add_costs(tr)
    tr = add_features(tr)
    say(f"{len(tr)} trades. safety-line touch coverage: "
        f"{100*(tr.safety_touches>0).mean():.0f}% have >=1 safety touch.")

    tr = tr.sort_values("entry").reset_index(drop=True)
    mid = len(tr) // 2
    ins, oos = tr.iloc[:mid], tr.iloc[mid:]
    say(f"\nsplit: in-sample {len(ins)} (to {ins.entry.iloc[-1].date()}), "
        f"out-of-sample {len(oos)} (from {oos.entry.iloc[0].date()})")
    say(f"base gross win rate: IS {100*ins.win.mean():.1f}%  OOS {100*oos.win.mean():.1f}%")
    say(f"base gross mean/tr:  IS {100*ins.gross.mean():+.3f}%  OOS {100*oos.gross.mean():+.3f}%")

    say("\n" + "#" * 78)
    say("# Spearman: does duration predict PAYOFF (gross) or just WIN? (IS vs OOS)")
    say("#" * 78)
    rel_feats = [f"rel_{d}" for d in DEFS]
    abs_feats = ["age_from_a_A", "age_from_a_S", "span_ab_A", "span_ab_S", "n_touches_S"]
    corr_table(ins, oos, rel_feats + abs_feats)

    say("\n" + "#" * 78)
    say(f"# OOS decile payoff for the primary ({PRIMARY}) and touch/span relatives")
    say("#" * 78)
    for f in [PRIMARY, "rel_span_ab", "rel_touch_span", "rel_n_touches"]:
        deciles(oos, f)

    # Gate on the primary: does the favourable tail clear costs? Test both tails,
    # believe only one consistent IS+OOS. Favourable tail = higher OOS gross decile.
    say("\n" + "#" * 78)
    say(f"# GATE on {PRIMARY}: keep the top/bottom third, full-sample + stress")
    say("#" * 78)
    q = tr[PRIMARY].dropna()
    lo_thr, hi_thr = q.quantile(1/3), q.quantile(2/3)
    top = tr[tr[PRIMARY] >= hi_thr]
    bot = tr[tr[PRIMARY] <= lo_thr]
    for lab, book in [("TOP third (action longer-lived)", top),
                      ("BOTTOM third (action shorter-lived)", bot)]:
        say(f"\n{lab}:")
        say(f"    gross {metrics(book,'gross')}")
        say(f"    net10 {metrics(book,'net_10')}")
    # stress whichever third is better gross
    better = top if top.gross.mean() >= bot.gross.mean() else bot
    stress(better, mkt_idx, bh, f"{PRIMARY} favourable third (best gross)")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
