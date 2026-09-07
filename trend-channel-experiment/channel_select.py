"""Is the channel a good NAME SELECTOR (as opposed to a timing rule)?

Pure selection, no dip-timing: hold an equal-weight book of every name that is
CURRENTLY in a qualifying channel (causal: trailing-window gradient >= G_MIN, R^2
>= R2_MIN, optional min width), rebalancing as names enter/leave qualification.
No buying the bottom, no selling the top - just own what is in an established rising
channel. Compared to equal-weight buy-and-hold of the whole universe.

Also reports the ex-post (look-ahead) version - names that EVER qualify, held the
whole period - to show how much of any edge is hindsight vs causal.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u channel_select.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import load_ftse
from channel_entry_sweep import precompute_rich

HERE = Path(__file__).resolve().parent
BARS_YR = 252
G_MIN = 0.10
R2_MIN = 0.80
SPREAD = 10 / 1e4
WINDOWS = [100, 250]
W_MINS = [0.0, 0.15, 0.25]
OUT = HERE / "channel_select_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def stats(daily: np.ndarray, yrs: float) -> tuple[float, float]:
    total = np.prod(1 + daily) - 1
    cagr = (1 + total) ** (1 / yrs) - 1 if total > -1 else float("nan")
    sh = float(np.mean(daily) / np.std(daily) * np.sqrt(BARS_YR)) if np.std(daily) > 0 else 0.0
    return cagr, sh


def main() -> None:
    names = load_ftse.universe()
    say(f"Loading {len(names)} names...")
    closes = {}
    for t in names:
        try:
            closes[t] = load_ftse.load(t)["close"]
        except Exception:
            pass
    dates = pd.DatetimeIndex(sorted(set().union(*[set(closes[t].index) for t in closes])))
    yrs = len(dates) / BARS_YR
    RETS = pd.DataFrame(closes).reindex(dates).pct_change()
    uni = RETS.mean(axis=1).fillna(0.0).to_numpy()
    uni_cagr, uni_sh = stats(uni, yrs)
    say(f"{len(closes)} names. Universe equal-weight B&H: CAGR {100*uni_cagr:+.2f}%  Sharpe {uni_sh:+.2f}\n")

    for Lw in WINDOWS:
        say("#" * 92)
        say(f"# CHANNEL-SELECTION (hold while in a qualifying channel), L={Lw}  vs universe B&H")
        say("#" * 92)
        ANN, RELW, R2 = {}, {}, {}
        for t, df in ((t, load_ftse.load(t)) for t in closes):
            a = precompute_rich(df, Lw)
            if a is None:
                continue
            _, _, ann, relw, r2 = a
            idx = df.index
            ANN[t] = pd.Series(ann, index=idx)
            RELW[t] = pd.Series(relw, index=idx)
            R2[t] = pd.Series(r2, index=idx)
        ANN = pd.DataFrame(ANN).reindex(dates)
        RELW = pd.DataFrame(RELW).reindex(dates)
        R2 = pd.DataFrame(R2).reindex(dates)
        base_ok = (ANN >= G_MIN) & (R2 >= R2_MIN)

        say(f"{'w_min':>6} {'avgHeld':>8} {'turnovr/yr':>11} {'CAGR(gross)':>12} {'CAGR(net)':>11} "
            f"{'Sharpe':>7} {'exBH_CAGR':>10}  vsUniv")
        for w in W_MINS:
            qual = (base_ok & (RELW >= w)).fillna(False)
            held = qual.shift(1).fillna(False)
            n_held = held.sum(axis=1)
            member_ret = RETS.where(held).mean(axis=1).fillna(0.0)
            changes = held.astype(int).diff().abs().sum(axis=1)
            cost = (SPREAD * changes / n_held.replace(0, np.nan)).fillna(0.0)
            gross = member_ret.to_numpy()
            net = (member_ret - cost).to_numpy()
            g_cagr, _ = stats(gross, yrs)
            n_cagr, n_sh = stats(net, yrs)
            turnover = float(changes.sum() / n_held.replace(0, np.nan).mean() / yrs)
            # ex-post: names that EVER qualify, held whole period (look-ahead)
            ever = qual.any(axis=0)
            ex = RETS.loc[:, ever[ever].index].mean(axis=1).fillna(0.0).to_numpy()
            ex_cagr, _ = stats(ex, yrs)
            beat = "WINS" if n_cagr > uni_cagr else "loses"
            say(f"{w:>6.2f} {n_held.mean():>8.0f} {turnover:>11.1f} {100*g_cagr:>11.2f}% "
                f"{100*n_cagr:>10.2f}% {n_sh:>7.2f} {100*ex_cagr:>9.2f}%  {beat}")
        say("")

    say("exBH_CAGR = hold-forever the names that EVER qualify (look-ahead upper bound).")
    say(f"universe B&H CAGR {100*uni_cagr:.2f}%, Sharpe {uni_sh:.2f}.")
    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
