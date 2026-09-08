"""Trade a rising regression channel: go long near the bottom, sell near the top,
and ask whether it beats buy-and-hold. Causal walk-forward over all FTSE names.

At each bar, fit a channel on the trailing L bars (ending at the decision bar).
Long-only state machine, acting at the next open:
  - enter when the channel qualifies (annual gradient >= G_MIN, R^2 >= R2_MIN) and
    the close is in the bottom ENTRY_FRAC of the channel;
  - exit at the top EXIT_FRAC, or on a close below the lower band (break), or when
    the gradient turns negative.
Costs: SPREAD bps per side + FINANCING per year while held. Compared per name to
buy-and-hold over the same span (total return and Sharpe), pooled across the 120
names - no cherry-picking the good-looking charts.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u channel_backtest.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import load_ftse
from channel import fit_channel

HERE = Path(__file__).resolve().parent
L = 250
G_MIN = 0.10          # min annualised gradient
R2_MIN = 0.80
K = 2.0
ENTRY_FRAC = 0.25
EXIT_FRAC = 0.25
SPREAD = 10 / 1e4     # per side
FIN = 0.05            # per year while held
BARS_YR = 252
OUT = HERE / "channel_backtest_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def run_ticker(df: pd.DataFrame, Lw: int, spread: float, fin: float) -> dict | None:
    close = df["close"].to_numpy()
    op = df["open"].to_numpy()
    dates = df.index
    logc = np.log(close)
    n = len(df)
    if n < Lw + 30:
        return None

    daily = np.zeros(n)          # net strategy daily return
    pos = False
    entry_idx = entry_px = None
    trades = []
    for i in range(Lw, n - 1):
        ch = fit_channel(logc[i - Lw + 1:i + 1])
        xL = Lw - 1
        lo, up = ch.lower(xL, K), ch.upper(xL, K)
        width = up - lo
        ann = ch.annual_gradient(BARS_YR)
        qualifies = ann >= G_MIN and ch.r2 >= R2_MIN and width > 0
        c = close[i]
        if not pos:
            if qualifies and c <= lo + ENTRY_FRAC * width:
                pos, entry_idx, entry_px = True, i + 1, op[i + 1]
        else:
            if (c >= up - EXIT_FRAC * width) or (c < lo) or (ann < 0):
                x_idx = i + 1
                days = max(1, (dates[x_idx] - dates[entry_idx]).days)
                gross = op[x_idx] / entry_px - 1.0
                trades.append({"gross": gross, "net": gross - 2 * spread - fin * days / 365,
                               "days": (x_idx - entry_idx)})
                # fill daily returns for the held span
                for d in range(entry_idx, x_idx + 1):
                    if d == entry_idx:
                        r = close[d] / entry_px - 1.0
                    elif d == x_idx:
                        r = op[x_idx] / close[d - 1] - 1.0
                    else:
                        r = close[d] / close[d - 1] - 1.0
                    daily[d] += r - fin / 365
                daily[entry_idx] -= spread
                daily[x_idx] -= spread
                pos = False
    if pos:  # close at final close
        x_idx = n - 1
        for d in range(entry_idx, x_idx + 1):
            r = (close[d] / entry_px - 1.0) if d == entry_idx else (close[d] / close[d - 1] - 1.0)
            daily[d] += r - fin / 365
        daily[entry_idx] -= spread
        days = max(1, (dates[x_idx] - dates[entry_idx]).days)
        trades.append({"gross": close[x_idx] / entry_px - 1.0,
                       "net": close[x_idx] / entry_px - 1.0 - spread - fin * days / 365,
                       "days": x_idx - entry_idx})

    strat_d = daily[Lw:]
    bh_d = np.concatenate(([0.0], close[Lw + 1:] / close[Lw:-1] - 1.0))
    tim = float(np.mean(strat_d != 0))
    def sharpe(r):
        s = np.std(r)
        return float(np.mean(r) / s * np.sqrt(BARS_YR)) if s > 0 else 0.0
    return {
        "strat_total": float(np.prod(1 + strat_d) - 1), "bh_total": float(np.prod(1 + bh_d) - 1),
        "strat_sharpe": sharpe(strat_d), "bh_sharpe": sharpe(bh_d),
        "n_trades": len(trades), "tim": tim,
        "trade_net": [t["net"] for t in trades], "trade_gross": [t["gross"] for t in trades],
    }


def summarise(rows: list[dict], label: str) -> None:
    rows = [r for r in rows if r and r["n_trades"] > 0]
    if not rows:
        say(f"\n{label}: no channel trades anywhere."); return
    st = np.array([r["strat_total"] for r in rows])
    bh = np.array([r["bh_total"] for r in rows])
    ss = np.array([r["strat_sharpe"] for r in rows])
    bs = np.array([r["bh_sharpe"] for r in rows])
    tim = np.array([r["tim"] for r in rows])
    beat_ret = np.mean(st > bh)
    beat_sh = np.mean(ss > bs)
    all_net = np.array([x for r in rows for x in r["trade_net"]])
    all_gr = np.array([x for r in rows for x in r["trade_gross"]])
    pf = all_net[all_net > 0].sum() / -all_net[all_net < 0].sum() if (all_net < 0).any() else np.inf
    say(f"\n{label}: {len(rows)} names traded, {len(all_net)} trades")
    say(f"  per-trade: gross {100*all_gr.mean():+.3f}%  net {100*all_net.mean():+.3f}%  "
        f"win {100*(all_net>0).mean():.1f}%  PF {pf:.2f}")
    say(f"  total return (median): strategy {100*np.median(st):+.1f}%  vs buy&hold {100*np.median(bh):+.1f}%")
    say(f"  % names beating B&H:  on total return {100*beat_ret:.0f}%   on Sharpe {100*beat_sh:.0f}%")
    say(f"  Sharpe (median): strategy {np.median(ss):+.2f}  vs buy&hold {np.median(bs):+.2f}")
    say(f"  time in market (median): {100*np.median(tim):.0f}%")


def main() -> None:
    names = load_ftse.universe()
    say(f"Loading {len(names)} FTSE names...")
    dfs = {}
    for t in names:
        try:
            d = load_ftse.load(t)
            if len(d) > L + 60:
                dfs[t] = d
        except Exception:
            pass
    say(f"{len(dfs)} usable.")

    say("\n" + "#" * 68)
    say(f"# CHANNEL TRADING vs BUY & HOLD  (L={L}, G_MIN={G_MIN}, R2_MIN={R2_MIN})")
    say("#" * 68)
    net_rows = [run_ticker(d, L, SPREAD, FIN) for d in dfs.values()]
    gross_rows = [run_ticker(d, L, 0.0, 0.0) for d in dfs.values()]
    summarise(gross_rows, "GROSS (no costs)")
    summarise(net_rows, "NET (10bps/side + 5%/yr)")

    # top names by strategy total (net) and their B&H, to see where it works
    traded = [(t, r) for t, r in zip(dfs, net_rows) if r and r["n_trades"] > 0]
    traded.sort(key=lambda tr: tr[1]["strat_total"], reverse=True)
    say("\n  top 8 names by net strategy total (strat% / B&H% / trades / TIM%):")
    for t, r in traded[:8]:
        say(f"    {t:6} {100*r['strat_total']:+8.0f} / {100*r['bh_total']:+8.0f} / "
            f"{r['n_trades']:>3} / {100*r['tim']:.0f}")

    say("\n" + "#" * 68)
    say("# WINDOW-LENGTH SENSITIVITY (net) — median totals & % beating B&H")
    say("#" * 68)
    for Lw in (120, 250, 500):
        rows = [run_ticker(d, Lw, SPREAD, FIN) for d in dfs.values()]
        rr = [r for r in rows if r and r["n_trades"] > 0]
        if not rr:
            say(f"  L={Lw}: no trades"); continue
        st = np.array([r["strat_total"] for r in rr]); bh = np.array([r["bh_total"] for r in rr])
        ss = np.array([r["strat_sharpe"] for r in rr]); bs = np.array([r["bh_sharpe"] for r in rr])
        say(f"  L={Lw:>3}: names {len(rr)}  median strat {100*np.median(st):+.0f}% vs B&H "
            f"{100*np.median(bh):+.0f}%  beat-ret {100*np.mean(st>bh):.0f}%  "
            f"beat-Sharpe {100*np.mean(ss>bs):.0f}%  medSharpe {np.median(ss):+.2f}/{np.median(bs):+.2f}")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
