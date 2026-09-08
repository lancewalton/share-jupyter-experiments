"""Fix 2: replace the hard sell-at-the-top exit with a trailing RATCHET so winners
ride the trend instead of being cut at the channel top.

Entry is unchanged (qualifying rising channel, close in the bottom quarter). We
compare exit policies:
  top          - baseline: sell in the top quarter of the channel.
  ratchet_width- once price reaches the top quarter, arm a stop that tracks the
                 (rising) resistance line at STOP_FRAC of the channel width below
                 it, ratcheting up; exit on a close through it (or below the band).
  ratchet_atr  - Chandelier: stop = highest close since entry - M*ATR, ratcheting.
Any policy also exits if the channel gradient turns negative (trend broken).

Rolling channels are precomputed once per name; the policies are cheap walks over
them, so the sweep is fast. Compared per name to buy-and-hold, all 120 FTSE names.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u channel_ratchet.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import load_ftse
from channel import fit_channel

HERE = Path(__file__).resolve().parent
L = 250
G_MIN, R2_MIN, K = 0.10, 0.80, 2.0
ENTRY_FRAC, ARM_FRAC = 0.25, 0.25
BARS_YR = 252
SPREAD, FIN = 10 / 1e4, 0.05
ATR_P = 22
OUT = HERE / "channel_ratchet_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def atr(df: pd.DataFrame, period: int) -> np.ndarray:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().bfill().to_numpy()


def precompute(df: pd.DataFrame, Lw: int):
    close = df["close"].to_numpy()
    n = len(df)
    if n < Lw + 30:
        return None
    logc = np.log(close)
    up = np.full(n, np.nan); lo = np.full(n, np.nan); ann = np.full(n, np.nan)
    for i in range(Lw, n):
        ch = fit_channel(logc[i - Lw + 1:i + 1])
        x = Lw - 1
        up[i], lo[i] = ch.upper(x, K), ch.lower(x, K)
        ann[i] = ch.annual_gradient(BARS_YR)
    return up, lo, ann


def run_policy(df, arrays, mode, stop_frac=0.5, m_atr=3.0) -> dict:
    up, lo, ann = arrays
    close = df["close"].to_numpy(); op = df["open"].to_numpy(); dates = df.index
    a = atr(df, ATR_P)
    n = len(df)
    daily = np.zeros(n)
    trades = []
    pos = False
    entry_idx = entry_px = None
    armed = False; stop = -np.inf; peak = -np.inf

    def close_trade(x_idx):
        nonlocal pos
        days = max(1, (dates[x_idx] - dates[entry_idx]).days)
        gross = op[x_idx] / entry_px - 1.0
        trades.append(gross - 2 * SPREAD - FIN * days / 365)
        for d in range(entry_idx, x_idx + 1):
            if d == entry_idx:
                r = close[d] / entry_px - 1.0
            elif d == x_idx:
                r = op[x_idx] / close[d - 1] - 1.0
            else:
                r = close[d] / close[d - 1] - 1.0
            daily[d] += r - FIN / 365
        daily[entry_idx] -= SPREAD; daily[x_idx] -= SPREAD
        pos = False

    for i in range(L, n - 1):
        u, l, an = up[i], lo[i], ann[i]
        if not np.isfinite(u):
            continue
        w = u - l
        c = close[i]
        if not pos:
            if an >= G_MIN and (up[i] - lo[i]) > 0 and c <= l + ENTRY_FRAC * w:
                pos, entry_idx, entry_px = True, i + 1, op[i + 1]
                armed = False; stop = -np.inf; peak = c
        else:
            exit_now = an < 0                      # trend broken
            if mode == "top":
                exit_now = exit_now or c >= u - ENTRY_FRAC * w
            elif mode == "ratchet_width":
                if c >= u - ARM_FRAC * w:
                    armed = True
                if armed:
                    stop = max(stop, u - stop_frac * w)
                exit_now = exit_now or (armed and c < stop) or c < l
            elif mode == "ratchet_atr":
                peak = max(peak, c)
                stop = max(stop, peak - m_atr * a[i])
                exit_now = exit_now or c < stop or c < l
            if exit_now:
                close_trade(i + 1)
    if pos:
        close_trade(n - 1)

    strat_d = daily[L:]
    bh_d = np.concatenate(([0.0], close[L + 1:] / close[L:-1] - 1.0))
    def sharpe(r):
        s = np.std(r)
        return float(np.mean(r) / s * np.sqrt(BARS_YR)) if s > 0 else 0.0
    return {"strat_total": float(np.prod(1 + strat_d) - 1), "bh_total": float(np.prod(1 + bh_d) - 1),
            "strat_sharpe": sharpe(strat_d), "bh_sharpe": sharpe(bh_d),
            "n_trades": len(trades), "tim": float(np.mean(strat_d != 0)),
            "trade_net": trades}


def summarise(rows, label) -> None:
    rows = [r for r in rows if r and r["n_trades"] > 0]
    if not rows:
        say(f"  {label}: no trades"); return
    st = np.array([r["strat_total"] for r in rows]); bh = np.array([r["bh_total"] for r in rows])
    ss = np.array([r["strat_sharpe"] for r in rows]); bs = np.array([r["bh_sharpe"] for r in rows])
    tim = np.array([r["tim"] for r in rows])
    net = np.array([x for r in rows for x in r["trade_net"]])
    say(f"  {label:22} | med strat {100*np.median(st):+6.0f}% vs B&H {100*np.median(bh):+6.0f}%  "
        f"beat-ret {100*np.mean(st>bh):>3.0f}%  beat-Sh {100*np.mean(ss>bs):>3.0f}%  "
        f"medSh {np.median(ss):+.2f}/{np.median(bs):+.2f}  TIM {100*np.median(tim):>3.0f}%  "
        f"net/tr {100*net.mean():+.3f}%")


def main() -> None:
    names = load_ftse.universe()
    say(f"Loading {len(names)} names; precomputing rolling channels (L={L})...")
    data = {}
    for t in names:
        try:
            d = load_ftse.load(t)
            arr = precompute(d, L)
            if arr is not None:
                data[t] = (d, arr)
        except Exception:
            pass
    say(f"{len(data)} usable.\n")

    say("#" * 96)
    say("# EXIT POLICY COMPARISON (net 10bps+5%), per-name vs buy-and-hold")
    say("#" * 96)
    summarise([run_policy(d, a, "top") for d, a in data.values()], "top (baseline)")
    for sf in (0.25, 0.5, 0.75):
        summarise([run_policy(d, a, "ratchet_width", stop_frac=sf) for d, a in data.values()],
                  f"ratchet_width sf={sf}")
    for m in (2.0, 3.0, 4.0):
        summarise([run_policy(d, a, "ratchet_atr", m_atr=m) for d, a in data.values()],
                  f"ratchet_atr M={m}")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
