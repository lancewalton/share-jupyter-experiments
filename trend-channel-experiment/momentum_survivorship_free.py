"""Survivorship-free 12-1 momentum on the EODHD UK universe (active + delisted).

At each month, the eligible universe is the top-N names by trailing-12m average daily
turnover (a point-in-time 'most liquid N UK stocks', INCLUDING names that were liquid
then and later delisted). Same 12-1 momentum as the survivor-only test:
  - long-only top quintile vs eligible-universe equal-weight buy-and-hold
  - market-neutral long-short (top - bottom quintile)
Monthly rebalance, turnover-based spread-bet costs. Reports CAGR/Sharpe/beta/alpha,
OOS split, bootstrap, recent, and the head-to-head vs the survivor-only result
(long-only CAGR 13.0% / Sharpe 0.83 / alpha +5.75% on 120 current constituents).

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u momentum_survivorship_free.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "eodhd"
OHLCV = DATA / "eodhd_uk_ohlcv.parquet"
START = "1998-01-01"
BT_START = "2001-01-01"        # backtest era (match the survivor-only test)
FRAC = 0.2
SPREAD = 10 / 1e4
SPLIT = pd.Timestamp("2013-01-01")
N_BOOT = 5000
RNG = np.random.default_rng(20260907)
OUT = HERE / "momentum_survivorship_free_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def ccy_map() -> dict:
    m = {}
    for fn in ("lse_active.json", "lse_delisted.json"):
        for x in json.load(open(DATA / fn)):
            if x.get("Code") and x["Code"] not in m:
                m[x["Code"]] = x.get("Currency")
    return m


def metrics(r: pd.Series, uni: pd.Series | None = None) -> dict:
    r = r.dropna()
    if len(r) < 12:
        return {"CAGR": float("nan"), "Sharpe": float("nan"), "maxDD": float("nan")}
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    sh = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0.0
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    out = {"CAGR": ann, "Sharpe": sh, "maxDD": mdd}
    if uni is not None:
        u = uni.reindex(r.index)
        b = np.cov(r, u)[0, 1] / np.var(u)
        out["beta"] = b
        out["alpha"] = r.mean() * 12 - b * u.mean() * 12
    return out


def yb_boot(r: pd.Series) -> tuple[float, float, float]:
    r = r.dropna()
    by = {y: r[r.index.year == y].to_numpy() for y in r.index.year.unique()}
    ys = [y for y in by if len(by[y])]
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = RNG.choice(len(ys), len(ys), replace=True)
        means[i] = np.concatenate([by[ys[j]] for j in pick]).mean() * 12
    return float(r.mean() * 12), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(mret, signal, liq, N):
    """Monthly long-only / long-short / eligible-EW returns for top-N liquid universe."""
    months = mret.index
    lo, ls, uni, lo_turn, ls_turn, idx = [], [], [], [], [], []
    wl_prev = pd.Series(0.0, index=mret.columns)
    ws_prev = pd.Series(0.0, index=mret.columns)
    for i in range(1, len(months)):
        t = months[i - 1]
        elig = liq.loc[t].dropna()
        if len(elig) < 50:
            continue
        elig = elig.nlargest(N).index
        s = signal.loc[t, elig].dropna()
        # winsorise the holding-month returns cross-sectionally within the eligible
        # universe (kills adjusted_close bad ticks; real delistings at -1 survive)
        r_e = mret.loc[months[i], elig].clip(lower=-1.0)
        r_e = r_e.clip(r_e.quantile(0.01), r_e.quantile(0.99))
        s = s.reindex(r_e.dropna().index).dropna()
        if len(s) < 30:
            continue
        k = max(1, int(len(s) * FRAC))
        top, bot = s.nlargest(k).index, s.nsmallest(k).index
        wl = pd.Series(0.0, index=mret.columns); wl[top] = 1.0 / k
        ws = pd.Series(0.0, index=mret.columns); ws[bot] = 1.0 / k
        lo.append(float(r_e.reindex(top).mean())); ls.append(float(r_e.reindex(top).mean() - r_e.reindex(bot).mean()))
        uni.append(float(r_e.mean()))
        lo_turn.append(float((wl - wl_prev).abs().sum()))
        ls_turn.append(float((wl - wl_prev).abs().sum() + (ws - ws_prev).abs().sum()))
        wl_prev, ws_prev = wl, ws
        idx.append(months[i])
    P = pd.DataFrame({"lo": lo, "ls": ls, "uni": uni, "lo_turn": lo_turn, "ls_turn": ls_turn},
                     index=pd.DatetimeIndex(idx))
    return P[P.index >= BT_START]


def main() -> None:
    say("Loading OHLCV and building monthly panels...")
    df = pd.read_parquet(OHLCV, columns=["date", "code", "close", "adjusted_close", "volume"])
    df = df[df["date"] >= START]
    ccy = ccy_map()
    fac = df["code"].map(lambda c: 0.01 if ccy.get(c) == "GBX" else 1.0).astype("float32")
    df["turnover"] = df["close"] * df["volume"] * fac
    adj = df.pivot(index="date", columns="code", values="adjusted_close")
    turn = df.pivot(index="date", columns="code", values="turnover")
    M = adj.resample("ME").last()
    mret = M.pct_change()
    liq = turn.resample("ME").mean().rolling(12).mean()
    signal = M.shift(1) / M.shift(12) - 1
    say(f"panel: {M.shape[1]} names x {M.shape[0]} months; backtest from {BT_START}\n")

    say("#" * 84)
    say("# SURVIVORSHIP-FREE 12-1 MOMENTUM (EODHD UK, active+delisted)")
    say("#" * 84)
    say("Reference — survivor-only (120 current constituents): long-only CAGR 13.0%, "
        "Sharpe 0.83, alpha +5.75%\n")

    for N in (100, 350):
        P = run(mret, signal, liq, N)
        uni = P["uni"]
        lo_net = P["lo"] - P["lo_turn"] * SPREAD
        ls_net = P["ls"] - P["ls_turn"] * SPREAD
        um = metrics(uni)
        lom = metrics(lo_net, uni)
        lsm = metrics(ls_net)
        say(f"=== top-{N} liquid universe ({len(P)} months) ===")
        say(f"  eligible-universe EW B&H: CAGR {100*um['CAGR']:+.2f}%  Sharpe {um['Sharpe']:.2f}")
        say(f"  LONG-ONLY tilt (net 10bps): CAGR {100*lom['CAGR']:+.2f}%  Sharpe {lom['Sharpe']:.2f}  "
            f"beta {lom.get('beta',0):.2f}  alpha {100*lom.get('alpha',float('nan')):+.2f}%  maxDD {100*lom['maxDD']:.0f}%")
        pre, post = lo_net[lo_net.index < SPLIT], lo_net[lo_net.index >= SPLIT]
        rc = lo_net[lo_net.index >= pd.Timestamp("2023-01-01")]
        b, lo_ci, hi_ci = yb_boot(lo_net)
        star = "CI>0" if lo_ci > 0 else ("CI<0" if hi_ci < 0 else "CI spans 0")
        say(f"    OOS: pre-2013 CAGR {100*metrics(pre)['CAGR']:+.2f}% | post-2013 {100*metrics(post)['CAGR']:+.2f}%"
            f" | 2023-26 {100*metrics(rc)['CAGR']:+.2f}%")
        say(f"    long-only bootstrap ann mean {100*b:+.2f}% 95% CI [{100*lo_ci:+.2f}%, {100*hi_ci:+.2f}%] ({star})")
        say(f"  LONG-SHORT (net 10bps): CAGR {100*lsm['CAGR']:+.2f}%  Sharpe {lsm['Sharpe']:.2f}  maxDD {100*lsm['maxDD']:.0f}%")
        say("")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"saved -> {OUT.name}")


if __name__ == "__main__":
    main()
