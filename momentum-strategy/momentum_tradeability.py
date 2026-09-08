"""Is 12-1 momentum genuinely tradeable, net of everything? On the survivorship-free
EODHD UK universe (top-350 liquid), stress the long-only tilt and market-neutral book:
  - liquidity-tiered spread costs (large-caps cheap, mid-caps dear)
  - rebalance frequency (monthly vs quarterly -- the turnover lever)
  - break-even cost (how high can round-trip costs go before it stops beating B&H)
  - momentum-crash drawdown profile (maxDD, worst months, 2008-09)

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u momentum_tradeability.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "eodhd"
OHLCV = DATA / "eodhd_uk_ohlcv.parquet"
BT_START = "2001-01-01"
FRAC, N = 0.2, 350
TIER_BPS = {"top": 15.0, "mid": 40.0, "low": 80.0}   # round-trip bps by turnover tercile
OUT = HERE / "momentum_tradeability_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def cagr(r):
    r = r.dropna()
    return (1 + r).prod() ** (12 / len(r)) - 1 if len(r) else float("nan")


def sharpe(r):
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0.0


def maxdd(r):
    c = (1 + r.dropna()).cumprod()
    return (c / c.cummax() - 1).min()


def build():
    df = pd.read_parquet(OHLCV, columns=["date", "code", "close", "adjusted_close", "volume"])
    df = df[df["date"] >= "1998-01-01"]
    ccy = {}
    for fn in ("lse_active.json", "lse_delisted.json"):
        for x in json.load(open(DATA / fn)):
            ccy.setdefault(x.get("Code"), x.get("Currency"))
    fac = df["code"].map(lambda c: 0.01 if ccy.get(c) == "GBX" else 1.0).astype("float32")
    df["turnover"] = df["close"] * df["volume"] * fac
    adj = df.pivot(index="date", columns="code", values="adjusted_close")
    turn = df.pivot(index="date", columns="code", values="turnover")
    M = adj.resample("ME").last()
    return M, M.pct_change(), turn.resample("ME").mean().rolling(12).mean(), M.shift(1) / M.shift(12) - 1


def tier_spread(turn_elig: pd.Series) -> pd.Series:
    q1, q2 = turn_elig.quantile(1 / 3), turn_elig.quantile(2 / 3)
    return turn_elig.map(lambda v: TIER_BPS["top"] if v >= q2 else
                         (TIER_BPS["mid"] if v >= q1 else TIER_BPS["low"]))


def _cost(new_w, old_w, sp):
    allc = new_w.index if old_w is None else new_w.index.union(old_w.index)
    nw = new_w.reindex(allc).fillna(0.0)
    ow = (nw * 0.0) if old_w is None else old_w.reindex(allc).fillna(0.0)
    dz = (nw - ow).abs()
    cost = float((dz * sp.reindex(allc).fillna(sp.median()) / 2 / 1e4).sum())
    return cost, float(dz.sum())          # cost, one-way turnover


def run(M, mret, liq, signal, K, cost_mode, flat_bps=0.0):
    months = M.index
    lo_w = ls_l = ls_s = None
    lo_r, ls_r, uni_r, idx = [], [], [], []
    turn_sum, n_years = 0.0, 0.0
    for i in range(1, len(months)):
        t, th = months[i - 1], months[i]
        elig = liq.loc[t].dropna().nlargest(N).index
        r_e = mret.loc[th, elig].clip(lower=-1.0)
        r_e = r_e.clip(r_e.quantile(0.01), r_e.quantile(0.99))
        lo_cost = ls_cost = 0.0
        if lo_w is None or (i - 1) % K == 0:
            s = signal.loc[t, elig].dropna().reindex(r_e.dropna().index).dropna()
            if len(s) < 30:
                continue
            k = max(1, int(len(s) * FRAC))
            top, bot = s.nlargest(k).index, s.nsmallest(k).index
            new_lo = pd.Series(1.0 / k, index=top)
            new_l, new_s = pd.Series(1.0 / k, index=top), pd.Series(1.0 / k, index=bot)
            sp = tier_spread(liq.loc[t, elig].dropna()) if cost_mode == "tier" else pd.Series(flat_bps, index=elig)
            lo_cost, tw = _cost(new_lo, lo_w, sp)
            c1, _ = _cost(new_l, ls_l, sp); c2, _ = _cost(new_s, ls_s, sp); ls_cost = c1 + c2
            lo_w, ls_l, ls_s = new_lo, new_l, new_s
            turn_sum += tw
        lo_r.append(float(r_e.reindex(lo_w.index).mean()) - lo_cost)
        ls_r.append(float(r_e.reindex(ls_l.index).mean() - r_e.reindex(ls_s.index).mean()) - ls_cost)
        uni_r.append(float(r_e.mean())); idx.append(th)
    P = pd.DataFrame({"lo": lo_r, "ls": ls_r, "uni": uni_r}, index=pd.DatetimeIndex(idx))
    P = P[P.index >= BT_START]
    yrs = len(P) / 12
    return P, (turn_sum / yrs if yrs else float("nan"))


def main() -> None:
    say("Building panel...")
    M, mret, liq, signal = build()
    say(f"panel {M.shape[1]} names x {M.shape[0]} months\n")

    say("#" * 96)
    say("# 12-1 MOMENTUM TRADEABILITY (survivorship-free UK, top-350)")
    say("#" * 96)
    base, _ = run(M, mret, liq, signal, 1, "flat", 0.0)
    uni = base["uni"]
    say(f"eligible-universe EW B&H: CAGR {100*cagr(uni):+.2f}%  Sharpe {sharpe(uni):.2f}\n")

    say(f"{'rebal':>9} {'cost':>10} {'LO_CAGR':>8} {'LO_Sh':>6} {'LO_maxDD':>9} {'turn/yr':>8} "
        f"{'beatsBH':>8} {'LS_CAGR':>8} {'LS_Sh':>6} {'LS_maxDD':>9}")
    for K, lab in ((1, "monthly"), (3, "quarterly")):
        for name, mode, bps in [("gross", "flat", 0.0), ("flat 10bp", "flat", 10.0),
                                ("flat 30bp", "flat", 30.0), ("tiered", "tier", 0.0)]:
            P, tpy = run(M, mret, liq, signal, K, mode, bps)
            beats = "yes" if cagr(P["lo"]) > cagr(uni) else "no"
            say(f"{lab:>9} {name:>10} {100*cagr(P['lo']):>+7.2f}% {sharpe(P['lo']):>6.2f} "
                f"{100*maxdd(P['lo']):>8.0f}% {tpy:>7.1f}x {beats:>8} "
                f"{100*cagr(P['ls']):>+7.2f}% {sharpe(P['ls']):>6.2f} {100*maxdd(P['ls']):>8.0f}%")
        say("")

    say("=== break-even flat round-trip cost (long-only CAGR drops to eligible B&H) ===")
    target = cagr(uni)
    for K, lab in ((1, "monthly"), (3, "quarterly")):
        pts = {c: cagr(run(M, mret, liq, signal, K, "flat", float(c))[0]["lo"]) for c in (0, 25, 50, 100, 150, 200)}
        be = None
        cs = sorted(pts)
        for a, b in zip(cs, cs[1:]):
            if pts[a] >= target > pts[b]:
                be = a + (b - a) * (pts[a] - target) / (pts[a] - pts[b]); break
        say(f"  {lab}: ~{be:.0f} bps round-trip" if be else
            f"  {lab}: still beats B&H at 200bps" if pts[200] > target else f"  {lab}: below B&H already")

    say("\n=== momentum-crash / drawdown profile (tiered costs, monthly) ===")
    P, _ = run(M, mret, liq, signal, 1, "tier", 0.0)
    for leg, nm in (("lo", "long-only tilt"), ("ls", "long-short")):
        r = P[leg]
        worst = ", ".join(f"{d.date()} {100*v:.0f}%" for d, v in r.nsmallest(3).items())
        say(f"  {nm}: maxDD {100*maxdd(r):.0f}%   worst months: {worst}")
    say(f"  long-short 2008-09 cumulative: {100*((1+P['ls'].loc['2008':'2009']).prod()-1):+.0f}% "
        f"(classic momentum crash)")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
