"""At £10,000, how badly does whole-share-only dealing (mid-caps on Trading 212) hurt?

The book holds ~69 near-equal positions. At £10k that's ~£145 per name -- so a name that can
only be dealt in WHOLE shares (T212 offers fractional on FTSE 100, patchier below) introduces
rounding error, and a name priced above the target can't be held at weight at all. We use the
strategy's ACTUAL held names each month and their REAL month-end prices to measure:
  - the per-name price distribution vs the £-target
  - names that can't be held (price > target) or are dropped (price > 2x target -> rounds to 0)
  - book-level tracking error under whole-share dealing, worst-case (all names whole-share) and
    realistic (top-100-by-liquidity = FTSE-100-like = fractional; the rest whole-share)
  - the same at fewer holdings (30, 20) -- concentrating raises the per-name target and eases it

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u wholeshare_10k.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_tradeability import build, DATA, OHLCV, N as UNIV_N, BT_START
from momentum_multifactor import build_factors, zscore

HERE = Path(__file__).resolve().parent
NAV = 10_000.0
FRACTIONAL_TOP = 100          # assume top-100 by liquidity (~FTSE 100) are fractional; rest whole-share
OUT = HERE / "wholeshare_10k_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True); _lines.append(s)


def raw_price_panel():
    df = pd.read_parquet(OHLCV, columns=["date", "code", "close"])
    df = df[df["date"] >= "1998-01-01"]
    ccy = {}
    for fn in ("lse_active.json", "lse_delisted.json"):
        for x in json.load(open(DATA / fn)):
            ccy.setdefault(x.get("Code"), x.get("Currency"))
    fac = df["code"].map(lambda c: 0.01 if ccy.get(c) == "GBX" else 1.0).astype("float32")
    df["px"] = df["close"] * fac                      # price in GBP
    return df.pivot(index="date", columns="code", values="px").resample("ME").last()


def held_with_rank(M, mret, liq, signal, quality, lowvol, want):
    """Per formation month: the top-`want` composite names and their liquidity rank in the universe."""
    months = M.index
    rows = []
    for i in range(1, len(months)):
        t = months[i - 1]
        if months[i] < pd.Timestamp(BT_START):
            continue
        elig = liq.loc[t].dropna().nlargest(UNIV_N).index
        z = pd.concat([zscore(signal.loc[t, elig]), zscore(quality.loc[t, elig]),
                       zscore(lowvol.loc[t, elig])], axis=1).mean(axis=1, skipna=True)
        z = z[zscore(signal.loc[t, elig]).notna()].dropna()
        if len(z) < 30:
            continue
        top = z.nlargest(min(want, len(z))).index
        rank = liq.loc[t, elig].rank(ascending=False)
        rows.append((t, list(top), rank))
    return rows


def assess(px, held_rows, want):
    tgt = NAV / want                                  # £ target per name (equal weight, full invest)
    all_err, real_err, prices = [], [], []
    ws_err, ws_drop, ws_n = [], 0, 0                  # whole-share (non-fractional) subset only
    for t, names, rank in held_rows:
        row = px.loc[t] if t in px.index else px.reindex([t]).iloc[0]
        for nm in names:
            p = row.get(nm, np.nan)
            if not np.isfinite(p) or p <= 0:
                continue
            prices.append(p)
            whole = round(tgt / p)
            e = abs(whole * p - tgt) / tgt
            all_err.append(e)
            fractional = rank.get(nm, 9999) <= FRACTIONAL_TOP
            real_err.append(0.0 if fractional else e)
            if not fractional:                        # this is a whole-share-only name
                ws_n += 1; ws_err.append(e)
                if whole == 0:
                    ws_drop += 1
    prices = np.array(prices)
    return dict(tgt=tgt, prices=prices, all_err=np.mean(all_err), real_err=np.mean(real_err),
                ws_n=ws_n, ws_frac=100 * ws_n / len(prices), ws_err=np.mean(ws_err) if ws_err else 0,
                ws_drop=100 * ws_drop / ws_n if ws_n else 0)


def main() -> None:
    say("Building + reconstructing held names with real prices...")
    M, mret, liq, signal = build()
    quality, _v, _ = build_factors(M)
    lowvol = -mret.rolling(12).std().shift(1)
    px = raw_price_panel()

    # sanity: a few known large-caps' prices at a recent month-end (GBP)
    tt = px.index[px.index.get_indexer([pd.Timestamp("2024-06-30")], method="nearest")[0]]
    say(f"sanity check prices @ {tt.date()} (GBP/share): " +
        ", ".join(f"{c} £{px.loc[tt, c]:.1f}" for c in ("AZN", "SHEL", "HSBA", "BP", "RIO", "ULVR") if c in px.columns))

    # price distribution of held names (at the quintile)
    rows69 = held_with_rank(M, mret, liq, signal, quality, lowvol, 69)
    a69 = assess(px, rows69, 69)
    p = a69["prices"]
    say(f"\nHeld-name price distribution (GBP, {len(p):,} share-months): "
        f"median £{np.median(p):.1f}, p90 £{np.percentile(p,90):.1f}, p99 £{np.percentile(p,99):.1f}, max £{p.max():.0f}")
    say(f"  share of held names priced over £50: {100*np.mean(p>50):.1f}%  over £100: {100*np.mean(p>100):.1f}%\n")

    say("#" * 84)
    say(f"# WHOLE-SHARE ROUNDING at £{NAV:,.0f}   (per-name £-target shown; equal weight, full invest)")
    say("#" * 84)
    say(f"{'#held':>6} {'£/name':>7} {'WS names%':>9} {'WS drop%':>9} "
        f"{'trkerr all-WS':>14} {'trkerr realistic':>17}")
    for want in (69, 30, 20):
        rows = held_with_rank(M, mret, liq, signal, quality, lowvol, want) if want != 69 else rows69
        a = assess(px, rows, want)
        say(f"{want:>6} {a['tgt']:>6.0f} {a['ws_frac']:>8.1f}% {a['ws_drop']:>8.1f}% "
            f"{100*a['all_err']:>12.1f}% {100*a['real_err']:>16.1f}%")

    say(f"\n  WS names%   = share of held names that are whole-share-only (rank > top-{FRACTIONAL_TOP} by liquidity)")
    say("  WS drop%    = of those whole-share names, the share too pricey to hold (1 share rounds to 0)")
    say("  trkerr all-WS   = mean per-name weight error if EVERY name is whole-share (worst case)")
    say(f"  trkerr realistic= same but top-{FRACTIONAL_TOP}-by-liquidity (~FTSE 100) are fractional, rest whole-share")
    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
