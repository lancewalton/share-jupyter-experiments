"""Full numbers for the promoted recommended build: mom + quality + low-vol.

Produces the tilt and the vol-targeted build, each net of spread and net of spread+stamp duty,
so the spec can quote accurate figures. Low-vol = z(-trailing 12m vol) added as a selection
factor (previous experiment showed it lifts mom+quality 0.80 -> 0.84).

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u momentum_lowvol_build.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from momentum_tradeability import build, cagr, sharpe, maxdd
from momentum_multifactor import build_factors
from momentum_stamp_duty import run           # run(M, mret, liq, facs, want, stamp_bps)
from vol_target_momentum import vol_target

HERE = Path(__file__).resolve().parent
OUT = HERE / "momentum_lowvol_build_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True); _lines.append(s)


def row(label, r):
    say(f"  {label:40} CAGR {100*cagr(r):>+6.2f}%  Sharpe {sharpe(r):>5.2f}  maxDD {100*maxdd(r):>4.0f}%")


def main() -> None:
    say("Building panel + rasters...")
    M, mret, liq, signal = build()
    quality, _v, _ = build_factors(M)
    lowvol = -mret.rolling(12).std().shift(1)
    facs = [signal, quality, lowvol]

    say("#" * 78)
    say("# RECOMMENDED BUILD: mom + quality + low-vol (~69 names)")
    say("#" * 78)

    P0, _ = run(M, mret, liq, facs, 69, 0.0)   # net tiered spread
    Ps, _ = run(M, mret, liq, facs, 69, 50.0)  # net spread + 0.5% stamp on buys
    r0, rs = P0["r"], Ps["r"]
    say(f"eligible-universe EW B&H: CAGR {100*cagr(P0['uni']):+.2f}%  Sharpe {sharpe(P0['uni']):.2f}\n")
    say("TILT (top quintile, equal-weight):")
    row("net spread", r0)
    row("net spread + stamp duty", rs)

    say("\nVOL-TARGETED (own-vol target; tilt series then scaled):")
    for base_r, tag in ((r0, "net spread"), (rs, "net spread + stamp")):
        bv = base_r.std() * np.sqrt(12)
        for cap, lab in ((1.0, "no leverage"), (1.5, "lev<=1.5")):
            vt, _w = vol_target(base_r, bv, cap)
            row(f"{lab}, {tag}", vt)

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
