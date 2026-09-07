"""Combine the two levers: vol-target the multi-factor (mom + quality / + value) tilt.

The factor overlay (momentum_multifactor.py) improves return/Sharpe; vol-targeting
(vol_target_momentum.py) cuts the ~-47% drawdown. Do they stack? For each book we take
the net monthly tilt series, then scale exposure to a constant vol (w = target/trailing,
causal 12m vol) and compare raw vs vol-targeted CAGR / vol / Sharpe / maxDD.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u momentum_combined.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from momentum_tradeability import build, cagr, sharpe, maxdd
from momentum_multifactor import build_factors, run_multi
from vol_target_momentum import vol_target

HERE = Path(__file__).resolve().parent
OUT = HERE / "momentum_combined_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True); _lines.append(s)


def row(name, r):
    say(f"  {name:34} CAGR {100*cagr(r):>+6.2f}%  vol {100*r.std()*(12**0.5):>5.1f}%  "
        f"Sharpe {sharpe(r):>5.2f}  maxDD {100*maxdd(r):>5.0f}%")


def main() -> None:
    say("Building panel + point-in-time factor rasters...")
    M, mret, liq, signal = build()
    quality, value, _ = build_factors(M)

    books = {
        "momentum only": {"mom": signal},
        "mom + quality": {"mom": signal, "q": quality},
        "mom + value + quality": {"mom": signal, "v": value, "q": quality},
    }

    say("\n" + "#" * 84)
    say("# VOL-TARGETED MULTI-FACTOR TILT (survivorship-free top-350, net tiered costs)")
    say("#" * 84)
    uni = run_multi(M, mret, liq, signal, {"mom": signal})["uni"]
    say(f"eligible-universe EW B&H: CAGR {100*cagr(uni):+.2f}%  Sharpe {sharpe(uni):.2f}\n")

    for name, comps in books.items():
        r = run_multi(M, mret, liq, signal, comps)["r"].dropna()
        base_vol = r.std() * (12 ** 0.5)
        say(f"{name}  (raw realised vol {100*base_vol:.1f}%)")
        row("raw (no targeting)", r)
        vt_own, w = vol_target(r, base_vol, 1.0)      # own-vol target, no leverage
        row("vol-target own-vol, no lev", vt_own)
        say(f"  {'':34} avg exposure {w.reindex(vt_own.index).mean():.2f}")
        vt_lev, w = vol_target(r, base_vol, 1.5)      # own-vol target, capped leverage
        row("vol-target own-vol, lev<=1.5", vt_lev)
        say("")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"saved -> {OUT.name}")


if __name__ == "__main__":
    main()
