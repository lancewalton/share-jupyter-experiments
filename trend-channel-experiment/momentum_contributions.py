"""Three capital-injection schemes on the recommended build's monthly returns.

Recommended build = mom+quality+low-vol, vol-targeted (no leverage), net tiered spread.
Compare, over the backtest:
  (A) LUMP SUM  -- invest the whole total at the start
  (B) FIXED     -- equal nominal contribution every month
  (C) INFLATION -- monthly contribution grows with inflation (2.5%/yr)
For each: total contributed, terminal value, multiple, money-weighted IRR, and the drawdown of
the portfolio VALUE actually experienced (contributions cushion it). Plus a zero-volatility
"smooth" counterfactual with the same CAGR, to ISOLATE the dip-buying benefit, and a fade check
(contribution as % of NAV over time).

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u momentum_contributions.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from momentum_tradeability import build, cagr, sharpe, maxdd
from momentum_multifactor import build_factors
from momentum_stamp_duty import run
from vol_target_momentum import vol_target

HERE = Path(__file__).resolve().parent
INFL_ANN = 0.025
OUT = HERE / "momentum_contributions_results.txt"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True); _lines.append(s)


def terminal(contribs, r):
    """Terminal value of a contribution stream: c_i contributed at start of month i, then earns r_i.."""
    n = len(r)
    G = np.append(np.cumprod((1 + r)[::-1])[::-1], 1.0)   # G[i] = growth of £1 from start of month i to end
    return float(np.sum(contribs * G[:n]))


def value_path(contribs, r):
    V, path = 0.0, []
    for c, ri in zip(contribs, r):
        V = (V + c) * (1 + ri)
        path.append(V)
    return np.array(path)


def irr_ann(contribs, term, r):
    """Money-weighted (internal) rate of return, annualised, via NPV bisection on monthly cashflows."""
    n = len(r)
    cf = -contribs.astype(float).copy()
    cf[-1] += term                                        # receive terminal at the end
    def npv(m):
        return np.sum(cf / (1 + m) ** np.arange(n))
    lo, hi = -0.9, 0.9
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) > 0: lo = mid
        else: hi = mid
    return (1 + (lo + hi) / 2) ** 12 - 1


def main() -> None:
    say("Building recommended-build monthly series (mom+quality+low-vol, vol-targeted)...")
    M, mret, liq, signal = build()
    quality, _v, _ = build_factors(M)
    lowvol = -mret.rolling(12).std().shift(1)
    P0, _ = run(M, mret, liq, [signal, quality, lowvol], 69, 0.0)
    r_tilt = P0["r"].dropna()
    vt, _w = vol_target(r_tilt, r_tilt.std() * np.sqrt(12), 1.0)
    r = vt.values
    n = len(r)
    say(f"{n} months ({vt.index[0].date()} -> {vt.index[-1].date()}); build CAGR "
        f"{100*cagr(vt):.2f}%, Sharpe {sharpe(vt):.2f}, per-unit maxDD {100*maxdd(vt):.0f}%\n")

    C = 1000.0                                            # base monthly contribution
    fixed = np.full(n, C)
    infl_m = (1 + INFL_ANN) ** (1 / 12) - 1
    inflation = C * (1 + infl_m) ** np.arange(n)
    total_fixed = fixed.sum()
    lump = np.zeros(n); lump[0] = total_fixed             # lump = same total as FIXED, all at t0

    # zero-vol counterfactual with the same total return (isolates the dip-buying effect)
    g = float(np.prod(1 + r)) ** (1 / n) - 1
    rs = np.full(n, g)

    say("#" * 92)
    say("# CONTRIBUTION SCHEMES (base £1,000/month; lump = same total invested at t0)")
    say("#" * 92)
    say(f"{'scheme':16} {'contributed':>12} {'terminal':>12} {'multiple':>9} {'IRR/yr':>8} "
        f"{'value maxDD':>12} {'vs smooth':>10}")
    for name, c in (("lump sum", lump), ("fixed", fixed), ("inflation-linked", inflation)):
        term = terminal(c, r)
        term_s = terminal(c, rs)
        V = value_path(c, r)
        vdd = (V / np.maximum.accumulate(V) - 1).min()
        dca = term / term_s - 1                            # dip-buying benefit vs zero-vol same-CAGR
        say(f"{name:16} {c.sum():>12,.0f} {term:>12,.0f} {term/c.sum():>8.2f}x {100*irr_ann(c, term, r):>7.2f}% "
            f"{100*vdd:>11.0f}% {100*dca:>+9.2f}%")

    say("\n(vs smooth = terminal into the strategy vs into a zero-volatility asset of the SAME CAGR;")
    say(" it isolates the 'buy more units when cheap' effect. Lump = 0 by construction.)")

    # fade: contribution as % of NAV over time (fixed scheme)
    say("\n--- how the dip-buying power fades: monthly contribution as % of portfolio NAV (fixed) ---")
    V = value_path(fixed, r)
    for yr in (2, 5, 10, 15, 20, 25):
        i = min(int(yr * 12), n - 1)
        say(f"  year {yr:>2}: £{C:,.0f} is {100*C/V[i]:>5.2f}% of the £{V[i]:,.0f} pot")

    OUT.write_text("\n".join(_lines) + "\n")
    say(f"\nsaved -> {OUT.name}")


if __name__ == "__main__":
    main()
