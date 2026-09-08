# IG "forward" (quarterly) spread-bet quotes — captured 2026-09-08

Source screenshots behind the cost analysis in `MOMENTUM_STRATEGY_SPEC.md` §16.1. Live IG
forward spread-bet prices for three large-cap UK shares across the SEP-26 / DEC-26 / MAR-27
expiries, used to measure the embedded financing (carry) of a quarterly forward vs a rolling bet.

| file | instrument | SEP-26 (sell/buy) | DEC-26 | MAR-27 |
|---|---|---|---|---|
| `barclays.png` | Barclays PLC | 495.6 / 497.7 | 499.9 / 504.0 | 504.1 / 510.3 |
| `tesco.png` | Tesco PLC | 479.23 / 481.35 | 483.35 / 487.44 | 487.45 / 493.54 |
| `hsbc.png` | HSBC Holdings PLC | 1565.9 / 1572.6 | 1579.7 / 1592.1 | 1596.4 / 1608.8 |

**Measured:** embedded net carry ≈ 4.25%/yr (mid rises ~1.06%/quarter, consistent across all
three → gross financing ≈ SONIA + ~3.75%); front-contract round-trip dealing spread ≈ 0.42–0.44%.
Conclusion (§16.1): forward spread bets are not more efficient for this fully-invested strategy —
carry swamps the per-trade spread advantage. Quotes are indicative, at the capture time only.
