# MACD — results

## Phase 1 — standard 12/26/9 MACD, long-only (FTSE / EODHD UK)

**Verdict: standard MACD loses to buy-and-hold — even gross of costs, and on both
halves of the sample. Confirms the programme's prior.**

Setup: survivorship-free EODHD UK panel (active + delisted), names with ≥ 750
observations, cleaned by capping daily log returns at |r| ≤ 0.6 (kills bad
`adjusted_close` ticks). Investable universe = top-350 by trailing-1y turnover,
refreshed point-in-time and lagged one day. Textbook MACD line (12/26) crossing its
signal line (9); long while MACD > signal, else cash. Equal-weight across the
eligible cross-section; benchmark is the same universe, always invested.
Reproduce: `run_phase1_baseline.py` (→ `phase1_baseline_results.txt`, `phase1_baseline.png`).

| Full sample (1998–2026) | CAGR | Sharpe | maxDD |
|---|---|---|---|
| Buy-and-hold (equal-weight) | **+12.55%** | **+0.65** | −61.9% |
| MACD timing, **gross** | +4.46% | +0.37 | −73.0% |
| MACD timing, net 10 bps | −22.99% | −0.86 | −100% |

Out-of-sample split is the same story: pre-2013 B&H +8.2% vs MACD net −18.3%;
2013-on B&H +17.6% vs MACD net −27.8%.

### Why it fails (the mechanism, not just the number)

Trade-level decomposition (gross, per-name holding spells, investable universe):

- **101,112 trades — hit-rate 37.8%, mean payoff +0.51%, median −1.05%.**

That is the textbook trend-following signature: a **low hit-rate with a positive mean
payoff** — most trades are small losers, a few are large winners. So the per-trade
edge is genuinely positive *gross*. It still loses because:

1. **Cash-drag beats the timing.** Sitting out of the market between crossovers
   forgoes more upside (UK equities drift up, +12.6%/yr equal-weight) than the exits
   save on the way down. Being always-invested wins.
2. **Costs bury it.** Even the +4.46% gross edge turns to −23% net at 10 bps of
   turnover — the crossover churns too much.

This matches the programme's headline findings directly: **direction timing is beaten
by buy-and-hold**, and the edge that exists is in *hit-rate/payoff shape*, not in
tradable net return.

### Implication for the modifications

The proposed trend-vs-flat filters (third EMA, ATR/volatility gating) are, per the
programme's trend-line work, expected to move **hit-rate, not payoff** — i.e. to
change *which* trades fire, not the net economics. Phase 2 (sweep + stability of the
standard form) is worth running to confirm no robust parameter plateau beats B&H
before spending effort on the modifications; the honest prior is that none will.

## Phase 2 — parameter sweep + stability

*Not yet run.*
