# Quick Flip Scalper

An opening-range **fade** strategy on 5-minute candles, tested against the
S&P 500 5-minute data in `../candle-data`.

## The strategy

1. **Box** the opening range: C1 = first three 5-min bars (09:30–09:45 NY);
   its high/low defines the box for the next 75 minutes.
2. **Liquidity filter**: keep the day only if C1's range ≥ threshold × the
   prior-day 14-day ATR (ATR from daily bars aggregated from the 5-min data).
3. **Reversal entry** within the 75-min box, *outside* C1's range, fading back
   towards the far side of C1 (bullish C1 → short to C1.low; bearish C1 → long
   to C1.high):
   - **Shooting star / hammer**: rejection candle (long wick opposite the fade,
     tiny wick with it). Entry at the *next* bar's open; stop at the candle's
     extreme.
   - **Engulfing**: C2 then a full-range-engulfing C3 of the opposite colour.
     Entry is a **breakdown/breakout stop at C2's extreme**, scanned from C4:
     gap through → fill at open, else fill when price pierces the level. Stop at
     C3's extreme.
   Exits: take-profit at the far side of C1; trail the stop to the near side of
   C1 once price retraces ≥ 50% into the box; hard time-stop at box end.

All intrabar fills are **conservative**: a bar that straddles both stop and
target is assumed to hit the stop first (5-min bars hide the true path).

## Files

- `scalper.py` — pure, unit-tested core (candle geometry, fills, per-day sim).
- `test_scalper.py` — `pytest` suite (run: `../heirarchical-adaptive-filter-experiment/bin/python3 -m pytest -q`).
- `run_backtest.py` — runs the whole universe, sweeps liquidity threshold and
  transaction cost, splits by side, writes `trades.parquet`, prints the funnel
  and metrics.

## Result (June–Aug 2026, current S&P 500, 60 trading days)

**No edge.** ~4,400 trades. Break-even before costs (avg ≈ +0.02R, profit
factor ≈ 1.0) and a clear loser after any realistic transaction cost
(5 bps round-trip ≈ 0.13R and turns every configuration negative).

Why it fails, from the excursion diagnostics:
- Median adverse excursion (0.86R) **exceeds** median favourable excursion
  (0.72R) — the average trade goes against us more than for us, the opposite of
  what a reversion edge needs.
- The far-side-of-C1 target is essentially unreachable in 75 minutes (hit 2.4%
  of trades); 51% time out, 45% stop out.
- A nearer fixed-R target makes it *worse*, not better (small wins can't offset
  full-R losses at a ~40% win rate).
- Raising the liquidity threshold (0.25 → 0.75) does not improve expectancy;
  the "liquidity candle" filter isn't selecting anything predictive.

### Robustness sweep (`sweep.py`)

36 parameter combinations (wick geometry × trail trigger × box length ×
stop-width multiplier), each taking the best liquidity threshold:

- **0 / 36 are positive net of a 5 bps round-trip cost.** Best net corner
  −0.038R (strict wick, 120-min box, 2× stop width, 0.75 threshold).
- **36 / 36 are positive gross (0 bps)** but only just — best +0.062R (~2.5 bps
  per trade). So there is a *faint, consistent* gross reversion signal (not
  noise), roughly an order of magnitude too small to clear costs.
- Wider stops and a longer box reduce the damage (whipsaw) but never cross zero.

The negative result is robust: no reasonable corner of the parameter space makes
this a net-profitable rule on 5-minute data.

## The opposite: breakout momentum (`orb.py`, `stress.py`)

The MAE ≥ MFE asymmetry says price *continued* more than it reverted, so we
tested the inverse — an opening-range **breakout** that follows the first break
of the box and rides it for 75 minutes.

Apparent result: promising. The liquidity filter (inert for the fade) becomes
**monotonically** useful; at ≥0.75× ATR, box-end hold, the combined book showed
gross +11.4 bps, **+6.4 bps net of 5 bps, PF 1.16** (n=1,928) — the first thing
here that looked cost-clearing.

It did not survive stress-testing (`stress.py`), which is why we bought no
out-of-sample data:
- **Day-block bootstrap** (resampling whole days, since same-day trades are
  correlated): mean net **+0.017R, 95% CI [−0.024, +0.060]** — crosses zero.
- **Concentration**: the best **3 days = 156%** of profit; excluding them,
  −0.011R. Net-positive days 22 vs 23.
- **Market-neutral (beta) control**: morning drift was −3.3 bps, so shorts won
  for free; neutralised, the edge is **−0.011R**. It was beta, not alpha.
- **Direction permutation** flags p≈0, but that only reflects the same down-tape
  beta the neutral control removes.

**Conclusion: direction fails both ways** — the fade on costs, the follow on
scrutiny. Consistent with the wider programme: volatility is predictable,
direction is not.

### Caveats
- **Single 60-day regime**, survivorship-biased universe. A different/more
  volatile regime could differ.
- **5-min granularity** + conservative stop-first fills are pessimistic for a
  tight-stop scalper. We chose **not** to buy finer/longer data: the fade lost
  on costs and the breakout dissolved into beta + a few days in-sample, so
  out-of-sample confirmation would buy nothing.
- Parameters were **pre-registered, not optimised** (deliberately, to avoid
  overfitting a single window).

## Follow-up: futures costs + a volatility-regime gate (`regime_and_costs.py`)

Two post-processing tests over the cached trades (full output in
`regime_and_costs_out.txt`):

**(3) Futures-level costs.** Re-charging 0.5–1 bp round-trip (index-future
execution) instead of 5 bps does **not** rescue the fade — it is break-even
gross and ≤0 the moment any cost applies. The breakout was never a cost problem,
so cheaper costs only make its gross edge more visible; they don't touch the
beta and day-concentration that killed it.

**(1) Volatility-regime gate.** A causal regime (`vol_state` = yesterday's true
range / trailing ATR14) does steer as hypothesised: the breakout earns +0.018R
in expansion vs −0.013R in compression (at 1 bp); the fade is less-bad in
compression but never positive. Gating (follow in expansion) produced the
**first beta-neutral-positive** version here — +0.043R alpha at 0.5 bp — but the
day-block bootstrap CI crosses zero ([−0.011, +0.095]) and profit sits in the
top-3 of 45 days. **Promising signal, unproven strategy** — the same wall,
nudged not broken. (The same technique on daily FTSE fails outright; see the
swing-trading experiment's regime-gate follow-up.)
