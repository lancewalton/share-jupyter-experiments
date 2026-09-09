# MACD — Findings

*Can the Moving Average Convergence-Divergence indicator — the canonical
trend-following signal — time entries and exits well enough to beat buy-and-hold at
the portfolio level, net of costs? Tested on the survivorship-free EODHD UK universe.
This was run as a **falsification**, not a search: the programme's prior is that
direction timing is a mirage, so the job was to try hard to break MACD cleanly and
cheaply rather than mine for a lucky parameter set.*

Sibling to the UK-equity programme (`../momentum-strategy`,
`../swing-trading-trend-lines-experiment`, etc.); it reuses the survivorship-free
EODHD panel and the same discipline (causal / point-in-time, net-of-cost honesty,
permutation nulls, plateau-not-peak).

## TL;DR

- **Standard MACD loses to buy-and-hold — even gross.** Textbook 12/26/9, long-only,
  concentrated book: gross **+3.0%** CAGR vs B&H **+12.6%**; **−7.9%** net of 10 bps.
  Both halves of the sample. (Phase 1)
- **No parameter set beats it, and there is no plateau.** All 48 fast×slow
  combinations lose (−17% to −23% excess); the least-bad point is an isolated,
  non-robust cell. (Phase 2)
- **The trend is *anti*-predictive, not merely uninformative.** A permutation null
  (200 per-name time-shuffles) gives **p = 1.000** — real MACD does worse than *every*
  surrogate. Shuffling away the serial structure *improves* results. The crossovers
  MACD chases tend to reverse (short-horizon mean-reversion). (Phase 2)
- **The volatility-regime gate *hurts*.** Under the correct portfolio it drops excess
  CAGR −20% → −24% and Sharpe −0.28 → −0.40. Its "improvement" in an earlier draft was
  a construction artefact (see the correction below). (Phase 3)
- **No modification rescues it; the two that help share one mechanism.** Low-for-long /
  high-for-short (Phase 4) lifts Sharpe −0.28 → −0.07 at unchanged participation;
  retrenchment re-entry with a wide window (Phase 6) is the best variant anywhere —
  Sharpe −0.28 → **+0.09**, the only positive reading — by skipping the first,
  reversal-prone breakout. Separate exit parameters (Phase 5) don't help at all. Every
  helpful variant still loses to B&H by 13–16%.

**Bottom line: standard MACD and its modifications do not beat buy-and-hold on UK
equities — confirmed at every turn.** The two variants that reduce the loss do so the
same way: by *avoiding the first breakout*, which the null shows is the most
reversal-prone. You can make a direction bet lose less; you cannot make it win.

## The correction that mattered

The first cut weighted the strategy across the *whole* eligible universe, leaving ~90%
of capital idle in cash for non-signalling names. That penalised MACD for a cash-drag
a real portfolio never carries — and, worse, it made the volatility gate look
*helpful* (cutting participation cut the phantom drag). Rebuilding the strategy as a
**concentrated** book — capital equal-weighted across just the currently-long names,
fully invested, cash only when none signal — is the same lesson as the momentum work
(`momentum_survivorship_free.py` weights the *selected* names, not the universe). Under
the correct construction the vol-gate "win" reversed to a loss, and the headline losses
shrank (net −23% → −7.9%) without changing the conclusion. A textbook reminder that a
backtest's benchmark construction can manufacture both false losses and false wins.

## Data

Survivorship-free **EODHD UK** daily panel (`../data/eodhd/eodhd_uk_ohlcv.parquet`,
~11.5M rows, active + delisted), names with ≥ 750 observations. Bad `adjusted_close`
ticks are removed by capping daily log returns at |r| ≤ 0.6 (the programme's glitch
threshold) and taking returns as `expm1` of the capped log return, so they are bounded
and never blow up to infinity. The investable universe is the **top-350 by trailing-1y
turnover**, refreshed point-in-time and lagged one day — essential, because the raw
2,865-name set is ~30% stale micro-caps whose bid-ask bounce wrecks any naive
equal-weight book. Benchmark: the same eligible universe, equal-weight, always
invested. All results net of 10 bps per unit turnover.

## The experiments

**Phase 1 — the standard hypothesis.** Textbook 12/26/9, long while the MACD line is
above its signal line, concentrated book, T-lagged for causality. It loses to
buy-and-hold gross and net, on both the pre-2013 and 2013-on halves. The trade-level
decomposition is the classic trend-following shape — hit-rate 37.8%, positive mean
payoff (+0.51%), negative median (−1.05%): a few big winners, many small losers. But
the names MACD flags as trending up *under-earn the market* per day (+4.8 vs +5.7 bps),
and concentrating in them is undiversified, so drawdowns blow out to −92%. A genuine
per-trade shape, no portfolio edge.

**Phase 2 — sweep, stability, null.** Sweeping fast × slow finds nothing: 0 of 48
combinations beat B&H, and the best is an isolated cell, not a plateau — the hallmark
of an absent edge rather than a mistuned one. The decisive test is the permutation
null: shuffle each name's daily returns in time (destroying the trend structure,
preserving the distribution), rebuild the price, and re-run. Real MACD scores **−20.4%**
excess; the shuffled surrogates average **−13.8%** — and **every one of the 200 beat
the real strategy (p = 1.000)**. So the serial structure MACD keys on is not just
noise, it is *worse* than noise. This is the fingerprint of short-horizon reversal: the
crossovers systematically precede pullbacks, so following them is anti-predictive.

**Phase 3 — volatility-regime gate (modification 1).** The README's idea: only trade
when recent volatility is high, on the premise that trending markets are more volatile
than flat ones. Gating MACD longs to days above a trailing volatility percentile *moves
hit-rate* (37.8% → 47.3%) but leaves per-trade payoff flat — and under the concentrated
book it *hurts* net return and Sharpe, because discarding positions just reduces
diversification. The gated days do carry higher gross return per day (+8.5 vs the
market's +5.7 bps), but that is risk compensation for holding only during volatile
stretches, not skill: it never reaches portfolio Sharpe or net return.

**Phase 4 — low-for-long / high-for-short (modification 2).** Enter on a MACD cross
computed from **low** prices (you only go long once even the intraday lows are trending
up — a stricter confirmation), and exit on a cross from **high** prices (you hold
through minor dips until even the highs roll over). Implemented as a hysteresis state
machine with different entry and exit signals. Unlike the vol gate this genuinely
improves risk-adjusted performance and is *not* an artefact: participation is unchanged
(~1.08M hold-days), yet Sharpe lifts −0.28 → −0.07 and both legs contribute (~2% each).
Fewer whipsaws, lower hit-rate (34.8%), similar payoff. But it still loses to
buy-and-hold by 16% with negative Sharpe.

**Phase 5 — separate exit parameters (modification 3).** Let the exit MACD use its own
lengths while the entry stays 12/26. Nothing helps: matching the exit to the entry is
the least-bad choice. A faster exit (cut losers quicker) realises whipsaw losses sooner
and craters payoff (−26%); a slower exit (hold winners longer) sits through the
reversals (−21%). There is no exit-timing edge to find.

**Phase 6 — ignore-first-signal / retrenchment re-entry (modification 4).** Skip the
first up-cross of an episode and enter on the second within a window (exit unchanged).
A tight window destroys it (w=5: −30%, too few, too selective). A *wide* window is the
best variant of the whole experiment: w=40 lifts Sharpe −0.28 → **+0.09** — the only
positive reading anywhere — and excess −20.4% → −13.1%, at unchanged hit-rate and
payoff (so, again, not the participation artefact). The reason ties the whole study
together: the crossovers are anti-predictive because the *first* breakout tends to
reverse, so dropping it and waiting for a later one avoids the worst entries. But even
this best case loses to buy-and-hold by 13% with a negligible Sharpe — least-bad, not a
win.

## How it fits the programme

This is the programme's most sharply falsified negative. Earlier trend-following work
(`../swing-trading-trend-lines-experiment`, `../momentum-strategy`'s trend-channel
provenance) found that direction timers lose to B&H and that filters move *hit-rate,
not payoff*; MACD reproduces both and adds a stronger result — a **p = 1.000 null**
showing the signal is actively worse than random. The two modifications that reduce the
loss (low/high sourcing, wide-window retrenchment) both work by *avoiding the first,
reversal-prone breakout* — never by adding predictive power — exactly the pattern the
programme predicts: you can make a direction bet lose *less*, never win.

## Reproduce

`run_phase1_baseline.py` … `run_phase6_retrench.py` (interpreter:
`../heirarchical-adaptive-filter-experiment/bin/python3`). Pure functions in `macd/`
(indicator, backtest, metrics, data, surrogate), 35 tests in `tests/`. Full numbers in
`RESULTS.md` and the `*_results.txt` files.

## Not done

The FX / minute-bar spread-bet track is deliberately deferred — intraday spread costs
dominate any MACD edge (mirror of the quick-flip scalper's "dead both ways"), and the
programme keeps FX/intraday findings separate from equities. All four proposed equity
modifications have now been tested; none beats buy-and-hold.

---

*A validated research negative, honestly caveated — not investment advice.*
