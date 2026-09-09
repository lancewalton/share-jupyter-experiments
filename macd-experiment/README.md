# MACD Experiments

Test whether the Moving Average Convergence-Divergence indicator can signal
trade entry and exit well enough to beat buy-and-hold at the **portfolio** level,
net of costs.

## Framing: this is a falsification, not a search

MACD is a trend-following, direction-timing method. This programme has already
reached strong, stress-tested verdicts on exactly that class of strategy (see the
repo-root `README.md` and `STRATEGY.md`):

- **Direction is a mirage.** Every direction forecast hit a ~1% information-
  coefficient ceiling; the statistically real ones did not survive costs. Every
  strategy traded on a direction view was beaten outright by buy-and-hold.
- **Trend-following adds value against sustained trends and destroys it against
  whipsaws.** The rising-channel work showed a trend *timer* loses to buy-and-hold
  across UK equities, US equities and bonds, and diagnosed the mechanism
  (drawdown shape: equity V-crashes whipsaw the exit).
- **Filtering a direction signal only steers *hit-rate*, never *payoff*.** Feeding
  the trend-line breakout extra information — geometry, volume, extent, age, vol
  regime — raised how *often* it won, never how *much*. Strong breakouts revert
  rather than continue.

That last finding matters most here, because the historical instinct with MACD is
to bolt on trend-vs-flat filters to suppress false crosses in consolidation — the
very move already shown not to change payoff. So the honest prior for this
experiment is **negative**, and the goal is to falsify cleanly and cheaply, not to
sweep until something looks good. The intent is to add another well-tested "no"
(or, if it survives, a genuinely surprising "yes") to the programme — not to mine
for a lucky parameter set.

Guarding rules, applied from the first script (matching `STRATEGY.md`):

- **Baseline first.** Establish whether *textbook* MACD beats buy-and-hold before
  adding a single discretionary parameter. Each added parameter must earn its
  place against that baseline or it is noise.
- **Plateau, not peak.** A result counts only if a *contiguous neighbourhood* of
  parameter space is also good. An isolated optimum is overfitting by definition.
- **Held-out time.** Reserve data never touched by any sweep, or evaluate
  walk-forward. Robustness to parameters is not robustness to *time*, and time is
  where these strategies usually die.
- **Permutation null.** Run the same machinery on shuffled / surrogate returns. If
  it finds "good" parameters on noise, the discovery threshold is too loose.
- **Costs from trade one.** Dealing costs and (for FX) spread net into every
  result. An edge that only exists gross is not an edge.
- **Point-in-time discipline.** Every quantity setting a position on day *t* must
  be computable strictly before the holding period; lag any online/filtered
  signal one day. Treat a surprisingly good backtest as a look-ahead bug until
  proven otherwise.

## Success measure

Portfolio-level return / Sharpe / max-drawdown versus the **equal-weight
buy-and-hold** benchmark — *not* per-trade win rate. Individual trades are not
expected to beat buy-and-hold; the question is whether the strategy does at the
book level. Intermediate proxy: **return per hold-day**, and — crucially — a
decomposition of any edge into **hit-rate versus payoff**, since the programme's
prior is that direction filters move the former and not the latter.

## Data

Two datasets, two trading approaches. They are **separate tracks** — the
programme repeatedly finds that FX/intraday results do not transfer to equities,
so no parameter or conclusion crosses between them.

### 1. FTSE — daily OHLCV (long holds) — primary

- Source: `../data/eodhd/eodhd_uk_ohlcv.parquet` — a long-format panel
  (~11.5M rows) with columns `date, open, high, low, close, adjusted_close,
  volume, code, status`, where `code` is the ticker and `status` marks
  `active`/delisted. Because delisted names are retained (also listed in
  `lse_delisted.json`), the backtest is **survivorship-free**. Use
  `adjusted_close` for returns. Metadata in `eodhd_uk_meta.csv`.
- The data has errors and must be **winsorised** (cap implausible daily moves;
  cross-check against the glitch filter used elsewhere, `|daily log return| ≤ 0.6`).
- Expect relatively long holding periods; costs are daily-rebalance costs, not
  spread.
- Interpreter: `../heirarchical-adaptive-filter-experiment/bin/python3` (the repo
  venv with pandas/numpy/pyarrow; base `python3` has none).

### 2. FX — minute OHLCV (spread bets) — deferred

- Intended source: EODHD intraday (API key in `EODHD_API_TOKEN`), pending
  confirmation it has the coverage and history needed.
- **Deferred until the FTSE track reports.** Two programme findings warn against
  spending here first: intraday spread-bet costs are large enough to dominate any
  MACD edge (the quick-flip intraday scalper came out dead both ways net of
  realistic costs), and FX/intraday findings are kept strictly separate from
  equities. Minute-bar spread costs go in *before* any optimism.

## MACD definition

Two exponential moving averages of price with different lengths:

- The **longer**-history EMA (larger time constant, slower, smoother) tracks the
  general bullish/bearish trend.
- The **shorter**-history EMA (smaller time constant, faster, more reactive)
  tracks recent movement.

*(Note on terminology: a longer averaging window is a **larger** time constant, a
shorter window a **smaller** one. An earlier draft of this file had these paired
the wrong way round.)*

Textbook MACD is built from these as:

- **MACD line** = EMA_fast(price) − EMA_slow(price). Classic lengths: fast = 12,
  slow = 26.
- **Signal line** = EMA of the MACD line. Classic length: 9.
- **Histogram** = MACD line − signal line (rate/acceleration of the move).

The standard trade trigger is the **MACD line crossing the signal line** (bullish
when it crosses up, bearish when down); a common variant triggers on the MACD line
crossing **zero** — which is equivalent to the two price EMAs crossing each other
directly.

The gap between the fast and slow EMA indicates the rate at which price is
changing.

## MACD applicability

MACD is meant to be useful when a market moves from consolidation into momentum.
In flat markets the two EMAs cross frequently, producing many false signals. The
belief is that constraining both EMAs to move in the same direction keeps the
strategy in trending regimes — but note this programme has already found that such
regime filters change hit-rate without changing payoff, so this belief is exactly
what the experiment must test rather than assume.

## The experiments (phased)

Following the sibling experiments' layout: phased `run_phaseN_*.py` scripts, a
`macd/` package for shared code, a `tests/` directory, one PNG and one
`*_results.txt` per phase, and a `RESULTS.md` / `WRITEUP.md` for conclusions.

### Phase 0 — data & loader ✓ done

Load the EODHD UK OHLCV panel, winsorise/clean bad ticks, build the eligible
universe (minimum history; glitch filter), and cache. Implemented in `macd/data.py`
(cap daily log returns at |r| ≤ 0.6; returns via `expm1` so they never blow up;
point-in-time top-N-by-turnover universe). Tested in `tests/`.

### Phase 1 — standard MACD baseline (the first hypothesis) ✓ done — see `RESULTS.md`

**Result: standard 12/26/9 MACD loses to buy-and-hold gross (+3.0% vs +12.6%) and net
(−7.9%), on both halves of the sample — the programme's prior holds.** The per-trade
edge is the classic trend-following shape (low hit-rate, positive payoff), but the
names MACD selects under-earn the market per day (+4.8 vs +5.7 bps), concentration
wrecks diversification (maxDD −92%), and turnover cost finishes it. Uses the
**concentrated** portfolio (capital in the signalling names, not idle cash — the
momentum-work lesson). Run `run_phase1_baseline.py`.


Textbook **12/26/9** MACD, signal-line crossover, long-only on the FTSE universe,
market-on-open T+1 execution, net of costs. Deliberately minimal — no trend
filter, no retrenchment logic, no high/low sourcing, no separate exit params.

Report at portfolio level versus equal-weight buy-and-hold: CAGR, Sharpe, max
drawdown, return per hold-day, and the **hit-rate vs payoff decomposition**. This
is the cheap, decisive test: if standard MACD cannot clear buy-and-hold per
hold-day *before* costs, and its payoff (not just hit-rate) does not respond to
regime, the prior is confirmed and the sweep is unnecessary.

### Phase 2 — parameter sweep + stability of the standard form ✓ done — see `RESULTS.md`

**Result: 0 of 48 fast/slow combinations beat B&H net, no plateau, and a permutation
null gives p = 1.000 — MACD on real data is *worse* than MACD on shuffled noise.** The
trend it follows is anti-predictive (short-horizon reversal); the standard hypothesis
is comprehensively falsified, so the modifications below are very unlikely to help.

### Later phases — the modifications (Lance's design ideas)

To be added only after the standard form is characterised, each tested as an
increment against the Phase 1/2 baseline and judged on **payoff, not hit-rate**:

- **Trend-vs-flat gate.** Either (a) a third, faster EMA required to be separated
  from the MACD EMAs by a threshold, or (b) a volatility / mean-|Δclose| / ATR
  threshold — the premise being that trending and flat regimes differ in these.
  Two extra parameters each.
  - **(b) volatility gate — ✓ done, `run_phase3_volgate.py`, see `RESULTS.md`.**
    Under the correct concentrated portfolio the gate strictly *hurts* (excess CAGR
    −20% → −24%, Sharpe −0.28 → −0.40): it moves hit-rate but discards positions and
    reduces diversification. No risk-adjusted edge — the prior holds.
- **Ignore-first-signal, re-enter-after-retrenchment — ✓ done, `run_phase6_retrench.py`,
  see `RESULTS.md`.** Skip the first up-cross, enter on the second within a window. A
  wide window (40d) is the **best variant of the whole experiment** — Sharpe −0.28 →
  **+0.09** (the only positive Sharpe anywhere), excess −20% → −13%, at unchanged
  hit-rate/payoff — because it drops the first, most reversal-prone breakout. Still
  loses to B&H by 13%. Least-bad, not a win.
- **Directional price sourcing — ✓ done, `run_phase4_hilo.py`, see `RESULTS.md`.**
  Enter on a low-price MACD cross, exit on a high-price one. The one modification that
  genuinely helps (Sharpe −0.28 → −0.07 at equal participation, both legs contribute)
  — fewer whipsaws — but still loses to B&H by 16% with negative Sharpe. Reduces the
  damage, no edge.
- **Separate exit parameters — ✓ done, `run_phase5_exitparams.py`, see `RESULTS.md`.**
  Exit MACD with its own lengths. No help: baseline (exit = entry) is least-bad; a
  faster exit craters payoff, a slower one holds through reversals. No exit-timing edge.

### Phase 7 — walk-forward adaptive re-tuning ✓ done — see `RESULTS.md`

The honest version of "re-tune until it works": every 6 months pick the best-trailing
fast/slow and trade it forward. **Re-tuning is real adaptation — it matches the
hindsight oracle and beats fixed 12/26 — but still loses to B&H by 18% and is *worse
than the same scheme on shuffled returns* (p = 0.96).** You cannot tune off a losing
surface; re-tuning explains why the belief survives, not why the strategy would work.
Run `run_phase7_adaptive.py`.

### FX / spread-bet track

Repeat the phased plan on FX minute bars *only after* the FTSE track reports and
*only with* realistic spread/financing costs in from the start. Kept entirely
separate from the equity conclusions.

## Caution

There is huge potential for this to be a fishing expedition — the more so because
the thing being fished for (a tradable direction/trend edge) has already been
looked for across this programme and not found. The volume of FTSE data helps but
does not rescue us on its own: the cross-section is correlated (one market
regime), so effective sample size is far smaller than the row count. The
anti-overfitting protocol above — baseline-first, plateau-not-peak, held-out time,
permutation null, costs from trade one — is what keeps this honest, not the data
volume alone.
