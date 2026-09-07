# Swing Trend-Lines — Results

## What was tested
A faithful, single-timeframe (daily) implementation of the method in `README.md`:

- **Structure:** support/resistance = convex hull of lows/highs in `ln(price)`, anchored on the all-history global extreme (exactly as specified). `trendlines.py`.
- **Data cleaning:** gross single-bar wick errors repaired (e.g. BP's 4.7p print on 2019-12-13). `cleaning.py`.
- **Signals:** action line = last (steepest, most recent) opposing trend line; needs ≥3 touches within a `k·ATR` band; entry when the latest bar breaks it by more than the band. `signals.py`.
- **Trades:** next-open entry; stop starts at `entry − 2·ATR` then ratchets toward the safety line (never loosening); exit on stop touch; one position at a time. Spread-bet costs: 10 bps/side spread + 5%/yr financing, no stamp duty. `backtest.py`.
- **Universe:** 20 liquid, sector-varied FTSE names, ~26 years each.

All logic is covered by 19 TDD unit/property tests.

## Verdict: the exact method loses, robustly

Base case **k = 0.5**:

| metric | value |
|---|---|
| tickers profitable | **0 / 20** |
| median strategy total return | **−83%** |
| median buy & hold | **+251%** |
| pooled win rate | 14.4% |
| mean net return / trade | −0.30% |
| profit factor | 0.39 |

**k-sweep** (robustness check) — profit factor 0.38–0.45 across `k ∈ {0.25, 0.5, 0.75, 1.0, 1.5}`, mean return/trade negative at every `k`, 0% profitable throughout. This is **not** a parameter-tuning problem.

Mechanism: a ~15% win rate. Most breakouts fail; the trailing safety-line stop chops the position out for many small losses, and the rare winners don't cover the losses plus costs.

### Signal vs costs — the edge is negative even at zero cost
Re-running k=0.5 with **all costs set to zero**:

| metric | value |
|---|---|
| mean **gross** return / trade | **−0.083%** |
| profit factor (gross) | **0.73** |
| win rate | 16.3% |
| **average holding period** | **1.2 days** |
| mean winner / mean loser | +1.41% / −0.91% |

Two things stand out. (1) The signal has **negative expectancy before any costs** — this is not a cost problem, the entry has no edge. (2) The **average hold is ~1 day**: the safety line is the *last* (steepest, nearest) support/resistance, so the trailing stop sits right under price and gets knocked out on the next bar's normal wobble. In practice this isn't "swing" holding at all — it's immediate whipsaw. That is the *least-robust-line* fragility hitting the **exit** as well as the entry.

Raw numbers: `backtest_results.txt`. Chart: `charts/backtest.png`.

## Scope / caveats
This refutes **this literal, single-daily-timeframe** version. It does **not** test the author's full approach: multi-timeframe confluence (months → weeks → days → 4h), the implicit "don't draw right up to the last candle", or discretionary judgement. Those remain open.

## Option (b): minimum bar-span on the action/safety lines — also fails

Pre-registered central value **S = 20 trading days**, with an S-sensitivity sweep. The min-span
fix *does* cure the whipsaw mechanically (holding period scales 4.6 → 35 days with S, win rate
rises), but it does **not** create an edge — it makes the edge **worse, monotonically**:

| S (days) | hold | gross PF | net PF | gross mean/trade | % profitable |
|---:|---:|---:|---:|---:|---:|
| 5  | 4.6d  | 0.80 | 0.63 | −0.24% | 0% |
| 10 | 8.7d  | 0.77 | 0.62 | −0.37% | 0% |
| **20** | **15.8d** | **0.72** | **0.57** | **−0.62%** | **0%** |
| 40 | 26.8d | 0.68 | 0.53 | −0.89% | 0% |
| 60 | 35.0d | 0.65 | 0.49 | −1.12% | 0% |

Profit factor and per-trade edge decline smoothly with S and never approach breakeven (1.0),
gross or net. The pre-registered S=20 sits on a uniformly losing surface — no plateau, no spike.
Making the lines *more* established makes entries *worse*: breaking a longer, shallower line takes
a bigger move, after which mean-reversion is stronger. Chart: `charts/backtest_span.png`.

## Author-faithful exit (safety line only, no ATR floor) — no different

The author's literal exit is to track the safety line itself, with no separate disaster stop.
Re-running with the ATR floor removed (`stop_atr=None`) at k=0.5:

| exit rule | S | gross PF | net PF | hold | % profitable |
|---|---:|---:|---:|---:|---:|
| pure safety line | 0  | 0.74 | 0.40 | 1.2d  | 0% net |
| **pure safety line** | **20** | **0.72** | **0.57** | 17.2d | **0% net** |
| 2·ATR floor + trail | 20 | 0.72 | 0.57 | 15.8d | 0% net |

The pure exit at S=20 is **identical** to the floored version (PF 0.72 gross / 0.57 net). The exit
rule doesn't matter — the entries have no edge, so how we manage them barely moves the result. Your
prediction was right on the mechanics: the 20-day minimum lifts the bare safety line's hold from
1.2 to 17 days. It just doesn't create profit. Raw: `backtest_pure_results.txt`.

## Overall conclusion
On a single daily timeframe, this method has **negative expectancy before costs**, and the result is
**robust to everything we can vary**: the band `k`, the min-span `S`, and the exit rule (pure
safety line vs ATR floor). The least-bad configuration (pure exit, S=0) is still −0.08% gross per
trade — no edge even before costs, and costs then bury it. The fragility fixes address the
*mechanism* (whipsaw); none creates an *edge*.

Multi-timeframe is very unlikely to change this: a higher timeframe is just a coarser (longer-span)
version of the same daily highs/lows, and the S-sweep — our proxy for that axis — got monotonically
worse. A rolling-lookback window is the only remaining small untested variant.

**Verdict: the method as it can be faithfully specified on daily bars does not work.**

## Addendum: does the entry geometry predict success?

A follow-up asked whether the probability of a trade winning is a function of the setup geometry —
the action/safety line **gradients**, their **separation** at entry, and their **spans**. Guard
against data-snooping: relationships were measured on the earlier half of ~11,900 trades and
re-checked on the later half; the summary is a logistic AUC on the held-out half.

**Yes — win *probability* is genuinely predictable, out-of-sample.** Three effects persist in both
halves: a **narrow action–safety separation** (tight channel), a **shallow safety gradient**, and
**low wedge convergence** all raise the win rate. Combined logistic AUC = **0.61 out-of-sample**
(~7 SE above chance). Ranked by predicted score, out-of-sample win rate climbs monotonically from
**9% (worst decile) to 25% (best)**. Dominant feature: separation (weight −0.34).

**But it is not profitable.** The features sort hit-rate, not payoff (Spearman vs trade *return* ≈ 0):
higher-win-rate tight channels also produce *smaller* wins. Even the best-geometry decile is only
**gross break-even** (+0.06%/trade at the very top), and after costs **every decile is net-negative**
(best −0.15%). Chart: `charts/setup_analysis.png`; data cached in `setup_trades.parquet`
(`analyze_setup.py`).

**Volume behaves identically.** Adding breakout-bar volume (relative to its 20-bar median) — the
README's question 3 — tells the same story. Volume has ≈0 correlation with the trade *return*
(Spearman −0.005 in / +0.006 out), so it does not predict move size. It *does* modestly predict win
probability (out-of-sample AUC rises 0.599 → 0.615 when volume is added; win rate climbs 15% → 20%
from the lowest to highest volume quintile). But every volume quintile is gross- and net-negative:

| breakout-volume quintile (OOS) | median vol× | win % | gross/trade | net/trade |
|---:|---:|---:|---:|---:|
| 1 (lowest) | 0.68× | 15.4% | −0.035% | −0.251% |
| 5 (highest) | 2.03× | 19.7% | −0.040% | −0.256% |

This is the study's most illuminating result: it shows *why* the method is a consistent near-miss —
the setups it picks are essentially random on expectancy. Every lever we found — channel geometry
*and* volume — re-sorts *whether* a trade wins, never *how much*. None predicts the payoff, so none
can make the method pay.

## Second addendum: futures instead of single equities

The author reportedly trades futures. Single equities are hostile to breakout methods (they
mean-revert); trends *persist* in commodities/FX/rates, the home of trend-following. Re-run on a
diversified 22-market futures basket (energy, metals, ags, rates, FX, + S&P/Nasdaq), ~25 years,
futures costs (5 bps/side, no financing; front-month continuous data, roll gaps and all):

- **Far more sympathetic.** Base case gross profit factor **0.97** (vs 0.73 on equities); 45% of
  markets gross-profitable individually (silver +61%, cotton, soybeans, copper …).
- **The min-span lever *flips sign*.** On equities, longer lines hurt; on futures they help
  monotonically (they let trades ride trends instead of whipsawing). Full-sample net went positive at
  S≥40 (net PF 1.04→1.06, +0.09→+0.16%/trade).

**But it does not survive out-of-sample.** Split at 2013 (train 2000–12, test 2013–25):

| S | train net PF | test net PF | test net mean | test gross PF |
|---:|---:|---:|---:|---:|
| 40 | 1.09 | 0.99 | −0.020% | 1.04 |
| 60 | 1.18 | 0.96 | −0.093% | 1.00 |

The positive result was almost entirely the **2000–2012 commodity supercycle**. Post-2013 the edge
decays to the line — gross a hair above break-even, net break-even-to-negative, and the test-era gross
(~+0.08%/trade at S=40) is ~1.4 SE from zero, i.e. indistinguishable from nothing. What the method
harvested was the **time-series-momentum / managed-futures premium** — real, documented, and largely
absent since ~2012. Code: `download_futures.py`, `load_futures.py`, `backtest_futures*.py`.

**Final position:** no *live*, exploitable edge on either equities or futures. But futures reframe the
verdict honestly — the idea captured a real premium that has faded, rather than being worthless.

## Regime-gate follow-up — does volatility steer direction? (2026-09-07)

Testing the intraday scalper's "follow breakouts in expansion regimes" idea on 27 years of daily FTSE
(20 names), as a gate over the cached breakout trades, plus a mean-reversion (fade) counterpart. Causal
regime = `vol_state` = 20-day / 100-day realised vol (>1 = expansion). Full harness: year-block
bootstrap, beta-neutral alpha, concentration, pre/post-2013 split, buy-and-hold. Code:
`regime_gate_ftse.py`, `fade_leg_ftse.py`; output in `*_results.txt`.

**Breakout (follow), gated — decisive negative, and the regime steers the WRONG way.**
Expansion gross −0.109%/tr vs compression −0.065% — both negative, expansion *worse*. Year-block
bootstrap gross −0.109% CI [−0.155, −0.070] (significantly negative); beta-neutral alpha −0.077%
CI [−0.124, −0.034] (not a beta artefact); net-positive names 0/20, years 0/27; pre- and post-2013
both negative; per-ticker compound median −52% vs buy-and-hold +251%. The intraday positive does **not**
replicate — it was instrument/timeframe-specific noise.

**Fade (mean-reversion), gated to compression — a real gross edge, killed by costs.**
z = (close − MA20)/SD20; fade |z| ≥ 2 back to the mean. Compression gross +0.409%/tr (win 59%, PF 1.26)
vs expansion +0.220% — here the regime steers the **right** way, and the gross edge is significant and
survives beta-neutralisation (+0.283% alpha, CI [+0.149, +0.417]). But net of spread-bet costs it is
marginal: +0.062%/tr, bootstrap CI [−0.069, +0.197] crosses zero; net profit concentrated in the top-3
names/years (>100% of total); net-positive only post-2013 (pre-2013 net −0.048%); per-ticker compound
median −13.7%, still far below buy-and-hold. A genuine reversion signal, regime-consistent — an order of
magnitude too small to clear costs live.

**Takeaway:** on daily single-name FTSE, momentum has no edge (even gross) and the vol regime steers it
the wrong way; mean-reversion has a real, beta-neutral gross edge that compression sharpens — but costs
erase it. The same programme wall, now seen from both sides.

## Relative action-vs-safety line duration — does it predict a valid entry? (2026-09-07)

New instrumentation records each line's endpoints and touch positions (`signals.line_touch_bounds`, extra
`Trade` fields; study in `line_duration_study.py`, output `line_duration_study_results.txt`). Tests whether
the **relative** duration of the action (broken) vs safety (opposite) line predicts win and payoff, over
five definitions of "duration" (span between defining vertices; age since inception; age since the recent
vertex; touch-span including later touches; age since last touch), guarded in-sample/out-of-sample.

*Structural note:* the latest hull line always terminates at the most recent bar, so `age_from_b` is
constant (=1) and `span_ab ≈ age_from_a − 1` — "duration" is carried by where the line **starts** and by
its **touches**, not its recent endpoint.

**Result — a real hit-rate signal, but payoff-blind (same wall as the geometry study).** Relative duration
predicts **win probability**, consistently IS/OOS: breaking the *shorter-lived / fewer-touch* action line
(relative to the safety line) wins more often. The touch-informed definition (`rel_touch_span`, which counts
later touches — the "extra touch point" subtlety) is the strongest (OOS Spearman vs win −0.068), edging raw
span/age (−0.046). Direction matches Lance's intuition: breaking a long-established line while the opposite
line is young reverts more often. **But it does NOT predict payoff** — OOS Spearman vs gross ≈ 0
(|rho| ≤ 0.02, p > 0.08 for every definition), deciles show no gross trend. Gating to the favourable
(shorter-action) third lifts win rate to 18% and trims the bleed, yet the book is still significantly
negative gross (−0.051%/tr, CI [−0.092, −0.014]) and net (−0.268%, CI excludes 0), beta-neutral alpha
−0.045% (CI just excludes 0), 0/20 names profitable, median compound −40% vs buy-and-hold +251%.

**Takeaway:** the relative-duration idea is *real* — signal validity does differ with the relative age of
the two lines, and the touch-based definition is best — but like every other geometry feature it sorts
*whether* a trade wins, not *how much*, so it cannot turn the method net-positive.

*Cross with the vol regime (`duration_regime_cross.py`):* within compression regimes relative duration
*still* does not predict payoff (OOS Spearman vs gross ≈ 0, |rho| ≤ 0.03, p > 0.11), and the doubly-gated
book (compression AND action-line shorter-lived) is still significantly negative — gross −0.049%/tr
(CI [−0.095, −0.003]), net −0.266%, beta-neutral −0.060% (CI excludes 0), 5% of names profitable. Stacking
two hit-rate levers raises the win rate to ~19% but never the payoff. **The trend-line breakout is closed:
no geometry, volume, min-span, exit rule, vol regime, relative line duration, or combination makes it
net-positive.** The only real gross edge in the daily FTSE work remains the mean-reversion fade in
compression — which costs still erase.
