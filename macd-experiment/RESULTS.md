# MACD — results

> **Portfolio construction (important).** The strategy is a **concentrated** book:
> each day capital is equal-weighted across just the names currently signalling long
> (fully invested; cash only when none signal), rebalanced daily. This is the
> portfolio approach — capital flows into the qualifying names, it never sits idle —
> and is the same lesson as the momentum work (`momentum_survivorship_free.py`
> equal-weights the *selected* names, not the whole universe). The benchmark is the
> whole eligible universe, equal-weight, always invested. *An earlier draft weighted
> the strategy across the whole universe, leaving ~90% of capital in cash; that
> penalised MACD for a cash-drag a real portfolio would not carry, and is corrected
> throughout below (`concentrate=True` in `backtest.portfolio`).*

## Phase 1 — standard 12/26/9 MACD, long-only (FTSE / EODHD UK)

**Verdict: standard MACD loses to buy-and-hold — even gross of costs, and on both
halves of the sample. Confirms the programme's prior.**

Setup: survivorship-free EODHD UK panel (active + delisted), names with ≥ 750
observations, cleaned by capping daily log returns at |r| ≤ 0.6 (kills bad
`adjusted_close` ticks). Investable universe = top-350 by trailing-1y turnover,
refreshed point-in-time and lagged one day. Textbook MACD line (12/26) crossing its
signal line (9); long while MACD > signal, concentrated book as above.
Reproduce: `run_phase1_baseline.py` (→ `phase1_baseline_results.txt`, `phase1_baseline.png`).

| Full sample (1998–2026) | CAGR | Sharpe | maxDD |
|---|---|---|---|
| Buy-and-hold (equal-weight) | **+12.55%** | **+0.65** | −61.9% |
| MACD timing, **gross** | +3.03% | +0.25 | −92.0% |
| MACD timing, net 10 bps | −7.89% | −0.28 | −96.0% |

Out-of-sample split is the same story: pre-2013 B&H +8.2% vs MACD net −13.1%;
2013-on B&H +17.6% vs MACD net −2.4% (Sharpe −0.01 — flat, still far below B&H).

### Why it fails (the mechanism, not just the number)

Trade-level decomposition (gross, per-name holding spells, investable universe):

- **101,112 trades — hit-rate 37.8%, mean payoff +0.51%, median −1.05%.**

That is the textbook trend-following signature: a **low hit-rate with a positive mean
payoff** — most trades are small losers, a few are large winners. So the per-trade
edge is genuinely positive *gross*. It still loses because:

1. **The selected names under-earn the market.** MACD-long name-days earn +4.8 bps
   gross vs the market's +5.7 bps/day — the names it picks as "trending up" do
   *worse* per day than the average name. Even fully invested, that loses to B&H.
2. **Concentration wrecks diversification.** Holding only the currently-trending
   names is undiversified, so drawdowns blow out to −92% to −96% vs B&H's −62%.
3. **Costs finish it.** +3.03% gross → −7.89% net at 10 bps — the crossover churns.

This matches the programme's headline findings directly: **direction timing is beaten
by buy-and-hold**, and the edge that exists is in *hit-rate/payoff shape*, not in
tradable net return.

### Implication for the modifications

The proposed trend-vs-flat filters (third EMA, ATR/volatility gating) are, per the
programme's trend-line work, expected to move **hit-rate, not payoff** — i.e. to
change *which* trades fire, not the net economics. Phase 2 (sweep + stability of the
standard form) is worth running to confirm no robust parameter plateau beats B&H
before spending effort on the modifications; the honest prior is that none will.

## Phase 2 — parameter sweep, stability, permutation null

**Verdict: no parameter set beats buy-and-hold, there is no plateau, and MACD on
real data is *worse* than MACD on shuffled noise (p = 1.000). The standard MACD
hypothesis is comprehensively falsified — the trend it follows is anti-predictive.**

Swept fast ∈ {5,8,10,12,15,19,24} × slow ∈ {20,26,32,40,50,60,80} (signal = 9), net
of 10 bps, over the 1206 ever-eligible names. Reproduce: `run_phase2_sweep.py`
(→ `phase2_sweep_results.txt`, `phase2_sweep.png`).

- **Sweep:** every one of the 48 valid combinations loses to equal-weight B&H, by
  −17% to −23% CAGR. The heatmap is uniformly red. **0 of 48 beat B&H net.**
- **Stability:** the least-bad point (24/60, −17.0%) is an isolated non-robust cell,
  not a contiguous plateau — and still deeply negative.
- **Permutation null (200 surrogates, per-name time-shuffle):** real 12/26/9 excess
  CAGR **−20.4%** is *worse than the whole null distribution* (mean −13.8%, sd 2.3%,
  95th pct −10.1%); **p = 1.000** — every surrogate beat the real strategy. So the
  real trend structure is not merely uninformative, it is **actively worse than
  random**: shuffling away the serial structure *improves* MACD. That is the
  signature of short-horizon mean-reversion — the crossovers MACD chases tend to
  reverse, so following them underperforms even random timing.

## Phase 3 — volatility-regime gate (first modification)

**Verdict: under the correct concentrated portfolio the gate strictly *hurts* — it
moves hit-rate but worsens net return and Sharpe. Prior holds, more firmly.**

Gate MACD long entries to days where each name's trailing-20d realised vol clears its
own trailing q-quantile (point-in-time). Swept q at 12/26/9, net 10 bps, over the 1206
ever-eligible names, concentrated book. Reproduce: `run_phase3_volgate.py`
(→ `phase3_volgate_results.txt`).

| gate q | excess CAGR vs B&H | Sharpe | hold-days | hit-rate | mean payoff | bps/hold-day |
|---|---|---|---|---|---|---|
| 0.00 (ungated) | **−20.44%** | **−0.28** | 1,079,774 | 37.8% | +0.507% | +4.80 |
| 0.30 | −21.47% | −0.30 | 228,458 | 44.8% | +0.553% | +7.42 |
| 0.50 | −23.18% | −0.37 | 170,084 | 45.9% | +0.574% | +8.38 |
| 0.70 | −23.65% | −0.40 | 109,578 | 47.3% | +0.516% | +8.49 |

*Average market day (B&H, gross): +5.70 bps/day.*

- **The gate makes it worse, not better** (−20.4% → −23.7% excess CAGR; Sharpe
  −0.28 → −0.40). An earlier draft showed the gate "helping" (−35% → −15%), but that
  was entirely an artifact of the flawed idle-cash construction, where cutting
  participation cut cash-drag. Once capital is properly concentrated, discarding
  positions just reduces diversification and hurts.
- **Hit-rate still rises +9.5 pts** (37.8% → 47.3%) with flat per-trade payoff — the
  textbook "a filter moves how often you win, not how much".
- **The tempting bit** — gated hold-days out-earn the average market day (+8.5 vs
  +5.7 bps) — is risk compensation (you only hold during volatile stretches), not
  skill: it does not survive into portfolio Sharpe or net return.

So the modification does **not** rescue MACD; it degrades it. It confirms the
mechanism — the filter steers hit-rate and participation, never risk-adjusted payoff.

## Phase 4 — low-for-long / high-for-short sourcing (second modification)

**Verdict: this is the one modification that genuinely improves MACD's risk-adjusted
performance — and it is *not* the idle-cash artifact (participation is unchanged). But
it still loses to B&H by 16% with negative Sharpe. It bleeds less; it does not create
an edge.**

Enter on a MACD cross computed from **low** prices (stricter uptrend confirmation),
exit on a cross from **high** prices (hold through minor dips). All four entry/exit
sourcings, concentrated book, net 10 bps. An integration check confirms close/close
reproduces the Phase 1 baseline exactly. Reproduce: `run_phase4_hilo.py`
(→ `phase4_hilo_results.txt`).

| entry / exit | excess CAGR vs B&H | Sharpe | hold-days | hit-rate | mean payoff |
|---|---|---|---|---|---|
| close / close (baseline) | −20.44% | −0.28 | 1,079,774 | 37.8% | +0.507% |
| **LOW / HIGH** | **−16.33%** | **−0.07** | 1,081,463 | 34.8% | +0.457% |
| low / close (entry only) | −18.35% | −0.17 | 1,063,744 | 34.0% | +0.447% |
| close / high (exit only) | −18.12% | −0.16 | 1,096,622 | 34.9% | +0.471% |

- **Real, not an artifact.** Unlike the vol gate, hold-days are essentially unchanged
  (~1.08M) — the improvement is not from trading less. Both legs contribute ~2%: the
  low-entry filters false starts, the high-exit holds through whipsaws. Fewer whipsaws
  lift Sharpe (−0.28 → −0.07) and cut hit-rate (37.8% → 34.8%) with similar payoff.
- **Still not tradable.** Sharpe is negative and excess CAGR is −16% — it loses to
  buy-and-hold by a wide margin. The sourcing reduces MACD's self-inflicted whipsaw
  damage but cannot overcome the anti-predictiveness the permutation null exposed.
  (A permutation null on this variant would confirm whether the residual is signal or
  noise; given it remains a large net loss, it does not change the decision.)

## Overall conclusion

Standard MACD is dead for this use, tested under the correct **concentrated**
portfolio (capital in the signalling names, not idle cash — the momentum-work lesson).
It is not "a real edge killed by costs": the permutation null shows MACD on real data
does **worse than every shuffled surrogate** (p = 1.000). The trend it follows is
*anti-predictive* — short-horizon reversal means chasing crossovers underperforms even
random timing. This is the strongest form of the programme's prior ("direction is a
mirage").

**Modifications tested.** The volatility gate (Phase 3) *degrades* the concentrated
book. Low-for-long / high-for-short sourcing (Phase 4) is the one variant that
genuinely helps — fewer whipsaws lift Sharpe from −0.28 to −0.07 at equal
participation — but it still loses to B&H by 16% with negative Sharpe: it reduces
MACD's self-inflicted damage without creating an edge. The remaining ideas
(retrenchment, separate exit params) reshape the same anti-predictive crossover and,
per the null and the trend-line follow-ups, can only move *which* trades fire, never
manufacture a positive net payoff. The prior — now confirmed four ways (loses gross,
loses on every swept parameter, worse-than-noise null, and no modification rescues it)
— is a firm "no". **Standard MACD and its modifications do not beat buy-and-hold on UK
equities.**

**Methodology note.** The idle-cash version of the portfolio (whole-universe weights)
inflated the loss and made the vol gate look helpful; both reversed once corrected.
The lesson — benchmark a *concentrated* selection book, never one that parks capital
in cash for non-signalling names — is recorded here to avoid repeating it.

