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

## Phase 2 — parameter sweep, stability, permutation null

**Verdict: no parameter set beats buy-and-hold, there is no plateau, and MACD on
real data is statistically indistinguishable from MACD on shuffled noise. The
standard MACD hypothesis is comprehensively falsified.**

Swept fast ∈ {5,8,10,12,15,19,24} × slow ∈ {20,26,32,40,50,60,80} (signal = 9), net
of 10 bps, over the 1206 ever-eligible names. Reproduce: `run_phase2_sweep.py`
(→ `phase2_sweep_results.txt`, `phase2_sweep.png`).

- **Sweep:** every one of the 48 valid combinations loses to equal-weight B&H, by
  −34% to −37% CAGR. The heatmap is uniformly red. **0 of 48 beat B&H net.**
- **Stability:** the least-bad point (19/80, −34.2%) is an isolated non-robust cell,
  not a contiguous plateau — and still deeply negative.
- **Permutation null (200 surrogates, per-name time-shuffle):** real 12/26/9 excess
  CAGR **−35.5%** sits inside the null distribution (mean −34.9%, sd 2.2%,
  95th pct −31.3%); **p = 0.62**. The real serial/trend structure adds *nothing* over
  shuffled returns — the ~35% net loss is the structural cost of timing (cash-drag +
  turnover), not a real signal blunted by costs.

## Phase 3 — volatility-regime gate (first modification)

**Verdict: the gate moves hit-rate and cuts participation, not risk-adjusted payoff.
It reduces the loss but never approaches beating B&H, and adds no Sharpe.** Prior holds.

Gate MACD long entries to days where each name's trailing-20d realised vol clears its
own trailing q-quantile (point-in-time). Swept q at 12/26/9, net 10 bps, over the 1206
ever-eligible names. Reproduce: `run_phase3_volgate.py` (→ `phase3_volgate_results.txt`).

| gate q | excess CAGR vs B&H | Sharpe | hold-days | hit-rate | mean payoff | bps/hold-day |
|---|---|---|---|---|---|---|
| 0.00 (ungated) | −35.54% | −0.86 | 1,079,774 | 37.8% | +0.507% | +4.80 |
| 0.30 | −16.81% | −0.42 | 228,458 | 44.8% | +0.553% | +7.42 |
| 0.50 | −15.96% | −0.38 | 170,084 | 45.9% | +0.574% | +8.38 |
| 0.70 | −15.08% | −0.36 | 109,578 | 47.3% | +0.516% | +8.49 |

*Average market day (B&H, gross): +5.70 bps/day.*

Reading it honestly:

- **The headline number improves** (−35.5% → −15% excess CAGR) — but not by trading
  *better*. **Mean payoff per trade is flat** (+0.51% → +0.57%) and **hold-days fall
  ~10×**. The gate mostly makes the strategy *do less of a losing signal*.
- **Hit-rate rises +9.5 pts** (37.8% → 47.3%) — the textbook "a filter moves how often
  you win, not how much" signature.
- **The one genuinely interesting bit:** gated hold-days out-earn the average market
  day (+8.5 vs +5.7 bps/hold-day), where ungated MACD *under*-earns it (+4.8). So
  high-vol-regime long days do carry higher gross return per day. **But it is risk
  compensation, not skill:** Sharpe stays −0.36, and the strategy still loses to B&H by
  15% because capturing those days means sitting in cash ~90% of the time — the forgone
  market drift (cash-drag) plus turnover swamps the per-day edge.

So the modification does **not** rescue MACD: no risk-adjusted edge, no path to beating
buy-and-hold. It confirms the mechanism — timing steers participation and hit-rate; the
market's drift punishes being out of it.

## Overall conclusion

Standard MACD is dead for this use. It is not "a real edge killed by costs" — the
permutation null shows there is no timing signal at all: shuffling away the trend
structure changes nothing. This is the strongest form of the programme's prior
("direction is a mirage").

**Implication for the proposed modifications.** The trend-vs-flat gate, retrenchment,
high/low sourcing, and separate exit params are all ways to *filter or reshape* the
same crossover signal. The permutation null says that signal carries no timing
information to begin with — so a filter can only change *which* no-information trades
fire (hit-rate), never manufacture payoff, exactly as the programme's trend-line
follow-ups found. Running them is very likely wasted effort. If any is worth a single
cheap check, it is the volatility/ATR regime gate — but the prior, now doubly
confirmed, is a clear "no".

