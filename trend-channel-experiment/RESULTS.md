# Trend-channel experiment — RESULTS

**Question:** can trading a rising channel — go long near the bottom, sell near the top, repeat — beat
buy-and-hold? The motivation was that some FTSE names show long, unbroken uptrends inside a fairly parallel
rising channel.

**Method (causal, all 120 FTSE names, ~27y — no cherry-picking the good charts).** At each bar, fit an OLS
regression channel on ln(close) over a trailing L-bar window; bands = fitted line ± 2σ of residuals
(parallel by construction; gradient = trend, width = 4σ). Long-only, acting at the next open: enter when the
channel *qualifies* (annualised gradient ≥ 10%, R² ≥ 0.80) and the close is in the bottom quarter of the
channel; exit at the top quarter, on a close below the lower band (break), or when the gradient turns
negative. Costs 10 bps/side + 5%/yr financing while held. Code: `channel.py` (fitter, unit-tested in
`test_channel.py`), `channel_backtest.py`; raw output `channel_backtest_results.txt`.

## Verdict: it does NOT beat buy-and-hold — decisively

L=250, net of costs, 7,918 trades over 120 names:

| metric | strategy | buy & hold |
| --- | --- | --- |
| median total return | **−4%** | **+160%** |
| median total return (gross, no costs) | +20% | +160% |
| median Sharpe | +0.02 | +0.28 |
| % of names it beats — total return | **24%** | — |
| % of names it beats — Sharpe | **10%** | — |
| per-trade net | +0.088% (win 38%, PF 1.06) | — |
| **time in market** | **~9%** | 100% |

**Why it fails is the ~9% time-in-market.** Buying only the bottom quarter of a channel and selling the top
quarter leaves you in cash ~91% of the time, so you forgo the upward drift that made the trend attractive in
the first place. The per-trade edge is real but tiny (+0.088% net); the oscillation you can harvest is an
order of magnitude smaller than the drift you give up by being flat.

**Robust** across window length — L=120/250/500 all give median-negative strategy totals and only 20–24% of
names beating B&H, Sharpe always worse.

**The irony:** the strongest trenders — the very "long unbroken uptrends" that motivated the idea — are
where the strategy loses by the *most* (RR +512% vs B&H +2159%; HILS +104% vs +4675%; AEP +126% vs +4226%).
In an unbroken uptrend the right move is to *hold*, not to sell each channel top and miss the continuation.

**Conclusion.** Selling near the top of a rising channel is systematically the wrong side of a name that
keeps trending; buy-and-hold wins on both return and Sharpe, across the universe and every window length.

## Fix attempts (2026-09-07)

### Ratchet exit (let winners ride) — makes it WORSE, not better

Replaced the sell-at-the-top exit with a trailing ratchet (`channel_ratchet.py`): arm near resistance and
track the rising resistance line at a fraction of the channel width below it (sweep 0.25/0.5/0.75), or a
Chandelier stop (highest close − M·ATR, M∈{2,3,4}). Channels precomputed once; policies swept cheaply.

Every ratchet variant **underperforms the sell-at-top baseline**: median strat total −15% to −26% (vs +8%
for "top"), beats B&H on 9–12% of names (vs 22%), per-trade net +0.13% (vs +1.56%). Reason: in a *genuine*
channel the upper band is where price reverts, so selling there is correct — trailing past it gives the
gains back on the down-swing. Ratchets help in *unbounded* trends; "qualifying channel" selects for
boundedness, so a ratchet is the wrong tool.

*Correction to the first backtest:* the original `c < lo` disaster stop sat inside the entry zone (you buy
near the lower band), so it often exited one bar after entry. Removing it lifts the fair channel baseline
from −4% to **+8% median (TIM 32%, net +1.56%/trade)** — still a decisive loss to B&H +160%, but with a
healthy per-trade edge, which motivates the portfolio test below.

### Capital rotation (portfolio) — solves exposure, still loses to B&H

`channel_portfolio.py` schedules trades into S slots, filling each free slot with the highest-gradient
channel-bottom signal available that day (so idle cash is redeployed into the next name at the bottom of its
channel — the intended fix). Rotation genuinely **fixes time-in-market: exposure 73–89%.** But every slot
count loses to equal-weight buy-and-hold (total +1068%, CAGR 9.57%, Sharpe 0.65): best is S=10 at total
+310% (CAGR 5.38%, Sharpe 0.43).

**The decisive diagnostic:** return per year *while deployed* (S=1, 89% exposure) is **+5.4%/yr — below the
market drift of +9.57%/yr.** The premise that a channel round-trip gains *faster* than buy-and-hold is false:
the median bottom-to-top hold is **138 calendar days (~4.5 months)**, so a +1.56%/trade capture annualises to
only ~4–5%/yr — slower than simply holding through the drift. Selling at the top and rotating swaps a
faster-drifting hold for a slower oscillation-capture; fully deployed or not, you cannot out-compound a
drift you are capturing more slowly.

### Decomposition: selection vs winner-capping (`channel_decompose.py`)

Is the shortfall because channel-forming names grow slowly, or because selling at the top caps winners?

- **Selection: none.** All 120 names form ≥1 qualifying rising channel over the 27 years — "channel-formers"
  *is* the whole universe (median single-name B&H CAGR +3.95%, equal-weight-hold +9.57%). The channel names
  are not a slow subset; they are everything.
- **Winner-capping: the whole story.** The strategy captures a tiny fraction of the big winners — median
  capture ratio **0.04** on the top B&H quartile (median B&H +1465% → strat +38%), and ~0 or negative on the
  top-10 winners (GDWN +23,828% B&H → strat −3%; ANTO +4,099% → −9%; best was HLMA at 0.22). On the two
  smallest-growth quartiles the strategy is net-negative (costs + time out). After a channel-top exit the
  name is still **+2.37% higher 60 trading days later (58% still rising)** — continuation systematically left
  on the table.
- Secondary: equal-weight B&H earns a large rebalancing/diversification bonus (portfolio CAGR +9.57% vs mean
  single-name +4.13%) that a part-time rotating book does not capture.

### Ride-the-winner exit (hold until the BOTTOM breaks) — best variant, still loses (`channel_ride.py`)

The fix for winner-capping: don't sell at the top — hold until price breaks below the lower band by
BREAK_FRAC of the channel height (optionally also exit if the gradient turns negative), then rotate the cash.

**Winner-capping is largely fixed.** Capture of the big winners jumps from ~0.00 to a median ~0.4: GDWN
+23,828% B&H → strat +11,584% (capture 0.49), HILS 0.69, NXT 0.96, AEP 0.40, DOM 0.44. Median hold ~2.2
years at break=0.5 — it genuinely rides trends now.

**But it still loses to B&H.** Best config (break=0.5, no gradient exit, 5 slots): total +461% (CAGR 6.62%,
Sharpe 0.44), exposure 95% — vs B&H +1068% (CAGR 9.57%, Sharpe 0.65). Every config loses on both return and
Sharpe, though the gap narrowed (CAGR 5.4% → 6.6%). Two residual reasons the decomposition predicted: (1) it
still captures only ~half the big winners, because entry needs an *established* 250-day qualifying channel
plus a dip, so it misses each winner's initial launch and loses chunks to break-and-re-enter; (2) the
equal-weight-B&H rebalancing/diversification bonus (portfolio CAGR 9.57% vs mean single-name 4.13%) can't be
matched by a 5–20-name rotating book. Tuning: break=0.5 is the sweet spot (0.0 whipsaws out in a day since
entry sits on the band; 1.0 holds too long); not exiting on gradient is better.

### Entry sweep: window × gradient × width (`channel_entry_sweep.py`)

Swept L∈{100,150,250}, min gradient∈{0.10,0.20,0.30}, min relative width∈{0,0.15,0.25}, ride exit, 5 slots.

- **Gradient (4): non-binding** — identical results across every g_min. An R²≥0.80 rising channel already
  implies a steep gradient, so a gradient floor changes nothing (redundant with the R² gate).
- **Width (3): the strong lever** — at L=100, w_min 0→0.15→0.25 lifts CAGR 3.24→4.88→**8.84%** (Sharpe
  0.26→0.33→0.54).
- **Window (1): shorter helps at wide channels** — at w_min=0.25, L=100 (+877%) > L=150 (+415%) > L=250 (+283%).
- **Best: L=100, w_min=0.25 → +877% (CAGR 8.84%, Sharpe 0.54)** vs B&H +1068% (9.57%, 0.65) — the closest yet,
  still losing on both return and Sharpe.

*Caveat (to be checked):* the best cell has only 117 trades (best-of-27, concentrated), and the width lever
is suspected to work by **degeneration** — a wide channel with a 0.5×width stop keeps the stop far away, so
the strategy holds a few volatile names for years, converging *toward* buy-and-hold rather than beating it.

### Degeneration check — the decisive test (`channel_width_check.py`)

Compare the strategy's CAGR to buy-and-hold of **the same names it trades** (L=100, ride 0.5), sweeping width:

| w_min | names | trades | med hold | strat CAGR | B&H of traded names | edge |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 120 | 574 | 952d | 4.75% | 9.57% | −4.82% |
| 0.15 | 113 | 395 | 1077d | 4.88% | 9.78% | −4.90% |
| 0.25 | 64 | 117 | 1072d | 8.84% | 11.38% | −2.54% |
| 0.35 | 28 | 42 | 1334d | 5.86% | 11.61% | −5.75% |

**The timing edge is negative at every width.** The width lever was pure name-selection: wider channels
select higher-return names (their B&H CAGR rises to 11.6%), and the impressive 8.84% at w_min=0.25 came from
names that would have returned 11.38% if simply held — the overlay captured *less*. Even with median holds of
3–4 years (nearly buy-and-hold already), the strategy loses to holding the same names, because it sits out
the post-break re-entry gaps.

### Touch-defined channel (idea #2) — also negative edge (`channel_touch.py`)

Defining/validating the channel by **≥N touches of each band** (replacing the R² gate) rather than by fit
quality does not help. Across L∈{100,250} and N∈{2,3,4}, the edge vs buy-and-hold of the traded names is
negative everywhere (−4.7% to −8.9%/yr); requiring more touches makes it worse (later entries), and the
touch gate doesn't even select a special subset (all 120 names trade, traded-B&H = universe 9.57%). So the
negative-edge verdict holds across **two independent channel-definition families** (fit/width and touches).

### Channel as a SELECTOR, not a timer — beats B&H (`channel_select.py`)

The comparisons above are all vs B&H of the *same names* — but the channel is what *selects* those names.
Testing pure selection (hold an equal-weight book of every name **currently** in a qualifying rising channel,
causal, no dip-timing; drop it when the channel stops qualifying) changes the verdict:

| L | w_min | avg held | turnover/yr | net CAGR | Sharpe | vs universe B&H (9.57%, 0.65) |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 250 | 0.00 | 14 | 4.8 | 11.25% | 0.66 | WINS |
| 250 | 0.15 | 10 | 7.0 | 10.41% | 0.61 | WINS |
| 250 | 0.25 | 4 | 9.3 | 14.61% | 0.69 | WINS |
| 100 | 0.15 | 3 | 21.6 | 10.56% | 0.55 | WINS (others lose) |

Causal and net of costs; L=250 wins at every width. And the causal book beats even holding those names
*forever* (ex-post upper bound 10.20% at w_min=0.25 vs causal 14.61%), so the **exit** (leaving a name when
its channel stops rising) adds drawdown-avoidance value — this is a trend/quality filter, not just picking.

**Caveats before belief:** the best cell holds only ~4 names (concentrated, high variance); it is essentially
a momentum/trend filter (real but decaying elsewhere in the programme), so needs an out-of-sample split; and
survivorship inflates both sides (the *relative* edge is the trustworthy part).

### Selection stress — FAILS out-of-sample and concentration (`channel_select_stress.py`)

- **OOS split @2013: the edge is pre-2013 only.** Selection net CAGR pre/post: 16.7%/**6.2%** (w=0),
  14.0%/**7.0%** (w=0.15), 21.8%/**8.0%** (w=0.25) — vs universe 10.2%/**8.98%**. It crushes B&H before 2013 and
  **loses to B&H after 2013** in every config. Same signature as the futures TSMOM premium: momentum real but
  decayed post-2012.
- **Concentration: a few names carry it.** Dropping the top-3 contributing names takes it below the universe
  (11.25% → 7.82% at w=0; 14.61% → 6.80% at w=0.25). The same name (AEP) tops every list — not diversified.

So the selection edge is the decayed momentum premium, concentrated in ~3 names; it is **not a live,
exploitable edge** and does not beat buy-and-hold out-of-sample.

## Reframed conclusion

The channel is worthless as a **timer** but useful as a **selector**. Every dip-buy/top-sell/ride *timing*
overlay has strictly negative edge versus holding the same names (−2.5 to −8.9 pp/yr, across exit rule,
rotation, window, width, gradient, and touch-count definition). But holding an equal-weight book of names
*currently in a qualifying rising channel* — pure selection, causal, net of costs — beats equal-weight
buy-and-hold on both return and Sharpe (L=250, robust across width), and beats holding those names forever,
because the exit side-steps their later drawdowns. **But that selection edge fails the stress tests:** it is
entirely pre-2013 (post-2013 it loses to B&H, 6–8% vs 8.98% CAGR) and concentrated (dropping the top-3 names
sinks it below the universe). It is the programme's familiar **decayed momentum premium** — real once, gone
since ~2012 — not a live edge. So neither timing nor selection gives a buy-and-hold-beating strategy that
survives out-of-sample.

## Original (timing) conclusion
The proposed fixes each did real work: **rotation** fixed exposure (~90%), the **ride-the-winner** exit fixed
winner-capping (capture 0.04 → 0.4), and a **minimum width** filter found the high-return names (best cell
L=100, w_min=0.25: CAGR 8.84% vs universe 9.57%). But the degeneration check settles it — the channel-timing
decision itself has **strictly negative edge versus holding the same instruments**, in every configuration
tried (exit rule, rotation, window, width, gradient, and touch-count definition), by 2.5–8.9 pp/yr. There is no dip-timing skill to
harvest; the strategy can only degenerate toward buy-and-hold from below. Buy-and-hold wins on return and
Sharpe. Experiment closed.

The decomposition pins down *why*: **not** slow instruments (there is zero selection — every name forms
channels), but **winner-capping** — taking profits at every channel top clips the right tail of big
multi-year winners, which is exactly where buy-and-hold's return lives. Buy-and-hold wins on both total
return and Sharpe. Experiment closed.
