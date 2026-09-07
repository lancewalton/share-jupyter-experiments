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

### Rolling-window & breadth: correcting the "2012 vol-suppression" story (`rolling_window.py`, `us_selector.py`, `download_us.py`)

A trailing-3-year rolling view refutes the clean narrative. It is **not** a sharp 2012 break: the FTSE
selector edge was front-loaded 2003–08 (+8 to +36%/yr, including sitting out the 2008 crash), ~flat 2009–17,
sharply **negative 2018–24** (−6 to −15%/yr, worst in the 2020–22 COVID/bear whipsaw), and **recovered in
2025–26** (+6 to +11%).

Crucially, **corr(rolling edge, market vol) = −0.39 (FTSE), −0.45 (US) — negative.** This **refutes the
vol-suppression hypothesis**: the selector does *worse* when volatility is higher, not better. It is a
long-only equity trend filter that thrives in smooth persistent trends and gets **whipsawed by sharp vol
spikes / V-shaped recoveries** (it exits at the crash low and misses the snap-back) — the opposite vol
relationship to a long/short CTA. Vols were similar pre/post-2013 (16.5% vs 15.3%), so suppression isn't the
driver; the post-2013 shortfall is mostly the 2020–22 whipsaw plus an extremely strong B&H benchmark.

**Breadth — not UK-specific.** The US large-cap selector (30 names, 2000–26) shows the same shape: loses to
EW B&H at w_min=0/0.25 (only the 2-name w_min=0.15 cell "wins"), negative edge–vol corr (−0.45), and the
same 2020–22 whipsaw damage (−12 to −13%/yr). It loses *more* clearly in the US because US B&H post-2013 was
extraordinary (+15.5%/yr). So the phenomenon is cross-market. Bonds untested here (no cross-section), but
note bonds ran their own secular bull that broke violently in 2020–22 — a different regime.

**Correction:** "decayed momentum, dead since 2012" was too strong. The selector is a long-only trend filter
whose relative edge is regime-dependent (good in smooth trends, badly negative in volatile whipsaws),
cross-market, and recovering since 2023 in the UK. Buy-and-hold still wins over the full sample — but the
mechanism is whipsaw-sensitivity plus a strong benchmark, **not** vol suppression.

### Bonds — the mechanism confirmed by a sign flip (`download_bonds.py`, `bond_selector.py`)

6 bond ETFs (TLT/IEF/SHY/LQD/AGG/TIP, 2002–26; thin, correlated cross-section — indicative). The selector
loses to bond EW B&H on total return at every gradient (CAGR 0.3–1.15% vs 3.58%) — same drift-forgone
problem, worse because bonds drift slowly and only ~1 ETF qualifies at a time. **But the diagnostic flips:**

- **corr(rolling edge, bond-market vol) = +0.33 — positive** (equities were −0.39 / −0.45).
- Rolling edge is negative 2006–2021 but turns **positive in 2022 (+0.6%), 2023 (+3.3%), 2024 (+4.6%)** —
  exactly the bond crash and aftermath.

This confirms the corrected mechanism and kills vol-suppression: the trend-following exit **helps** when a
drawdown is a **sustained downtrend** (bonds 2022 — the selector drops bonds as channels break, avoiding the
crash) and **hurts** when it is a sharp **V-recovery** (equities 2020 — exits at the low, misses the
snap-back). The edge–vol correlation flips sign with the *shape* of the drawdown — not with a date or a vol
level. If vol-suppression were the cause, the sign would be the same across assets.

**Breadth conclusion:** the channel-selector loses to buy-and-hold on total return across UK equities, US
equities, *and* bonds (a part-time long-only filter forgoes the drift). But its trend-following value-add is
regime-shape-dependent: negative in whipsaw-prone equities, positive in sustained-trend bond drawdowns. Not
UK-specific, and not vol suppression — drawdown shape.

### Cross-strategy regime dashboard (`regime_dashboard.py`, `charts/regime_dashboard.png`)

Rolling 3-year edge (vs equal-weight FTSE B&H, gross) of four implementable strategies, to see which ideas'
regimes have turned. The annual panel shows a clear **rotation between trend and reversion**:

- **2018–2023 (whipsaw era):** channel −5 to −15%/yr, momentum ~0 to −3%, **reversal +2 to +9%** — reversion's regime.
- **2024–2026 (recovery):** channel **+11.7 / +6.3%**, momentum **+5.5 / +5.0%**, reversal turns **−4.5 / −9.4%** — trend's regime.

**12-1 momentum is the most consistent** (positive in most years, full-sample +4.2%/yr gross, only mildly
negative 2009–11 and 2020–23); the **channel selector is the most regime-sensitive** (swings +37% to −15%);
**low-vol is persistently weak** on this universe (−2.3%/yr); **reversal is anti-trend** (its good years are
the trend strategies' bad ones).

*Caveat:* the efficiency-ratio regime indicator used here is 3y-smoothed and secularly declines, so its
correlation column is confounded (all four spuriously positive) — read the annual panel, not the corr.

**Implication for the 2024–26 recovery:** of the ideas worth revisiting, **12-1 momentum is the strongest
candidate** (consistent, currently favourable, and the programme's one robust directional edge); the channel
selector is speculative (high variance); the reversion fade is in its *adverse* regime; low-vol is out. The
regime-timing caveat still applies — small sample, and the programme repeatedly found premia can't be timed
for free.

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
