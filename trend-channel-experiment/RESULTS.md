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

### 12-1 momentum, full rigour (`momentum_backtest.py`) — the programme's best result, caveated

Monthly-rebalanced cross-sectional momentum on the FTSE universe, net of turnover-based spread-bet costs.

**Long-only top-quintile tilt (net 10 bps):** CAGR **13.0%** vs universe B&H 8.3%; Sharpe **0.83** vs 0.58;
consistent across eras (pre-2013 13.5%/Sh 0.77, post-2013 12.6%/Sh 0.91 — *no decay*); 2023–26 CAGR 16.9%,
Sharpe 1.21; year-block bootstrap +13.7%/yr, 95% CI **[+6.2%, +20.8%] — clears zero**; robust to 20 bps
(12.4%), turnover ~5.8×/yr. Beta 0.85, so most of the return is equity beta and the **alpha over beta is
+5.75%/yr** (still significant), with equity-sized drawdowns (−47%).

**Market-neutral long-short (the pure factor):** CAGR 2.8%, Sharpe **0.25**, maxDD **−66%** (momentum
crashes), beta ≈ 0; bootstrap +5.1%/yr CI **[−5.7%, +14.1%] — spans zero** (not significant); *did* decay
(rolling 36m negative 2021–24) and is only mildly recovering (+3.8/+3.6% 2025–26). So the regime/revival
story fits the long-short; the long-only tilt was robustly good throughout.

**Two caveats.** (1) The tilt is a high-beta long book, not market-neutral alpha. (2) **Survivorship bias**
— the universe is *current* FTSE constituents, and delisted losers (exactly what momentum avoids/shorts) are
missing, inflating momentum's measured edge more than most factors. The +5.75% alpha is very likely
overstated; confirming the magnitude needs a point-in-time (survivorship-free) universe we don't have.

**Verdict:** 12-1 long-only momentum is the strongest, most robust, cost-surviving, era-consistent,
significant result in the whole programme — the one idea that beats buy-and-hold net and keeps doing so
out-of-sample and today. But it's a high-beta tilt with real-but-smaller alpha, inflated by survivorship —
"the best candidate to actually build," not "a proven 13% edge."

### Survivorship-free 12-1 momentum (EODHD, active + delisted) — the caveat RESOLVED (2026-09-07)

Rebuilt on a genuine survivorship-free UK universe from EODHD: 3,237 GBP/GBX common stocks (1,570 live +
1,667 **delisted** — Carillion, Thomas Cook, NMC, Intu, Sirius...), full OHLCV. Point-in-time top-350-by-
liquidity universe (FTSE-350-like, incl. names that were liquid then and later died), same 12-1 momentum,
monthly, net of turnover costs. Returns winsorised cross-sectionally (1/99 within eligible) to kill
adjusted_close bad ticks — the raw universe is messy (even top-350 had +1000%+ monthly ticks; an unwinsorised
first pass gave nonsense CAGRs of +50-70%). Files: `download_eodhd_uk.py`, `momentum_survivorship_free.py`,
`momentum_diagnose.py`.

**The momentum alpha survives survivorship correction — essentially intact:**

| metric | survivor-only (120 current) | survivorship-free (top-350 incl. delisted) |
| --- | --- | --- |
| eligible EW B&H CAGR | ~8–9% (inflated) | 5.65% (true) |
| long-only tilt CAGR (net 10bps) | 13.0% | 11.4% |
| long-only Sharpe | 0.83 | 0.77 |
| long-only alpha over market | +5.75% | **+6.35%** |
| OOS pre / post-2013 | 13.5% / 12.6% | 11.7% / 11.2% |
| bootstrap 95% CI | [+6.2, +20.8] | [+4.2, +19.5] (clears 0) |
| market-neutral long-short Sharpe | 0.25 | **0.65** |

**Key insight:** survivorship bias inflated the *market benchmark* (~3pp/yr) MORE than it inflated momentum
(which already avoids losers), so momentum's *alpha is robust* — marginally higher survivorship-free. And the
market-neutral long-short IMPROVES markedly (0.25 → 0.65) because shorting the delisted losers (absent from the
survivor set) pays. The earlier worry that survivorship inflated the alpha was wrong in direction: it inflated
the benchmark, not the edge.

**Caveats:** high-beta tilt (0.84, −46% DD); long-short carries momentum-crash DD (−64%); winsorisation is a
cleaning judgment; top-100 unreliable (insufficient winsorisation in a small set — beta 3.5), use top-350.

**Verdict: 12-1 momentum is real, significant, cost-surviving, era-consistent, and survivorship-robust — the
programme's genuine standout edge.** The survivorship test meant to undo it instead validated it.

### 12-1 momentum tradeability (survivorship-free, top-350) — genuinely tradeable (`momentum_tradeability.py`)

Stress the tilt net of realistic frictions (liquidity-tiered spread 15/40/80 bps by turnover tercile; monthly
vs quarterly; break-even cost; crash profile):

- **Costs are almost irrelevant.** Turnover only 6.3×/yr; net-of-tiered-cost long-only CAGR 10.58%, Sharpe
  0.72 vs eligible B&H 5.65% / 0.42. **Break-even ~200 bps round-trip** — it still beats B&H at absurd costs
  (realistic is ~15–40 bps). The edge dwarfs the frictions.
- **Monthly beats quarterly** (Sharpe 0.72 vs 0.63; maxDD −47% vs −54%) — the turnover saving from quarterly
  isn't worth the performance loss, because costs weren't the constraint.
- **The real cost of admission is drawdown, not cost.** Long-only maxDD −47% (beta 0.84 → eats full market
  crashes; worst months 2008-09 −20%, COVID-2020 −17% are *market*, not momentum-specific).
- **Market-neutral long-short is the crash-prone one:** net Sharpe 0.58 but maxDD −64%, a −48% single month
  (Apr-2009 momentum crash), −50% through 2008-09, −32% Nov-2020. Real but a brutal tail.

**Verdict:** the long-only 12-1 momentum tilt is a genuinely tradeable edge — net Sharpe ~0.72 vs
buy-and-hold's 0.42, robust to any realistic cost, survivorship-confirmed; its binding constraint is
equity-beta drawdowns (~−47%), not frictions. The market-neutral factor is real but momentum-crash-tailed.

### Vol-targeting the momentum tilt — a clean risk dial + modest Sharpe uplift (`vol_target_momentum.py`)

Scale the long-only tilt's exposure to constant vol (w = target / trailing-12m vol, causal; small scaling cost):

- **No leverage, target = the tilt's own vol (exposure-neutral):** CAGR 10.31% (≈ raw 10.58%), Sharpe 0.76
  (from 0.72), **maxDD −38% (from −47%)** — same return, ~10pp less drawdown, a clean win.
- **Mostly a risk dial:** target 10/12/15% vol → maxDD −29/−32/−37%, CAGR 8.0/9.0/10.2%, Sharpe ~0.73–0.77;
  lever to 20% → CAGR 14.0%, DD −48%. Sharpe is roughly constant across the dial — chiefly re-scaling risk,
  plus a small timing bonus.
- The Sharpe bonus is modest because some drawdowns are V-shaped (2020: de-risk then miss the snap-back); it
  helped more in the sustained 2008-09 crash (cut to −18%).

**Verdict:** vol-targeting is worth applying — real drawdown control (−47% → −38% at no return cost) and a
small Sharpe uplift, plus a clean way to set the risk level. An improvement, not a transformation (consistent
with the programme's earlier "vol-targeting is real but modest").

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

---

## Value / quality overlay on the momentum tilt (point-in-time fundamentals)

Adds cross-sectional **value** and **quality** factors to the 12-1 momentum tilt, on the
survivorship-free top-350 universe, net of tiered costs. Factors are point-in-time (as-of the
statement `filing_date`, or fiscal-end + 120 days when missing — never used before it was reported):

- **quality** = ½·rank(ROE = net income / book equity) + ½·rank(GP/assets) — both ratios are
  currency-neutral, so quality covers ~all names.
- **value** = ½·rank(B/P) + ½·rank(E/P) — needs price and statement in aligned currency, so it is
  restricted to GBX-quoted GBP/GBX reporters (mega-cap USD reporters are excluded → thinner coverage).

Composite = equal-weight of available component z-scores; long the top quintile, monthly.

**Bug found and fixed first (this invalidated an earlier run).** The point-in-time raster builder used
`piv.reindex(months, method="ffill")` on a *DataFrame*, which maps each month to the single most-recent
filing **row** and copies it wholesale — so a month only showed the handful of names that filed on that
one exact date. Factor coverage read a nonsensical 175/124 names and the overlay looked useless. The fix
is a per-**column** forward-fill (`reindex(index.union(months)).ffill().reindex(months)`). Coverage then
rose to 2,175 (quality) / 1,325 (value) of 2,978, and the result reversed. (A second bug — the standalone
"X only" rows re-ran momentum because `run_multi` ignored the swapped-in signal — was fixed too.)

```
eligible-universe EW B&H: CAGR +5.65%  Sharpe 0.42

strategy                    CAGR  Sharpe   maxDD
momentum only            +10.58%    0.72    -47%
mom + quality            +11.38%    0.80    -47%
mom + value              +10.85%    0.71    -53%
mom + value + quality    +12.60%    0.84    -51%
quality only             +10.77%    0.74    -41%
value only               +12.86%    0.74    -61%
```

**Verdict:** value and quality *do* improve the momentum tilt — the opposite of the buggy run's reading.

- **Quality is the natural complement.** `mom + quality` lifts Sharpe 0.72 → 0.80 at the *same*
  drawdown (−47%). Quality standalone matches momentum's return (10.77%) with the *shallowest* drawdown
  of the lot (−41%) — a genuinely diversifying, defensive premium.
- **Value adds return but buys deeper drawdowns.** Value standalone has the highest CAGR (12.86%) but a
  brutal −61% maxDD; `mom + value` barely beats momentum and worsens the drawdown. Classic value.
- **The full three-factor book** (mom + value + quality) has the best risk-adjusted return of all
  (CAGR 12.60%, Sharpe 0.84) but does not cut the drawdown (−51%). If the binding constraint is
  drawdown, `mom + quality` is the better trade-off; if it is Sharpe, the three-factor book wins.

Caveat: value coverage (1,325) is thinner than quality (2,175) because of the currency restriction, and
all books share momentum's structural ~−47% drawdown vulnerability. Vol-targeting (previous section) is
the lever for the drawdown; the factor overlay is the lever for return/Sharpe. They are complementary.

---

## Combining the levers: vol-targeted multi-factor tilt (`momentum_combined.py`)

The factor overlay is the return/Sharpe lever; vol-targeting is the drawdown lever. Do they
stack? Take each book's net monthly tilt series, then scale exposure to a constant vol
(w = target / trailing-12m-vol, causal). Survivorship-free top-350, net tiered costs.

```
eligible-universe EW B&H: CAGR +5.65%  Sharpe 0.42

                                    CAGR    vol  Sharpe  maxDD
momentum only
  raw (no targeting)              +10.58%  15.8%   0.72   -47%
  vol-target own-vol, no lev      +10.31%  14.2%   0.76   -38%
  vol-target own-vol, lev<=1.5    +11.95%  17.6%   0.73   -42%
mom + quality
  raw (no targeting)              +11.38%  15.0%   0.80   -47%
  vol-target own-vol, no lev      +10.88%  13.6%   0.83   -39%
  vol-target own-vol, lev<=1.5    +12.93%  16.6%   0.82   -42%
mom + value + quality
  raw (no targeting)              +12.60%  15.7%   0.84   -51%
  vol-target own-vol, no lev      +11.28%  14.2%   0.83   -45%
  vol-target own-vol, lev<=1.5    +13.30%  17.3%   0.81   -49%
```

**The two levers stack cleanly — they act on different axes and don't fight.**

- **Best risk-controlled build: `mom + quality`, vol-targeted, no leverage → CAGR 10.88%,
  Sharpe 0.83, maxDD −39%.** Versus plain momentum (10.58% / 0.72 / −47%): more return,
  a big Sharpe jump (0.72 → 0.83), and the drawdown cut by ~8pp — all at once, no leverage.
- **Best return build: `mom + value + quality`, own-vol with lev≤1.5 → CAGR 13.30%,
  Sharpe 0.81, maxDD −49%.** Leverage buys ~2.7pp of CAGR but hands most of the drawdown
  back; the drawdown stays wherever the underlying book sits (value's deep-DD nature shows).
- **Quality is the sweet spot for the combination.** It enters with the shallowest raw
  drawdown, so vol-targeting starts from a better base: `mom + quality` no-lev reaches
  Sharpe 0.83 at only 13.6% vol and −39% DD — the best all-round point on the frontier.

Frontier summary: pick `mom + quality` no-leverage if drawdown is the binding constraint
(0.83 Sharpe, −39% DD at ~13.6% vol); step up to leverage / add value only to chase raw
CAGR, accepting the drawdown creeps back toward −45/−49%. Every combined build still beats
the eligible universe B&H (5.65% / 0.42) decisively. This is the programme's best build:
survivorship-free, net of costs, point-in-time factors, causal vol-targeting.

---

## Calendar-phase robustness: is the edge an artefact of rebalancing on month-ends? (`momentum_phase_robustness.py`)

The whole strategy resamples to month-end. If the edge only exists because rebalances land
on month-ends, that would be a red flag. Test: shift the entire monthly cycle by 0–27 days
(shift the daily index, then `resample("ME").last()` — which nulls empty bins so delisted
names drop out; `offset=` is silently ignored for month-end frequency, and reindex-ffill
would wrongly resurrect delisted names as flat-price zombies — two harness bugs found and
fixed before trusting any number). shift=0 reproduces the month-end baseline exactly.

```
                         CAGR range      mean edge/B&H   Sharpe range   beats B&H
long-only momentum    +10.3%..+13.6%       +4.58pp        0.50..0.72      10/10
mom + quality         +10.4%..+14.5%       +5.40pp        0.53..0.80      10/10
```

**Verdict: robust.** Both builds beat the eligible-universe B&H in every one of the 10
calendar phases. Momentum's CAGR is tightly clustered at 10.3–10.5% in nine of ten phases
(std ±0.95pp); the edge over B&H is +4.6pp (momentum) / +5.4pp (mom+quality) and never
drops below +3.3pp. The result is not a month-end artefact — you would have beaten the
market starting the cycle on any day of the month.

Wrinkle: at shift=12 (mid-month sampling) both strategy AND benchmark CAGR jump ~3pp
together (13.6% vs 10.2%) and Sharpe dips to 0.50 — a data artefact (a mid-month sampling
date catching bad ticks that survive the per-month 1/99 winsorisation), inflating the
*level* not the *edge*: the edge there is +3.3pp, the smallest of all phases, so even the
anomalous phase clears the bar. It is the conservative case, not a failure. (Robustness is
inherited by the vol-targeted builds, which are a monotone re-scaling of these same series.)

---

## Is 69 names optimal? Holding-count sweep (`momentum_breadth_sweep.py`)

The default holds ~69 names -- not a chosen number, but `floor(0.20 x ~347 eligible)`, the
conventional top quintile. Sweeping the holding count directly (top-M of the mom+quality
composite, long-only, net tiered costs):

```
 #held  ~pctile     CAGR  Sharpe   maxDD  turn/yr
    20       6%  +12.86%    0.75    -55%     6.8x
    30       9%  +12.15%    0.76    -54%     6.0x
    50      14%  +11.37%    0.77    -51%     5.0x
    69      20%  +11.40%    0.80    -47%     4.3x
   100      29%  +10.84%    0.77    -45%     3.5x
   150      43%   +9.81%    0.71    -45%     2.7x
   200      58%   +8.91%    0.66    -47%     2.1x
```

**Verdict: 69 is emergent, not chosen -- but well-placed.** Two clean monotonic effects:
concentrating RAISES CAGR (20 names 12.9% -> 200 names 8.9%) but DEEPENS drawdown (-55% ->
~-45%) and RAISES turnover (6.8x -> 2.1x). Sharpe is single-peaked at the quintile (~69, 0.80)
and forms a plateau (0.77-0.80 across 50-100 names), so it is not a fragile in-sample optimum.

The count is a deliberate return-for-drawdown lever: concentrate to 20-30 names for ~12-13%
CAGR (same as the value+leverage route, fewer moving parts) if you can stomach ~-55% drawdown;
stay at the quintile for max Sharpe. A smaller book is also easier to hold accurately at small
NAV with whole shares (see spec 7.4), making a 20-30 name build the natural small-account
configuration. Do not fine-tune to a single number -- the plateau is the robust choice.

---

## Breadth × factor: does value earn its place when concentrated? (`momentum_concentration_factors.py`)

Following the breadth sweep: at ~69 names, adding value gives the best Sharpe of all
(mom+value+quality 0.84). Does that survive concentration? Each factor book at 20/30/69
names, long-only, net tiered costs:

```
book                 #held     CAGR  Sharpe   maxDD  turn/yr
mom+quality             20  +12.86%    0.75    -55%     6.8x
mom+quality             30  +12.15%    0.76    -54%     6.0x
mom+quality             69  +11.40%    0.80    -47%     4.3x
mom+value               20  +13.39%    0.73    -61%     7.3x
mom+value               30  +13.15%    0.75    -62%     6.7x
mom+value               69  +10.90%    0.71    -53%     4.9x
mom+value+quality       20  +12.48%    0.72    -59%     7.0x
mom+value+quality       30  +12.18%    0.73    -59%     6.4x
mom+value+quality       69  +12.66%    0.84    -51%     4.8x

vol-target on 20-name mom+value+quality (own-vol 19.0%):
  no leverage  CAGR +12.16%  Sharpe 0.76  maxDD -50%  avg exp 0.94
  lev<=1.5     CAGR +15.18%  Sharpe 0.77  maxDD -53%  avg exp 1.20
```

**Verdict: value needs BREADTH; quality suits CONCENTRATION.** Value gives the best Sharpe
(0.84) only in the wide ~69-name book. Concentrate and its slow, deep-drawdown nature
dominates faster than its return: at 20 names mom+value+quality (12.5%/0.72/-59%) is worse
than mom+quality (12.9%/0.75/-55%) on every axis, and mom+value has the highest raw CAGR
(13.4%) but the deepest drawdown anywhere (-61%) with no Sharpe gain. So use mom+quality for
a concentrated (20-30 name) build and reserve value for the ~69-name book. The maximum-
aggression corner -- 20-name mom+value+quality, vol-targeted lev<=1.5 -- reaches the
programme's highest CAGR (15.2%) at Sharpe 0.77 / maxDD -53%: a raw-return extreme, not a
well-run default. Ends of the risk dial: Sharpe-best = mom+value+quality @69 (0.84);
CAGR-best = 20-name three-factor levered (15.2%).

---

## UK stamp duty: the largest single friction for a share account (`momentum_stamp_duty.py`)

The headline cost model charged only the tiered bid/offer spread (15/40/80 bps) and OMITTED
UK Stamp Duty Reserve Tax (0.5% on purchases, buys only). Re-run with SDRT added:

```
                                   net spread   net spread+SDRT   drag
momentum only  (~69)               10.6% / 0.72   9.0% / 0.63     -1.6pp
mom + quality  (~69)               11.4% / 0.80  10.2% / 0.73     -1.2pp
mom+quality vol-target (no lev)    10.9%/0.83/-39%  9.8%/0.76/-40% -1.1pp
mom+quality vol-target (lev<=1.5)  12.9% / 0.82  11.6% / 0.75     -1.3pp
mom + quality  (20 names)          12.9%         11.0%            -1.9pp
```

**Verdict: stamp duty is a ~1.2-1.6pp/yr drag at the quintile (~1.9pp concentrated) -- LARGER
than the spread** (top tier only ~15 bps vs SDRT 50 bps on buys; ~2.1x of NAV bought/yr for
mom+quality). This walks back the earlier "costs are almost irrelevant" -- true of spread, not
of the full picture. BUT the edge still clears B&H comfortably (mom+quality 10.2% net-of-
everything vs market 5.65%, Sharpe 0.73), so it revises the numbers ~1.2pp lower without
breaking the strategy. The reported figures were mildly optimistic for a taxable share account.

Instrument/tax implication (see MOMENTUM_STRATEGY_SPEC.md section 16): SDRT is avoided only by
CFDs/spread-bets (NOT by ISA/SIPP, which still pay it), but CFDs pay ~2.5-3pp financing on a
month+ long hold, so an unlevered CFD book is net worse than shares; CFDs win only for the
levered variant. Best tax route for a UK investor is usually owning the shares in an ISA/SIPP
(no CGT/dividend tax). CFDs give NO incremental loss-offset over shares (share losses already
offset gains); spread bets are tax-free on gains but losses aren't deductible.
