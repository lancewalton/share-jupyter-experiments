# Momentum Strategy — Implementation Specification

**Strategy:** quality- and low-volatility-tilted 12-1 cross-sectional momentum, long-only,
volatility-targeted, on survivorship-free UK equities.
**Status:** research-validated (see `momentum-strategy/RESULTS.md` and `momentum.html`);
this document specifies how to build it as an automated system.
**This is not investment advice.** It is a specification of a backtested strategy, with its
known limitations stated explicitly. Deploy real capital only after independent validation and
appropriate risk/compliance review.

---

## 0. Recommended build (the default this spec describes)

| | Value |
|---|---|
| **Signal** | Equal-weight composite of 12-1 momentum + quality (ROE + gross-profitability) + low-volatility |
| **Universe** | Survivorship-free LSE common stock (GBP/GBX), top 350 by trailing liquidity |
| **Selection** | Long the top quintile of the composite (~69 names) |
| **Weighting** | Equal weight, then scaled by a volatility target |
| **Vol target** | 15% annualised, exposure cap 1.0 (unlevered) |
| **Rebalance** | Monthly |
| **Backtest result** | CAGR ≈ 11.2%, Sharpe ≈ 0.89, max drawdown ≈ −40% (net of tiered spread); **≈ 10.1% / 0.81 / −41% after UK stamp duty** on a taxable share account (§8) |

**Optional leverage variant:** raise the exposure cap to 1.5. Backtest: CAGR ≈ 13.8%, Sharpe ≈
0.90, max drawdown ≈ −43% net of spread (**≈ 12.5% / 0.83 / −44% after stamp duty**) — roughly
+2–2.5pp/yr of return for +3pp of drawdown, the most efficient point on the frontier. Everything
else is identical. Do not add the *value* factor: it needs breadth to help (§5.5), has a much
deeper standalone drawdown (−61%), and is currency-restricted; low-volatility reaches the same or
better Sharpe while being currency-neutral and fully covered.

The rest of this document specifies the **default (unlevered)** build; the leverage variant is a
one-parameter change (§6.3).

---

## 1. Scope and assumptions

### 1.0 Two contexts: the live loop vs. validation (read this first)
This document specifies a **live automated trading system**. That system stands in the present: it
can only ever see and trade **currently listed** names, and it inherently **cannot see the future**.
Two consequences follow, and they are the source of most potential confusion:

- **"Point-in-time" is automatic live.** The live universe on any date is just the names listed and
  tradeable *that day*. There is nothing to reconstruct and no survivorship bias to avoid — a
  company that has delisted simply isn't in the market any more.
- **"No lookahead" is automatic live.** You physically don't have tomorrow's prices. The elaborate
  anti-lookahead discipline exists to stop the *historical simulation* from cheating, not the live
  system.

So a distinct set of requirements — **survivorship-free history, point-in-time reconstruction,
enforced no-lookahead** — belongs to **validation**: building the backtest that proved this works,
and periodically re-checking it on fresh data. Those are collected in **§15** and tagged
**[VALIDATION]** where they appear. Everything else is the **live loop**. Two genuine live remnants
of the lookahead question survive into §5/§7 and are called out there: the momentum **skip-month**
(a signal feature, not a data rule) and **point-in-time fundamentals** (filing-date gating, which
matters live because vendor data is often keyed to fiscal-period-end).

- **Market:** London Stock Exchange, GBP- and GBX-quoted common stock. The edge was validated on
  UK equities only; do not assume it transfers unchanged to other markets (US equities show the
  same effect; extend only after re-validation).
- **Instrument type:** cash equities, long-only. No derivatives, no shorting. (A market-neutral
  long/short version exists but carries momentum-crash tails — out of scope here.)
- **Data vendor:** The *live* system needs only a feed of the **current** LSE universe (EOD OHLCV,
  corporate actions, filed fundamentals) — see §3. The *validation* dataset additionally needs
  **survivorship-free** history including delisted names ([VALIDATION], §15); EODHD "All-In-One"
  supplied this for the research. §3 lists requirements vendor-agnostically.
- **Rebalance cadence:** monthly. The edge is **robust to the day of the month** you rebalance on
  (validated across 10 calendar phases — see `momentum_phase_robustness.py`), so the exact day is
  an operational choice, not a source of edge.
- **Capital:** the strategy holds ~69 names across three liquidity tiers; capacity is bounded by
  the mid/low tiers (§8). Size accordingly.
- **Broker requirement — fractional shares / micro-lots.** The book is ~69 near-equal positions
  (~1.4% of NAV each) further scaled by a fractional vol-target exposure, so precise sizing needs
  sub-share granularity. Whole-share dealing is only adequate at large NAV (§7.4); at small/retail
  NAV it makes accurate equal-weighting impossible. **The execution venue must support fractional
  UK-equity dealing (or the NAV must be large enough that whole-share rounding is immaterial), and
  expose it via API.** This is a hard selection criterion for the broker.

---

## 2. Definitions and notation

- `t` — a rebalance date (a **formation** date). Signals are computed using data known **on or
  before** `t`. `t+1` — the next rebalance date (the **holding** period is `t → t+1`).
- **Adjusted close** — total-return-adjusted close (dividends reinvested, splits applied). Used
  for **all signal and return calculations**.
- **Raw close** — the actual traded price. Used for **order sizing and execution only**.
- **12-1 momentum** — the total return from 12 months before `t` to 1 month before `t` (i.e. the
  trailing 12-month return **skipping the most recent month**).
- **Cross-sectional** — computed across names at a single date (e.g. a z-score over the eligible
  universe on date `t`).
- **Point-in-time (PIT)** — a datum tagged with the date it became publicly known, never the date
  it refers to. Fundamentals are keyed by **filing date**, not fiscal-period end.

---

## 3. Data requirements

**Live feed:** a daily feed covering the **current** LSE universe — the names listed and tradeable
now — is all the live loop consumes. It needs enough *trailing history per current name* to compute
the signals (≈13 months of prices for momentum; the latest filed fundamentals), which is ordinary
back-history for a live name, not a survivorship-free archive.

**[VALIDATION]** The dataset used to *build and re-check* the backtest is different: it must be
**survivorship-free** — including companies that later delisted, merged, or went bankrupt, with data
to their last trading day — or it will overstate returns. That requirement lives in §15; it does
**not** constrain the live universe (a delisted name is simply gone, §4).

### 3.1 Price / volume (daily, per name)
| Field | Use |
|---|---|
| date | index |
| raw close | order sizing, execution |
| adjusted close (total return) | momentum signal, portfolio returns |
| volume | liquidity ranking |
| currency (GBP or GBX) | turnover normalisation, share sizing |

Turnover (for liquidity ranking) = `raw_close × volume × fx`, where `fx = 0.01` for GBX-quoted
names (pence → pounds) and `1.0` for GBP. This puts all names on a common (GBP) liquidity scale.

### 3.2 Corporate actions
Splits and dividends, per name, dated. Required to (a) maintain the adjusted-close series and
(b) reconcile live share positions through corporate actions. If the vendor supplies a maintained
adjusted-close series, that covers signals/returns; you still need raw actions for live position
keeping.

### 3.3 Fundamentals (point-in-time, per name, per annual statement)
| Field | Source line | Use |
|---|---|---|
| filing date | statement metadata | **availability date** (no lookahead) |
| total stockholder equity | balance sheet | book equity → ROE |
| total assets | balance sheet | gross-profitability denominator |
| net income | income statement | ROE numerator |
| gross profit | income statement | gross-profitability numerator |

**Availability date rule (matters live, not only in backtest):** use `filing_date` if present, else
`fiscal_period_end + 120 days` as a conservative proxy. A fundamental value must **never** enter a
signal before its availability date. This is a genuine *live* requirement, not merely a backtest
nicety: vendors commonly key a figure to its **fiscal-period end**, which is weeks or months before
it was actually published — act on it then and you are trading on information the market did not yet
have. Also store fundamentals **as first published** and beware later **restatements** silently
overwriting the original (that reintroduces lookahead even in a live system that re-pulls history).
(The research parser `parse_fundamentals.py` implements the filing-date rule.)

**Currency note:** the recommended build's quality factor is built entirely from **ratios**
(ROE = net income / book equity; GP/A = gross profit / total assets), which are **currency-neutral**.
No FX conversion is required. (This is a deliberate reason to prefer quality over value for the
default build: value needs price and accounts in the same currency and is therefore restricted to
a subset of names.)

---

## 4. Universe construction (per rebalance date `t`)

1. **Base universe:** every LSE common stock (GBP/GBX) that is **currently listed and has a valid
   recent price** on `t`. Live this is simply the tradeable market as it stands — a name that has
   delisted or is suspended has no price and is naturally absent; there is no list of "dead names"
   to include or exclude. ([VALIDATION] the backtest reconstructs this same set historically from
   survivorship-free data, and must **not** carry a delisted name's last price forward — doing so
   resurrects dead companies as flat-price zombies, a bug caught during research; see
   `momentum_phase_robustness.py`. Live, this cannot happen, because a delisted name has no live
   quote.)
2. **Liquidity ranking:** for each name compute trailing **12-month mean daily turnover** (mean of
   daily GBP turnover over the trailing ~252 trading days, requiring at least ~60 valid days).
3. **Eligible set:** the **top `N = 350`** names by that liquidity measure. This is the tradeable
   universe for date `t`. All subsequent cross-sectional calculations are over this set.
4. **Minimum breadth:** require at least **30** eligible names with a valid signal; otherwise skip
   the rebalance (relevant only to a cold start on a thin market).

---

## 5. Signal construction (per rebalance date `t`, over the eligible set)

### 5.1 Momentum factor
For each eligible name:
```
mom_raw = adj_close[t − 1 month] / adj_close[t − 12 months] − 1
```
Then standardise cross-sectionally: `z_mom = zscore(mom_raw)` over the eligible set
(`zscore(x) = (x − mean) / std`).

**Why skip the most recent month (the "1" in 12-1).** This is a **signal feature, not a
data-availability rule** — it is retained unchanged live. The most recent month exhibits short-term
**reversal** (last month's biggest movers tend to snap back), which is noise for a 12-month
momentum signal, so the standard construction excludes it. A useful side effect: the signal depends
only on prices **through one month ago**, so it is comfortably causal and needs no just-executed
data — the live system computes it with no timing pressure, which is also why the strategy is so
insensitive to *which* day of the month you rebalance (§5.5 / phase robustness). You still need a
**current** price on `t`, but only for order sizing and execution (§7), never for the signal.

### 5.2 Quality factor
Using the latest fundamentals **available on or before `t`** (by availability date, §3.3):
```
ROE = net_income / book_equity        (only where book_equity > 0)
GPA = gross_profit / total_assets     (only where total_assets > 0)
quality_pct = 0.5 · rank_pct(ROE) + 0.5 · rank_pct(GPA)   (percentile ranks over eligible set)
z_qual = zscore(quality_pct)
```
Names with no available fundamentals have `z_qual = NaN` (they contribute only the other factor
scores — see 5.3).

### 5.2b Low-volatility factor
Prefer calmer names — the low-volatility premium, and a defensive complement to momentum. Uses only
return history (no fundamentals, currency-neutral, effectively full coverage):
```
vol = trailing 12-month standard deviation of the name's monthly returns, lagged 1 month (causal)
z_lowvol = zscore(-vol)                                   # higher score = lower volatility
```
This is the volatility **level**, not its change — we tested vol *change* (as both a factor and a
gate, signed and absolute) and it consistently *hurt* (`momentum_vol_change.py`), so it is not used.

### 5.3 Composite
```
composite = mean( [z_mom, z_qual, z_lowvol] , skipping NaNs )
```
**Momentum is mandatory:** drop any name without a valid `z_mom`. Quality and low-vol are additive
where present (both are; low-vol needs only price history). A name with momentum but no quality is
ranked on the factors it has — intentional, and matches the validated build. Adding low-vol lifts
the tilt's Sharpe 0.80 → 0.85 and, once vol-targeted, 0.83 → 0.89 (§8), at ~no drawdown cost.

### 5.4 Selection
Rank the eligible names that have a valid composite (i.e. a valid `z_mom`). Let `M` be their count.
Select the **top quintile**:
```
k = max(1, floor(0.2 × M))          # ≈ 69 names for M ≈ 347
targets = the k names with the highest composite
```

### 5.5 Selection breadth is a deliberate lever
The ~69 names is **not a tuned number** — it is `floor(0.2 × M)`, the conventional top quintile.
Sweeping the holding count (`momentum_breadth_sweep.py`, net tiered costs) shows a smooth
concentration/diversification trade-off, and the quintile sits on the **Sharpe peak** (a plateau,
not a knife-edge):

| # held | CAGR | Sharpe | maxDD | turn/yr |
|---:|---:|---:|---:|---:|
| 20  | +12.9% | 0.75 | −55% | 6.8× |
| 30  | +12.2% | 0.76 | −54% | 6.0× |
| 50  | +11.4% | 0.77 | −51% | 5.0× |
| **69** | +11.4% | **0.80** | −47% | 4.3× |
| 100 | +10.8% | 0.77 | −45% | 3.5× |
| 150 | +9.8%  | 0.71 | −45% | 2.7× |
| 200 | +8.9%  | 0.66 | −47% | 2.1× |

- **Concentrating raises CAGR but deepens drawdown and turnover.** 20–30 names reaches ~12–13%
  CAGR (comparable to the value+leverage route, with fewer moving parts) at the cost of a ~−55%
  drawdown and higher turnover/capacity strain.
- **The quintile (~69) maximises Sharpe** (~0.80). Choose ~69 for the smoothest risk-adjusted
  ride; concentrate to 20–30 only with the drawdown tolerance to match.
- **Interaction with granularity (§7.4):** a smaller book is *easier* to hold accurately at small
  NAV with whole shares — so at low capital, a 20–30 name concentrated build is both more feasible
  and higher-returning, trading the deeper drawdown for it. This is the natural small-account
  configuration if fractional dealing is unavailable.

Do not fine-tune the count to a single in-sample optimum; the plateau across 50–100 names is the
robust choice, with concentration a deliberate return-for-drawdown trade rather than an
optimisation.

**Breadth × factor interaction — which factor to pair with which breadth**
(`momentum_concentration_factors.py`). The best *third* ingredient depends on how concentrated the
book is:

| book | 20 names | 69 names |
|---|---|---|
| mom + quality | +12.9% / 0.75 / −55% | +11.4% / 0.80 / −47% |
| mom + value | +13.4% / 0.73 / −61% | +10.9% / 0.71 / −53% |
| mom + value + quality | +12.5% / 0.72 / −59% | +12.7% / **0.84** / −51% |

*(CAGR / Sharpe / maxDD, net tiered costs.)*

- **Value earns its keep only when diversified.** Adding value gives the programme's best Sharpe
  (0.84) at ~69 names, but under concentration its slow, deep-drawdown nature dominates faster than
  its return: at 20 names `mom+value+quality` (12.5% / 0.72 / −59%) is **worse on every axis** than
  `mom+quality` (12.9% / 0.75 / −55%). `mom+value` at 20 has the highest raw CAGR (13.4%) but the
  deepest drawdown anywhere (−61%) and no Sharpe gain.
- **Rule of thumb: quality for a concentrated book, value (added) for a wide one.** For a
  concentrated (20–30 name) build, use `mom + quality`; reserve value for the ~69-name book.
- **Maximum-aggression corner:** 20-name `mom+value+quality`, vol-targeted at lev ≤1.5, reaches the
  programme's highest CAGR (**15.2%**) at Sharpe 0.77 / maxDD −53% — a raw-return extreme, not a
  well-run default.
- **The chosen third factor is low-vol, not value** (§5.2b). Low-vol matches or beats value's Sharpe
  as an addition to mom+quality while being currency-neutral (full coverage), shallower on drawdown,
  and it compounds with the portfolio vol-target — so it, not value, is in the recommended build.
  Value stays an *optional* wide-book return-chaser with the caveats above.

---

## 6. Portfolio construction

### 6.1 Base weights
Equal weight across the selected names: `w_base[i] = 1 / k`.

### 6.2 Volatility target (exposure scaling)
Estimate the **strategy's own** trailing realised volatility, causally:
```
rv_t = std( strategy_monthly_return[t−12 … t−1] ) × sqrt(12)      # annualised, uses months up to t−1 only
exposure_t = min( TARGET_VOL / rv_t , CAP )                        # TARGET_VOL = 0.15, CAP = 1.0
```
Final weights: `w[i] = exposure_t × w_base[i]`. The remainder `(1 − exposure_t)` is held in cash.
`exposure_t` is a single portfolio-level scalar (not per-name). Average exposure runs ≈ 0.94, i.e.
the target mostly *de-risks* in turbulent periods rather than levering up.

**Cold start:** vol-targeting needs 12 months of realised strategy returns. For the first 12
months run **unscaled** (`exposure = 1.0`, i.e. `CAP`), or seed `rv` from a paper/backtested vol
estimate. Never use future returns to estimate `rv` (the `t−1` cutoff enforces this).

### 6.3 Leverage variant
Set `CAP = 1.5` to allow up to 1.5× gross exposure (borrowing when `rv_t < TARGET_VOL / 1.5`).
This is the only change for the higher-return variant (§0). Ensure margin, borrow cost, and risk
limits support it before enabling.

---

## 7. Rebalancing and execution

### 7.1 Schedule
- **Frequency:** monthly.
- **Day:** any fixed, operationally convenient day (e.g. first business day of the month, or a
  fixed month-end). The edge is calendar-phase robust, so pick one and keep it stable. Prefer a day
  on which a clean, validated closing cross-section is available.
- **Execution sequencing (the one live "lookahead" concern):** a live system cannot see the future,
  so there is no future data to guard against — but you must not assume you can *trade at the very
  price you computed the signal from*. Safe pattern: compute target weights from data through the
  close of day `D`, then execute at or after the **open of day `D+1`**. (The momentum signal already
  uses only prices through a month ago, §5.1, so it is never the binding constraint here; this rule
  is really about the *current* price you use for sizing versus the price you actually get filled at.
  Point-in-time fundamentals gating is covered in §3.3.)

### 7.2 Order generation
1. Compute target weights `w` (§6).
2. Convert to target notionals: `notional[i] = w[i] × NAV`.
3. Convert to target shares using the **raw** close and currency
   (`shares = notional / (raw_close × fx)`), rounded to the venue's minimum increment — a
   **fractional quantity** where supported (see §7.4), otherwise a whole share.
4. Diff against current holdings → buy/sell orders. Names dropped from the target get fully
   liquidated; new names get established.
5. Execute to minimise impact: prefer limit / VWAP / participation orders over market orders,
   especially in the mid- and low-liquidity tiers. Turnover is low (§8), so there is no need to
   rush fills within the rebalance day. (Some venues route fractional quantities as market-on-
   aggregate orders — check that fractional dealing still permits your intended order type.)

### 7.3 Held-name corporate actions / delisting between rebalances
- **Splits/dividends:** adjust share counts / cash; the adjusted-close series handles signal
  continuity.
- **Delisting or suspension of a held name:** exit via the corporate-action mechanism (cash
  received, or write-off at last traded price). Do not assume you can sell at a model price. This is
  a real, expected event when trading a broad equity universe live, and must be handled
  operationally. (It is also the live counterpart of the backtest's survivorship handling: a name
  leaves your book by dying, exactly as it leaves the historical panel.)

### 7.4 Share granularity and minimum NAV
Each position targets `p ≈ exposure × (1/k) × NAV ≈ 0.0136 × NAV` (for `exposure ≈ 0.94`, `k ≈ 69`).
With **whole-share** dealing, rounding a name costs up to half a share, so the relative sizing
error per name is up to `(0.5 × raw_price) / p ≈ 37 × raw_price / NAV`.

- To keep that per-name error under ~10% of the target position for a typical LSE share price,
  NAV must be on the order of **£100k+**; for higher-priced names it is larger still, and a name
  whose share price exceeds `p` cannot be held at target weight **at all** — it forces either an
  overweight (≥1 share) or a skip.
- **Fractional-share dealing removes this floor entirely** — you size each name to `p` directly and
  the equal-weighting is exact regardless of NAV. This is why fractional support is a hard broker
  requirement (§1) for anything but a large book.
- **Rounding policy (whichever granularity applies):** round to the venue increment, and apply a
  **no-trade band** so a name is only re-traded when its weight drifts beyond a tolerance (e.g.
  ±25% of the equal weight). This suppresses churn from rounding noise and keeps realised turnover
  near the modelled ~4–6×/yr. Sweep residual rounding cash into the next rebalance rather than
  forcing tiny corrective trades.

---

## 8. Cost, turnover, and capacity

- **Turnover:** ≈ **4–6× one-way per year** (≈ 4.3× for the mom+quality build). Low, because
  monthly 12-1 momentum is slow-moving.
- **Cost model (spread):** liquidity-tiered round-trip spreads applied to traded notional —
  **15 bps** (top liquidity tercile), **40 bps** (mid), **80 bps** (low). Cost per rebalance
  ≈ `Σ |Δweight_i| × spread_i / 2`.
- **UK stamp duty (SDRT) — the largest single friction for a share account.** A cash-share
  purchase pays **0.5% SDRT on buys** (not sells). At the quintile this is a **~1.2–1.6 pp/yr**
  drag (≈2.1× of NAV bought per year for mom+quality → 0.5%×2.1 ≈ 1.05pp, a bit more for
  momentum-only); it rises to **~1.9 pp/yr** for a concentrated 20-name book. It is *larger than
  the spread* (top tier only ~15 bps). Net figures **including** SDRT (`momentum_stamp_duty.py`):

  | build (~69 names) | net spread only | net spread + SDRT |
  |---|---|---|
  | momentum only | 10.6% / 0.72 | **9.0% / 0.63** |
  | mom + quality | 11.4% / 0.80 | **10.2% / 0.73** |
  | mom + quality + low-vol (the tilt) | 11.6% / 0.85 | **10.5% / 0.77** |
  | **+ vol-target (no lev) — recommended** | 11.2% / 0.89 / −40% | **10.1% / 0.81 / −41%** |
  | + vol-target (lev ≤1.5) | 13.8% / 0.90 / −43% | **12.5% / 0.83 / −44%** |

  SDRT is avoided only by CFDs / spread bets (which carry other costs — §16), **not** by an ISA or
  SIPP (both still pay it on UK share purchases).
- **Break-even:** even net of spread **and** SDRT, the edge clears buy-and-hold by a wide margin
  (mom+quality 10.2% vs market 5.65%). The long-only tilt survives round-trip *spread* up to
  ≈ **200 bps** before losing to B&H, so realistic frictions (spread + SDRT ≈ 65 bps on the buy
  side of the liquid tier) leave ample headroom. Cost is real but **not the binding constraint —
  drawdown is** (§9).
- **Capacity:** ~69 equally weighted names, but the low-liquidity tier caps size. Constrain each
  order to a small fraction of the name's trailing average daily volume (e.g. ≤ 5–10% of 20-day
  ADV) and spread execution over the day (or multiple days) if a target exceeds that. At scale,
  consider capping individual position size relative to ADV, which slightly reduces the effective
  quintile but protects fills.

---

## 9. Risk management and expected behaviour

- **This is a high-beta long equity tilt** (market β ≈ 0.84), **not** market-neutral. Roughly
  two-thirds of the return is equity-market beta; the value-add over that is ≈ +6%/yr.
- **Expect deep drawdowns.** Backtested max drawdown is ≈ **−40%** (unlevered) / **−43%** (levered
  1.5×). The worst months are **market crashes** (e.g. 2008, the 2020 COVID crash), not
  idiosyncratic blow-ups. Vol-targeting *reduces* but does **not** remove this.
- **Position/exposure limits (recommended):** per-name weight cap (e.g. 2× the equal weight,
  ~3%) to contain single-name error; gross-exposure cap = `CAP` (§6.2); a kill-switch / review
  trigger if realised drawdown exceeds the backtested max by a set margin.
- **Regime awareness:** the edge is real but time-varying (strong pre-2013, flat/negative through
  the 2018–2023 whipsaws, recovering 2024–26). Do not size to the best historical sub-period.

---

## 10. Backtest → live gaps (read before deploying)

These are places where the research code takes a shortcut that is valid for a historical study but
must be replaced by real machinery live.

1. **Winsorisation is a data-cleaning proxy, not a trading rule.** The backtest winsorises monthly
   cross-sectional returns at the 1st/99th percentile to stop bad ticks in the historical file from
   corrupting results. Live, you cannot "winsorise" a return you actually earned. Replace it with:
   (a) **real-time price validation** on the data feed (reject/flag implausible ticks before they
   reach the signal), and (b) staying in the **liquid top-350**, where bad prints are rarer. Also
   validate the *history* feeding the momentum signal — a bad print 12 months ago distorts the
   ranking.
2. **No lookahead — enforce it mechanically.** Fundamentals via filing date (§3.3); signals from
   data strictly before execution (§7.1). Any accidental use of same-day or future data will
   manufacture phantom edge.
3. **Delisting must produce a real exit, not a modelled one** (§7.3).
4. **FX / quotation:** GBX vs GBP handling for turnover and share sizing must be exact; a factor-100
   error silently corrupts liquidity ranking and position sizes.
5. **Vol-target cold start** (§6.2): the first 12 live months have no realised-vol history.
6. **The mid-month data wrinkle:** in the phase study, one mid-month sampling phase inflated the
   *level* (not the edge) via bad ticks surviving winsorisation. Live, this is subsumed by proper
   tick validation (point 1) — another reason to validate, not winsorise.
7. **Slippage/impact beyond the spread model:** the tiered-spread model is for expectation-setting.
   Measure realised slippage live and compare to the 200 bps break-even headroom.

---

## 11. Parameters (single source of truth)

| Parameter | Symbol | Value | Notes |
|---|---|---|---|
| Universe size | `N` | 350 | top-N by trailing liquidity |
| Liquidity window | — | ~252 trading days (12m), ≥60 valid | mean daily GBP turnover |
| Momentum lookback | — | 12 months | total return … |
| Momentum skip | — | 1 month | … skipping the most recent month |
| Selection quantile | `FRAC` | 0.20 | top quintile long-only (~69 names); Sharpe-optimal, but a deliberate lever — §5.5 |
| Min breadth | — | 30 | else skip rebalance |
| Quality blend | — | 0.5·ROE + 0.5·GP/A (percentile) | currency-neutral ratios |
| Low-vol factor | — | z(−trailing 12m return vol) | currency-neutral; needs only returns |
| Factor combine | — | equal-weight z-scores of momentum + quality + low-vol; momentum mandatory | |
| Weighting | — | equal weight | before vol scaling |
| Vol target | `TARGET_VOL` | 0.15 annualised | ≈ the book's natural vol |
| Vol window | — | 12 months, lagged 1 month | causal |
| Exposure cap | `CAP` | 1.0 (default) / 1.5 (levered) | |
| Rebalance | `K` | monthly | any fixed day |
| No-trade band | — | ±25% of equal weight | suppress rounding/drift churn (§7.4) |
| Cost tiers | — | 15 / 40 / 80 bps round-trip | top / mid / low liquidity tercile |

---

## 12. End-to-end algorithm (pseudocode)

```
# Nightly / pre-rebalance data pipeline (LIVE: current universe only)
maintain daily panel for currently-listed names: raw_close, adj_close, volume, currency (validate ticks)
maintain corporate-actions ledger
maintain fundamentals keyed by filing date (use only figures already published)

# On each rebalance date t:
elig      = top 350 currently-listed names by trailing-12m mean daily GBP turnover (valid price at t)
if count(elig) < 30: skip

mom_raw   = adj_close[t-1m] / adj_close[t-12m] - 1                 # over elig
z_mom     = zscore(mom_raw)
ROE, GPA  = latest PIT fundamentals available <= t                # over elig
z_qual    = zscore( 0.5*rank_pct(ROE) + 0.5*rank_pct(GPA) )
z_lowvol  = zscore( -std(monthly_returns[t-12 .. t-1]) )          # low-vol factor, over elig
composite = nanmean([z_mom, z_qual, z_lowvol]); drop names with NaN z_mom

k         = max(1, floor(0.20 * count(valid composite)))
targets   = top-k names by composite
w_base    = 1/k each

rv        = std(strategy_monthly_return[t-12 .. t-1]) * sqrt(12)  # cold start: exposure=CAP
exposure  = min(0.15 / rv, CAP)                                   # CAP = 1.0 (or 1.5 levered)
w         = exposure * w_base                                     # remainder in cash

orders    = diff(target_shares(w, NAV, raw_close, currency), current_holdings)
execute(orders)  with limit/VWAP, respecting ADV caps            # T+1 open after a T-close signal

# Between rebalances: apply corporate actions; force-exit delisted/suspended held names.
```

---

## 13. Pre-deployment validation checklist

- [ ] Reproduce the headline backtest from raw data end-to-end (no manual steps).
- [ ] Confirm **zero lookahead**: shift every input back one period and confirm the edge degrades
      as expected (a sanity check that the live edge isn't timing-of-data artefact).
- [ ] Re-run the **calendar-phase** test on your own data pipeline (`momentum_phase_robustness.py`
      pattern): the edge must hold across rebalance days.
- [ ] Paper-trade for ≥ 1–2 rebalance cycles; reconcile modelled vs actual fills and slippage.
- [ ] Verify corporate-action and delisting handling on at least one real event.
- [ ] Confirm GBX/GBP handling with a spot check against a known price and known ADV.
- [ ] Confirm the broker supports **fractional UK-equity dealing via API** (or that NAV is large
      enough for whole-share rounding — §7.4); verify a fractional test order fills as expected.
- [ ] Risk sign-off on the expected −40%-class drawdown and the (optional) use of leverage.

---

## 14. Provenance

Derived from the research programme in this repository. Core scripts:
`momentum_survivorship_free.py` (survivorship-free confirmation), `momentum_tradeability.py`
(universe/liquidity/cost engine), `parse_fundamentals.py` (point-in-time fundamentals),
`momentum_multifactor.py` (quality/value overlay), `vol_target_momentum.py` (vol targeting),
`momentum_combined.py` (the combined build), `momentum_phase_robustness.py` (calendar robustness),
`momentum_lookback_sweep.py` (12-1 confirmed optimal), `momentum_universe_sweep.py` (universe size),
`momentum_vol_overlay.py` (low-vol level helps; inverse-vol weighting does not),
`momentum_vol_change.py` (vol change does not help — negative result),
`momentum_stamp_duty.py` (UK SDRT), `momentum_lowvol_build.py` (the recommended build's numbers).
Full results in `momentum-strategy/RESULTS.md`; narrative in `momentum.html`. Backtest
period 2001–2026, net of tiered costs, on a 3,237-name survivorship-free LSE universe.

**Not investment advice. Backtested performance is not a guarantee of future results.**

---

## 15. Validation and re-validation [VALIDATION]

This section collects the requirements that belong to **proving the strategy**, not to running it.
They are what made the historical evidence trustworthy, and what an honest periodic re-check must
repeat. **None of them constrains the live loop** (§1.0): the live system is automatically
point-in-time and automatically free of future data.

1. **Survivorship-free history.** The validation dataset must include delisted / merged / bankrupt
   names with data to their last trading day (the research used 3,237 LSE names: 1,570 live + 1,667
   delisted). Without this, the backtest overstates returns — most of all it overstates the
   *benchmark*, which is what makes the true alpha visible. This is the requirement that §3's live
   feed does **not** need.
2. **Point-in-time reconstruction.** Rebuild the eligible universe and every signal *as it would
   have been known* on each historical date: liquidity from trailing data only; fundamentals by
   filing date (§3.3); and never carry a delisted name's last price forward (the flat-price-zombie
   bug — see §4 and `momentum_phase_robustness.py`).
3. **Enforce no-lookahead mechanically.** In the sim, signals must use only data strictly before the
   holding period. Sanity check: shift every input back one period and confirm the edge degrades as
   expected (§13). This is a property of the *simulation*; live it is automatic.
4. **Data-cleaning caveats that are not trading rules.** The backtest winsorises cross-sectional
   returns (1st/99th pct) to neutralise bad ticks in the archive; live, that becomes real-time tick
   validation, not a return clip (§10.1). Keep both facts in view when comparing live PnL to the
   backtest.
5. **Re-validation cadence.** Periodically re-run the whole chain on fresh survivorship-free data,
   including the calendar-phase test (§13) and the breadth/factor sweeps (§5.5), to confirm the edge
   has not decayed (it is known to be time-varying — §9). Treat a persistent post-period breakdown
   as a regime change, not noise.

The scripts in §14 are the reference implementation of every item above.

---

## 16. Instrument choice and tax wrapper (UK)

**Not tax advice; UK rules and rates change — confirm with a qualified adviser.** This compares how
the same strategy can be *held*, on cost and tax, for a UK investor.

The choice is between owning the shares (in a taxable account, or an ISA/SIPP) and taking synthetic
exposure via CFDs or spread bets. The strategy's shape drives it: **long-only, ~fully invested,
names held a month or more, ~2–3× of NAV bought per year.**

| | Taxable shares | ISA / SIPP shares | CFDs | Spread bet |
|---|---|---|---|---|
| CGT on gains | yes (losses **offsettable** vs gains) | **none** (sheltered) | yes (losses offsettable) | none — but **losses not deductible** |
| Dividend tax | yes | none | via price adjustment | none |
| Stamp duty (0.5% buys) | yes (~1.2–1.6pp/yr, §8) | **yes** (not sheltered) | **no** | no |
| Overnight financing | none | none | **yes, ~SONIA+2–3% on notional** | yes (built into spread) |
| Leverage | no (or margin loan) | no | **yes, native** | yes, native |
| Fractional sizing | needs fractional broker (§7.4) | needs fractional broker | **native** | native |
| Ownership / counterparty | you own it | you own it | broker counterparty risk | broker counterparty risk |

**On the original question — "CFDs to offset losses against tax":** the premise doesn't hold. Share
losses in a taxable account are **already** offsettable against capital gains, so CFDs give no
incremental loss-relief. The product that changes the tax character is the **spread bet** — gains
tax-free but losses **not** deductible, i.e. the opposite of the goal.

**Cost verdict for this strategy:**
- **CFDs avoid stamp duty (~1.2–1.6pp/yr) but pay financing (~2.5–3pp/yr net) on a long book held
  for a month+** — so an *unlevered* CFD book is net **worse** than taxable shares. CFDs pull ahead
  only for the **levered** variant (where you would pay financing either way), and they conveniently
  solve fractional sizing (§7.4).
- **The strongest tax route is usually owning the shares in an ISA/SIPP** — no CGT, no dividend tax —
  for the sheltered portion (ISA subscription is capped, ~£20k/yr; SIPP has its own rules). Stamp
  duty still applies, but the CGT/dividend saving typically dwarfs it for a taxable investor. Above
  the shelter, a taxable share account with loss-offsetting is the fallback.

**Rough tax-efficiency ladder:** ISA/SIPP shares → taxable shares (loss offset) → CFDs (only if you
want leverage or native fractional exposure). Spread bets suit only someone who values the tax-free
gain more than loss-deductibility and accepts the financing spread.

### 16.1 Measured: IG "forward" (quarterly) spread bets — examined 2026-09-08
We checked real IG forward quotes on three large-cap UK shares (Barclays, Tesco, HSBC) across the
SEP-26 / DEC-26 / MAR-27 expiries to see whether front-loading the financing into a quarterly forward
is cheaper than a rolling daily-funded bet. It is **not** — the financing is simply embedded in the
price, and the measured cost is *higher* than the earlier estimate:

- **Embedded net carry ≈ 4.25%/yr**, remarkably consistent across all three names (the forward mid
  rises ~1.06%/quarter: Barclays +5.3 on ~497, Tesco +5.1 on ~480, HSBC +16.7 on ~1569). This is
  (financing − dividend yield); adding back a ~3.5% yield implies **gross financing ≈ SONIA + ~3.75%**
  — wider than an index because these are single stocks. A long buys above spot and, absent a price
  move, settles lower by the carry.
- **Front-contract dealing spread ≈ 0.42–0.44%** round-trip (competitive — *cheaper* than a stamped
  cash-share round trip of ~0.65%, and tax-free on gains). The spread widens with tenor
  (~2 → ~4 → ~6 points for Barclays), so longer-dated contracts cost more to deal.

**Consequence for this (fully-invested, year-round) strategy:** the ~0.2pp per-trade spread advantage
is swamped by ~4.25pp/yr of carry. That drags a ~10% net share build down to **~5–6% ≈ the market
B&H** — the edge is essentially gone — and the tax-free-gains benefit (~2%/yr at best, only in
profitable years, forfeiting loss relief) does not cover a ~4%/yr *certain* drag. **Verdict:
confirmed — forward spread bets are not more efficient here; carry is the decider. They (and CFDs)
make sense only if you specifically want leverage.** The raw quotes are the
screenshots in `momentum-strategy/charts/ig-forward-quotes-2026-09-08/` (with a README
tabulating them). (The auto-close at expiry also
forces calendar-timed exits, and quarterly rebalancing is already worse than monthly — 9.35% / 0.63
vs 10.58% / 0.72, §8 / `momentum_tradeability.py`.)

---

## 17. Risk posture and operating discipline

The mechanics (§0–§12) define *what* to hold. This section defines *how to run it* — the parts that
actually determine whether the edge is captured, because the strategy's failure modes are behavioural
and structural, not analytical.

### 17.1 Leverage — unlevered is the default
Run the strategy **unlevered** (exposure cap = 1.0). The leverage variant (§0, §6.3) is opt-in only,
and the reasons to resist it are strong:
- **It doesn't improve quality.** Leverage lifted CAGR 11.2% → 13.8% but Sharpe barely moved
  (0.89 → 0.90). It's amplification, not alpha — more risk for proportional return, not better return.
- **The backtest doesn't charge the borrowing.** The levered figures apply no financing cost on the
  borrowed portion; a real margin loan (~SONIA + spread) would shave the extra return, narrowing the gap.
- **Vol-targeting lags a fast crash.** Exposure scales off *trailing* vol, so in a sudden crash
  (Feb 2020) you are fully — or, levered, over- — exposed for the first leg down before it can de-risk.
- **The real danger a backtest can't show: forced liquidation.** A paper drawdown you can sit through;
  a margin call forces a sale at the bottom, converting a temporary loss into a permanent one and
  knocking you out of the recovery. **Staying unlevered is precisely what preserves your ability to
  hold through a drawdown** — which is the whole game (§17.2).

### 17.2 Drawdown — pre-commit, don't react
The ≈ −40% drawdown is **not a flaw; it is the price of the premium.** You earn the alpha *because*
you endure drawdowns most people can't — if it were comfortable it would be arbitraged away. So:
- **Decide the rules while calm; follow them mechanically.** In the middle of a −40% drawdown your
  judgement is at its worst.
- **Size for worse than backtest.** Treat this as long-horizon money (5–10 yr) sized so that a
  **−60% drawdown held for years cannot force you to sell** (no leverage, no money you'll need). If
  nothing external forces a decision, holding through is easy.
- **The only legitimate exit is model failure, never pain.** A pre-registered kill-criterion based on
  the strategy *behaving unlike its design* (e.g. live results outside the bootstrap CI for a sustained
  period), decided in advance. **Drawdown depth is not the signal** — healthy strategies have deep
  drawdowns; a broken one behaves wrongly.
- Vol-targeting already automates the "de-risk in turbulence" reflex (it cut −47% → −40%), causally and
  without emotion. Your job is to **not override it in panic.**

### 17.3 Funding — regular contributions, honestly framed
Fund the position with **regular, pre-committed contributions** (fixed nominal, or inflation-linked to
preserve real size). This is the disciplined embodiment of "buy more when it's cheap" — but frame its
value correctly (`momentum_contributions.py`):
- **It is NOT a return-enhancer.** The hoped-for dip-buying bonus did not appear: dollar-cost-averaging
  into the volatile build returned ~12% *less* than into a zero-volatility asset of the same CAGR (you
  also buy fewer units near tops; sequencing dominates). And the dip-buying power **fades fast** — a
  fixed contribution is ~3% of the pot at year 2, <1% by year 5, negligible by year 25.
- **What it genuinely gives:** (a) the optimal, zero-timing-risk way to deploy income as it arrives;
  (b) a **cushioned experienced drawdown** — the portfolio *value* fell ~−30% under a monthly drip vs
  the −40% on invested capital, because inflows soften the fall — which materially aids the discipline
  of §17.2; and (c) it removes the human from the loop (a standing order executes whether you feel
  euphoric or sick — the #1 failure of any "buy the dip" plan is that people freeze).
- Contributions are an **accumulation** discipline, **not** a drawdown-management tool for an
  already-large pot (against £500k invested, a £1k contribution can't offset a −40% fall). Managing the
  existing pot is §17.1–17.2. Fixed vs inflation-linked are near-identical per pound (IRR 10.38 vs
  10.30); inflation-linking just preserves real contribution size. Contributions also map naturally onto
  annual ISA/SIPP allowances (§16).

### 17.4 Reserve deployment — the beta/alpha release rule
If you additionally hold a **bounded, ring-fenced reserve** to deploy into deep drawdowns, release it by
rule, not reflex (`momentum_beta_alpha_monitor.py`). When the strategy's drawdown exceeds a trigger
(e.g. −15%), classify it:
- **BETA-driven** — the market is *also* down (e.g. market drawdown < −10%). The fall is cheap equity
  beta, which has historically recovered. **Release a reserve tranche.**
- **ALPHA-driven** — the market is *not* down but the strategy fell anyway. This is a momentum-specific
  problem (a momentum crash, or genuine decay). **Freeze and investigate** — do not add.

The monitor also reports the **beta-adjusted (alpha) drawdown**; use its depth as a caution light *even
inside a beta drawdown*. Historically (2002–26, β ≈ 0.71) **all seven of the strategy's >15% drawdowns
were beta-driven** — the long-only tilt's drawdowns *are* market drawdowns (momentum-crash risk lives in
the long-short book, not here), so "release" would have fired each time and bought cheap beta ahead of
recovery; in the GFC the strategy fell −40% vs the market's −53%. The ALPHA/freeze branch never fired —
it guards the not-yet-seen case. The one amber flag: Feb 2022 was beta-classified (market −12%) yet had
the deepest alpha drawdown (−14%, the growth→value rotation), which the alpha-DD gauge catches.
Reserve adds must be **bounded and pre-committed** — open-ended "add more as it deepens" is a martingale
and a ruin path.

### 17.5 What not to do
- **Never bail into a drawdown** out of pain — it converts paper losses to permanent ones, usually near
  the bottom. The only exit is the §17.2 model-failure rule.
- **Never add open-endedly** ("more as it falls, funded by whatever I can find") — bounded reserve only.
- **Never use leverage that can trigger a margin call** — it removes your ability to hold, the one
  thing the whole discipline depends on.

---

## 18. Reference implementation: Trading 212 (ISA)

The chosen deployment target. It is the rare broker that satisfies all three hard requirements at
once: **fractional UK-equity dealing** (§7.4), a **Stocks & Shares ISA** wrapper (the best tax route,
§16), and a **REST API** for automated execution — and it is commission-free. *Not an endorsement or
advice; verify current terms yourself. Facts below marked "confirmed" were checked against the cited
sources; anything else is to be settled on the demo account (§18.4).*

### 18.1 Confirmed fit
- **API places orders inside the ISA** (not only the taxable Invest account) — confirmed:
  `https://docs.trading212.com/api` (quickstart). This is the make-or-break item and it passes.
- **Fractional shares are ISA-eligible** — confirmed: HMRC reversed its earlier position, see
  `https://www.ukfinance.org.uk/news-and-insight/blog/hmrc-reverses-position-isa-fractional-shares`.
  Trading 212 supports fractional holdings in the ISA.
- **Order types** — the API supports **limit orders** for the ISA. Fractional orders are placed by
  specifying the **share quantity** (a fraction), not a cash value — so the pipeline computes
  `quantity = target_weight × NAV / raw_price` (§7.2) and sends that.
- **Commission-free**, and turnover is low (~4–6×/yr, ~69 names), so API rate limits are not a
  constraint for a monthly rebalance.

### 18.2 Architecture — data vs execution are separate
Trading 212 is the **execution and custody layer only**. It does not supply the survivorship-free
history or point-in-time fundamentals needed to *compute* the signals. The live pipeline is:

```
EODHD (or equivalent)  ->  signal engine (§4–§6)  ->  target weights
                                                        |
                                    reconcile vs current positions (T212 API)
                                                        |
                              fractional-quantity limit orders (T212 API, ISA)
```

Balances and positions are read back from the T212 API each cycle for reconciliation (§13).

### 18.3 The fractional-coverage filter (the one open mechanical detail)
Not every LSE name is fractionally dealable: FTSE 100 is, but less-liquid mid/small-caps may be
whole-share-only. The API exposes a per-instrument fractional flag — **use it to restrict the
eligible universe to fractional-tradeable names** (or whole-share-round / drop the rest). Because the
edge does not need the illiquid tail (§5.5, universe sweep), this should cost little.

Whole-share impact at **£10,000** was measured (`wholeshare_10k.py`) against the actual held names and
real prices: at ~69 names (£145/name) only **~5%** of held names are too pricey to hold as one whole
share — and the highest-priced ones (AZN ≈ £124, RIO ≈ £52) are FTSE 100 and fractional anyway; a
pessimistic "only top-100 fractional" assumption gives ~6.7% per-name tracking error, which
diversifies down at book level and is lower in reality (T212 fractional-izes more than 100 names).
**Concentrating to ~20–30 names roughly halves the problem** (£333–500/name; ~1–2% un-holdable) — and,
per §5.5, a concentrated book also fits a small NAV better and returns more (at a deeper drawdown). So
the **small-account (£10k) configuration is: fractional-filtered universe + a ~20–30 name book.**

### 18.4 Demo account = the go/no-go gate
Trading 212 offers a **practice (demo) account with its own API key**. Validate the entire integration
there before any real capital — this is the concrete form of §13 and §17.2's "prove it first":
1. Pull the **live instrument list + fractional flags** — this replaces the top-100 assumption with the
   *true* fractional universe, settling §18.3.
2. Place **test limit orders in fractional quantity** and confirm fills, precision, and rejections.
3. Confirm the API reads/writes the **ISA** account; reconcile balances and positions against the model.
4. Run **one full demo rebalance end-to-end** and reconcile modelled vs actual holdings — treat a clean
   reconciliation as the go/no-go gate.

### 18.5 Costs and caveats
- **Commission-free ≠ free:** 0.5% UK **stamp duty** still applies on purchases (~1.2pp/yr, the largest
  friction — §8, unavoidable in any wrapper), plus the bid/offer spread and any current platform/FX
  fees (check T212's live schedule; FX is irrelevant for GBP/GBX LSE names).
- **Deposit fee — avoidable, confirmed.** T212 charges 0.7% on deposits above £2,000 via card/e-wallet,
  but **bank transfers are free** (confirmed 2026-09-08) — so funding by bank transfer avoids the fee
  entirely. Even if incurred it would be a one-off toll on capital *as it enters* (reducing invested
  principal ~0.7%, **not** lowering the IRR; ~£56 on a £10k lump), but via bank transfer it is a
  non-issue. Fund by bank transfer.
- **ISA subscription cap** (~£20k/yr) fits the regular-contribution discipline (§17.3) but limits how
  fast the sheltered pot can grow.
- **Counterparty/platform:** a newer platform; FSCS protection is £85k. A consideration at larger NAV,
  not at £10k.
