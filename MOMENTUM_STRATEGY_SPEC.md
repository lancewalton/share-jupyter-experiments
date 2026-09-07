# Momentum Strategy — Implementation Specification

**Strategy:** quality-tilted 12-1 cross-sectional momentum, long-only, volatility-targeted, on
survivorship-free UK equities.
**Status:** research-validated (see `trend-channel-experiment/RESULTS.md` and `momentum.html`);
this document specifies how to build it as an automated system.
**This is not investment advice.** It is a specification of a backtested strategy, with its
known limitations stated explicitly. Deploy real capital only after independent validation and
appropriate risk/compliance review.

---

## 0. Recommended build (the default this spec describes)

| | Value |
|---|---|
| **Signal** | Equal-weight composite of 12-1 momentum + quality (ROE + gross-profitability) |
| **Universe** | Survivorship-free LSE common stock (GBP/GBX), top 350 by trailing liquidity |
| **Selection** | Long the top quintile of the composite (~69 names) |
| **Weighting** | Equal weight, then scaled by a volatility target |
| **Vol target** | 15% annualised, exposure cap 1.0 (unlevered) |
| **Rebalance** | Monthly |
| **Backtest result** | CAGR ≈ 10.9%, Sharpe ≈ 0.83, max drawdown ≈ −39%, net of tiered costs |

**Optional leverage variant:** raise the exposure cap to 1.5. Backtest: CAGR ≈ 12.9%, Sharpe ≈
0.82, max drawdown ≈ −42% — roughly +2pp/yr of return for +3pp of drawdown, the most efficient
point on the frontier. Everything else is identical. Do not add the *value* factor to a levered
book: value has a much deeper standalone drawdown (−61%) and levering it is expensive.

The rest of this document specifies the **default (unlevered)** build; the leverage variant is a
one-parameter change (§6.3).

---

## 1. Scope and assumptions

- **Market:** London Stock Exchange, GBP- and GBX-quoted common stock. The edge was validated on
  UK equities only; do not assume it transfers unchanged to other markets (US equities show the
  same effect; extend only after re-validation).
- **Instrument type:** cash equities, long-only. No derivatives, no shorting. (A market-neutral
  long/short version exists but carries momentum-crash tails — out of scope here.)
- **Reference data vendor:** EODHD "All-In-One" was used to build the research dataset
  (survivorship-free EOD OHLCV, dividends, splits, fundamentals). Any vendor providing the same
  point-in-time, delisting-inclusive data is acceptable; §3 lists the requirements vendor-agnostically.
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

All series must be **survivorship-free**: they must include companies that later delisted,
merged, or went bankrupt, with data up to their last trading day. Using only current constituents
inflates every backtest and will not reproduce live behaviour.

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

**Availability date rule:** use `filing_date` if present, else `fiscal_period_end + 120 days` as
a conservative proxy. A fundamental value must **never** enter a signal before its availability
date. (The research parser `parse_fundamentals.py` implements exactly this.)

**Currency note:** the recommended build's quality factor is built entirely from **ratios**
(ROE = net income / book equity; GP/A = gross profit / total assets), which are **currency-neutral**.
No FX conversion is required. (This is a deliberate reason to prefer quality over value for the
default build: value needs price and accounts in the same currency and is therefore restricted to
a subset of names.)

---

## 4. Universe construction (per rebalance date `t`)

1. **Base universe:** all LSE common stocks (GBP/GBX), live **and** delisted, that have a valid
   adjusted close on `t`. A name that has stopped trading has no price on `t` and is automatically
   excluded — do **not** carry a stale/last price forward (that resurrects dead companies; it was a
   bug caught during research — see the note in `momentum_phase_robustness.py`).
2. **Liquidity ranking:** for each name compute trailing **12-month mean daily turnover** (mean of
   daily GBP turnover over the trailing ~252 trading days, requiring at least ~60 valid days).
3. **Eligible set:** the **top `N = 350`** names by that liquidity measure. This is the tradeable
   universe for date `t`. All subsequent cross-sectional calculations are over this set.
4. **Minimum breadth:** require at least **30** eligible names with a valid signal and a valid
   forward return; otherwise skip the rebalance (relevant only to early history / cold start).

---

## 5. Signal construction (per rebalance date `t`, over the eligible set)

### 5.1 Momentum factor
For each eligible name:
```
mom_raw = adj_close[t − 1 month] / adj_close[t − 12 months] − 1
```
Then standardise cross-sectionally: `z_mom = zscore(mom_raw)` over the eligible set
(`zscore(x) = (x − mean) / std`).

### 5.2 Quality factor
Using the latest fundamentals **available on or before `t`** (by availability date, §3.3):
```
ROE = net_income / book_equity        (only where book_equity > 0)
GPA = gross_profit / total_assets     (only where total_assets > 0)
quality_pct = 0.5 · rank_pct(ROE) + 0.5 · rank_pct(GPA)   (percentile ranks over eligible set)
z_qual = zscore(quality_pct)
```
Names with no available fundamentals have `z_qual = NaN` (they contribute only their momentum
score — see 5.3).

### 5.3 Composite
```
composite = mean( [z_mom, z_qual] , skipping NaNs )
```
**Momentum is mandatory:** drop any name without a valid `z_mom`. Quality is additive where
present. (A name with momentum but no quality is ranked on momentum alone — this is intentional and
matches the validated build.)

### 5.4 Selection
Rank the eligible names with a valid composite and a valid forward-return capability. Let `M` be
their count. Select the **top quintile**:
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
- **Timing / lookahead discipline:** compute signals from prices/fundamentals known **strictly
  before** the execution point. A safe pattern: use data through close of day `D`, generate target
  weights, execute at or after the open of day `D+1`. Never compute a signal from the same print
  you then trade on.

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
  a real, expected event in a survivorship-free universe and must be handled operationally.

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
- **Cost model (for expectation-setting):** liquidity-tiered round-trip spreads applied to traded
  notional — **15 bps** (top liquidity tercile), **40 bps** (mid), **80 bps** (low). Cost per
  rebalance ≈ `Σ |Δweight_i| × spread_i / 2`.
- **Break-even:** the long-only edge survives round-trip costs up to ≈ **200 bps** before it stops
  beating buy-and-hold — an order of magnitude above realistic execution. **Costs are not the
  binding constraint.**
- **Capacity:** ~69 equally weighted names, but the low-liquidity tier caps size. Constrain each
  order to a small fraction of the name's trailing average daily volume (e.g. ≤ 5–10% of 20-day
  ADV) and spread execution over the day (or multiple days) if a target exceeds that. At scale,
  consider capping individual position size relative to ADV, which slightly reduces the effective
  quintile but protects fills.

---

## 9. Risk management and expected behaviour

- **This is a high-beta long equity tilt** (market β ≈ 0.84), **not** market-neutral. Roughly
  two-thirds of the return is equity-market beta; the value-add over that is ≈ +6%/yr.
- **Expect deep drawdowns.** Backtested max drawdown is ≈ **−39%** (unlevered) / **−42%** (levered
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
| Factor combine | — | equal-weight z-scores; momentum mandatory | |
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
# Nightly / pre-rebalance data pipeline
maintain daily survivorship-free panel: raw_close, adj_close, volume, currency  (validate ticks)
maintain corporate-actions ledger
maintain point-in-time fundamentals keyed by filing date

# On each rebalance date t:
elig      = top 350 names by trailing-12m mean daily GBP turnover, having a valid adj_close at t
if count(elig) < 30: skip

mom_raw   = adj_close[t-1m] / adj_close[t-12m] - 1                 # over elig
z_mom     = zscore(mom_raw)
ROE, GPA  = latest PIT fundamentals available <= t                # over elig
z_qual    = zscore( 0.5*rank_pct(ROE) + 0.5*rank_pct(GPA) )
composite = nanmean([z_mom, z_qual]); drop names with NaN z_mom

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
`momentum_combined.py` (the combined build), `momentum_phase_robustness.py` (calendar robustness).
Full results in `trend-channel-experiment/RESULTS.md`; narrative in `momentum.html`. Backtest
period 2001–2026, net of tiered costs, on a 3,237-name survivorship-free LSE universe.

**Not investment advice. Backtested performance is not a guarantee of future results.**
