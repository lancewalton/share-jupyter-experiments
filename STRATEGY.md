# The Defensive Stack — operational specification

A precise, implementable definition of the strategy pitched in *The Defensive
Stack* / *The Volatility Axis*. This is the "how to run it" document; for the
"why", see those write-ups. Every parameter below matches
`vol-risk-experiment/run_composed.py`.

> **One line.** Hold a low-volatility equity tilt, add a market-neutral 12-1
> momentum sleeve, and size the whole book to a constant volatility target. No
> forecast of market direction anywhere. Net Sharpe ≈ 0.51, max drawdown ≈ −38%,
> market-beta ≈ 0.41 (backtest, UK equities 2001–2026, net of 10 bps; data
> current to 2026-08-20).

---

## 1. Objective & scope

- **Objective:** capture the equity + momentum premia at materially better
  risk-adjusted return and lower drawdown than the market — *not* to time
  direction.
- **Benchmark:** the equal-weight investable universe.
- **Asset class:** a single equity market's cross-section (spec uses UK equities;
  the method is market-agnostic). **Do not mix asset classes** — FX/intraday
  findings are separate and do not transfer.
- **Rebalance frequency:** every **H = 20 trading days** (≈ monthly).

## 2. Universe & data

- **Instruments:** all names with ≥ **1500** daily observations of history.
- **Inputs:** daily close prices → daily log returns `r`.
- **Glitch filter:** exclude any series whose max \|daily log return\| > **0.6**
  (splits / corporate-action artefacts).
- **Point-in-time discipline:** every quantity used to set a position on day *t*
  must be computable from data available strictly **before** the holding period.
  Any filtered/online signal (e.g. a regime probability) must be **lagged one day**.
  Treat a surprisingly good backtest as a look-ahead bug until proven otherwise.
- **Provenance:** history from investing.com; extended forward to **2026-08-20**
  from the Yahoo chart API, **chain-linked** at the join date (scale the appended
  tail by `file_close / source_close` so returns are continuous and unit/
  adjustment differences are absorbed). Names delisted from the LSE (moved listing
  or acquired) freeze at their last date and self-exclude from later rebalances via
  the rolling-window `dropna`. Keep sources internally consistent; do **not** mix
  in another asset class (FX/intraday findings are separate).

## 3. Components

### 3a. Core — low-volatility tilt (long-only)

- **Volatility forecast** `σ_i,t` per stock: RiskMetrics EWMA of squared daily
  returns, `var_t = λ·var_{t-1} + (1−λ)·r_{t-1}²`, **λ = 0.94**, seeded on the
  first **60** observations; `σ = √var`. Causal (uses data before *t*).
- **Selection:** at each rebalance, rank the eligible cross-section by `σ`
  (ascending); go **long the lowest quintile (20%)**.
- **Weighting:** equal-weight within the quintile; total gross = **1.0**,
  long-only. *(Variant: inverse-vol weighting within the quintile — marginal.)*
- **Hold** until the next rebalance.

### 3b. Sleeve — 12-1 cross-sectional momentum (market-neutral)

- **Signal** `M_i,t` = cumulative log return over the past 12 months **skipping the
  most recent month**: `M = Σ r` over `[t−252, t−21]`. In code: a 231-day rolling
  sum, shifted forward 21 days.
- **Selection:** rank by `M`; **long the top quintile, short the bottom quintile**.
- **Weighting:** equal-weight within each leg, **dollar-neutral** — long leg
  +0.5 gross, short leg −0.5 gross (sleeve gross = 1.0, net = 0).
- **Purpose:** it is ~uncorrelated with the core (−0.03) and negatively correlated
  with the market (−0.31), so it **diversifies** rather than doubling equity beta.
  Run it as a **separate sleeve** — do **not** merge its score with the core's
  (combining signals of unequal strength dilutes the strong one).

### 3c. Sizing — volatility targeting the combined book

- **Combined pre-sizing book:** `B = core + W_MOM · sleeve`, **W_MOM = 0.5**.
  (Gross exposure ≈ 1.5, net ≈ 1.0 long.)
- **Book vol forecast:** EWMA (λ = 0.94, burn 60) of `B`'s own daily returns,
  annualised ×√252.
- **Exposure multiplier:** `m_vt = clip( TARGET / σ_B,ann , 0, MAXLEV )`,
  **TARGET = 12%** annualised, **MAXLEV = 2.5**. Lever up in calm, cut in
  turbulence — constant-risk sizing.
- **Final book:** `m_vt · B`.

### 3d. Optional — the systemic regime overlay

A market-level regime detector (HMM, causal, lagged one day) that de-risks in
crises is, on the corrected data, **roughly Sharpe-neutral** for this stack
(composed 0.51 → 0.53) while **cutting risk**: max drawdown −54% → −37%
(log-space) and market-beta 0.41 → 0.27, at the cost of a little return
(ann. 6.2% → 5.2%). Treat it as an **optional defensive layer** — add it when the
mandate prioritises drawdown/beta reduction over marginal return; omit it for
maximum return. *(This reverses an earlier "do not add" verdict that showed a
sharp 0.58 → 0.31 drop — an artifact of stale data in which 19 large names were
frozen at 2023. Worth a fuller robustness re-check before promoting to core.)*

## 4. Portfolio assembly (per rebalance)

```
1. Refresh eligibility (history ≥ 1500d, glitch filter).
2. Compute σ_i (EWMA vol) and M_i (12-1 momentum) for all eligible names.
3. CORE   = long lowest-σ quintile, equal-weight, gross 1.0.
4. SLEEVE = long top-M quintile / short bottom-M quintile, equal-weight,
            dollar-neutral (±0.5 gross).
5. B      = CORE + 0.5 · SLEEVE.
6. σ_B    = EWMA vol of the book's recent daily returns (annualised).
7. m      = clip(0.12 / σ_B, 0, 2.5).
8. TARGET WEIGHTS = m · B.  Trade toward them; hold to next rebalance.
```

Between rebalances, `m` may be refreshed daily (the vol forecast updates) while
the underlying core/sleeve weights are held — this is where most day-to-day
turnover, and its cost, comes from; refresh `m` on a band (e.g. only when it
moves > 10%) to contain it.

## 5. Risk management

- **Ex-ante risk target:** 12% annualised on the combined book (the vol-target).
- **Leverage cap:** gross ≤ 2.5 × (1.5 base) via `MAXLEV`.
- **Tail / VaR:** size and set limits from the **calibrated FHS volatility
  envelope** (from the monte-carlo-experiment package), which gives honest tail
  coverage — do not use a naive Gaussian VaR, which under-covers.
- **Market-beta:** monitor; the stack ran ≈ 0.22. A drift toward 1.0 means the
  momentum hedge has decayed and the book is becoming long-only beta.
- **Momentum crash control:** the sleeve's worst month was −15%. Cap the sleeve's
  gross contribution (the 0.5 weight) and consider a per-sleeve vol-scale;
  neither fully removes the crash, so size the sleeve to survive it.

## 6. Costs & turnover

- **Assumption used:** 10 bps per unit turnover, applied to both the monthly
  rebalance and the daily `m` resizing. Results above are **net** of this.
- Both core and sleeve are chosen partly for **low turnover**; keep it that way.
  Every edge here is modest and cost-fragile — turnover discipline is not
  optional. Widen the rebalance band before adding any higher-frequency signal.

## 7. Parameters reference

| Parameter | Symbol | Value |
|---|---|---|
| Rebalance horizon | H | 20 trading days |
| EWMA vol decay | λ | 0.94 |
| EWMA seed / burn-in | — | 60 days |
| Low-vol / momentum quintile | QUINT | 20% |
| Momentum formation (skip 1m) | — | Σ r over [t−252, t−21] |
| Momentum sleeve weight | W_MOM | 0.5 |
| Book volatility target | TARGET | 12% annualised |
| Max exposure multiplier | MAXLEV | 2.5 |
| Turnover cost | COST | 10 bps |
| Min history / glitch filter | — | 1500 days / \|r\| ≤ 0.6 |

## 8. What it is / isn't

- **Earns:** the equity premium held efficiently (lower vol, shallower drawdown),
  a modest diversifying momentum premium, and genuine downside protection.
- **Does not earn:** anything from predicting direction, timing, or chart
  patterns. It is not a standalone-fund alpha engine.

## 9. Known limitations & governance

- **Execution — market-on-open, T+1.** Decide on tonight's close; submit
  **market-on-open orders for the next session**. The overnight gap rides on the
  *old* book, the new weights earn from the open forward — no look-ahead, and every
  intended name fills. Three models were compared (`run_execution.py`, decomposing
  `r = overnight + intraday`): **A** trade-at-observed-close (idealised),
  **B** T+1 close (conservative), **C** T+1 open (realistic) gave **6.7 / 6.8 / 7.0%
  CAGR** and **0.55 / 0.56 / 0.58 Sharpe** — indistinguishable. Realism is ~free
  here *because* the book turns over monthly; the entry-day gap is noise against a
  20-day hold. The same rule on a daily/intraday book would **not** be free (mirror
  finding from the FX intraday work — kept separate).
  - **Limit orders were tested and rejected.** A patient buy-limit *at* the
    decision close fills at that price only if the next day trades down to it
    (Low ≤ close), so the effect is pure **adverse selection**: the gap-up names
    you skip are the winners (forward-20d **+2.1%** low-vol / **+2.9%** momentum,
    vs **+0.2%** for the fills). On the core that cut per-cycle return 0.51% → 0.39%
    (selection only) → 0.31% (cash on the ~10% unfilled) and Sharpe 0.47 → 0.36 →
    0.30, worse for momentum (`run_limit.py`). A name that won't trade back to your
    price is signalling strength; waiting for the pullback systematically avoids it.
    **Cross the spread and take the fill.**
- **Decaying edge:** backtest Sharpe by thirds was +1.49 / +0.29 / **−0.16** —
  most earned in the first third. Momentum's contribution has faded (post-
  publication decay); crisis benefits are concentrated in 2008/2020. Year-by-year
  recently: 2022 −14.6%, 2023 +3.5%, 2024 +0.5%, 2025 +11.2%, 2026 YTD +2.7%.
  **Review the momentum sleeve's live IC and market-beta quarterly; de-weight if
  the hedge fades.**
- **Single market / sample:** UK equities, 2001–2026. Re-validate out-of-sample
  on the intended market, with realistic borrow, capacity, and cost assumptions,
  before committing capital.
- **Shorting:** the momentum sleeve shorts the bottom quintile — requires borrow;
  a long-only implementation captures only part of it.
- **Not investment advice.** A validated *research* portfolio, honestly caveated.

## 10. Reproduce

`vol-risk-experiment/run_composed.py` (the composed backtest, net, no look-ahead),
`run_momentum_profile.py` (sleeve risk profile), `run_lowvol.py` (the core),
`run_pricefactors.py` (the momentum factor), `run_execution.py` (A/B/C execution
models), `run_limit.py` (limit-order adverse-selection test). Uses the shared
`mc` package.
