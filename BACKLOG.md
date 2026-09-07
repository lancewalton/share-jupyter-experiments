# Ideas backlog — share-jupyter-experiments

Cross-project running list of things to try. Four projects so far:
`monte-carlo-experiment` (calibration/FHS), `pattern-discovery` (direction),
`heirarchical-adaptive-filter-experiment` (haf, volatility), `vol-risk-experiment`
(risk sizing + regimes). Overarching finding: **value is in volatility, not
direction; robust/adaptive beats overfit/complex; the ~1% SNR ceiling on direction
is real.**

Status: [ ] todo  [~] in progress  [x] done  [-] dropped

## vol-risk-experiment (active)
- [x] Fold the regime-conditional width correction into **full FHS** (multi-day)
      — REDUNDANT: FHS's vol propagation already handles the turbulent regime
      (multi-day turbulent coverage already near/below nominal), so the correction
      over-corrects it. FHS's residual 99% tail gap is GENERAL (both regimes, worse
      in calm), not regime-localised. Regime correction helps only a NAIVE envelope
      (EWMA-Gaussian, Part 3), not FHS. (`run_regime_fhs.py`)
- [ ] **3-state HMM** (calm / normal / crisis) — 2-state left turbulent at 6.5% > 5%.
- [x] **Regime-conditional sizing / DO THE PIECES COMPOUND** (`run_defensive.py`):
      YES. low-vol selection + vol-targeting + regime overlay: market Sharpe 0.25 →
      0.61, maxDD −105% → −38%, zero direction. Vol-target biggest genuine add;
      regime overlay adds risk reduction. CAVEATS: caught a LOOK-AHEAD (regime filter
      peeked at same-day return → inflated Sharpe to 1.14; lagged → 0.61); edge is
      CRISIS-CONCENTRATED (sub-thirds 1.53/0.56/−0.23); overlays gross of turnover.
- [ ] Regime object on **individual stocks / cross-asset** (fit per series) — is the
      turbulent→forward-return sign different (where vol-targeting *does* help)?
- [ ] **Options / variance-risk-premium** strategy — the one untested place for real
      directional-independent alpha. BLOCKED: needs option-implied-vol data.

## monte-carlo-experiment (calibration) — largely complete
- [ ] Recalibrate the fhs_stack pooled fit's mild body over-coverage.
- [ ] Feed the **regime correction** back here as an FHS upgrade (= the active item).

## pattern-discovery (direction) — largely exhausted
- [ ] Other **vol-normalised amplitude** measures beyond peak-to-trough depth
      (e.g. up/down-move asymmetry, realised-skew) — do any add independent signal?
- [-] Neural net / signed combination for direction — tested, fails (low SNR).

## equities / longer-term (overnight queue, 2026-08-20)
- [x] **1. 3-state HMM** (`run_regime3.py`) — does NOT beat 2-state. Middle "normal"
      state DEGENERATES into a 1-day jump-catcher (residual std 2.6) that poisons the
      soft-blend correction; hard-assignment+clip fixes calm well but crisis-state
      95% breach 0.062 ≈ 2-state turbulent 0.065 (no gain). Residual crisis 99% 0.022
      is fat-tail risk width-scaling can't fix. 2-state remains robust. Robust>complex.
- [x] **2. Regime object on individual stocks** (`run_regime_stocks.py`, 142 stocks):
      FTSE pattern GENERALISES — 80% of stocks have turbulent-regime forward return >
      calm (turbulent +1.3% vs calm +0.2% /20d) = individual-stock turbulence is a
      REBOUND precursor (idiosyncratic mean-reversion), not continuation. Per-stock
      regime-TIMING (exit when turbulent) HURTS: Sharpe 0.16→0.13, wins 45%. KEY:
      regime de-risking is a MARKET/SYSTEMIC tool, NOT per-stock (per-stock sells
      idiosyncratic bottoms before they bounce). Explains why vol-targeting was modest.
- [x] **3. S&P 500 vol-targeting + regime** (`run_sp500.py`, FRED 2016-26, 2511d):
      vol-targeting HELPS on the high-premium S&P (Sharpe 0.70→0.82, maxDD −41%→−25%) —
      the Moreira-Muir benefit that was ABSENT on weak-premium FTSE. So FTSE failure was
      the MARKET not the method; vol-targeting benefit is PREMIUM-DEPENDENT. Turbulent→
      rebound holds here too (calm +0.85% ≈ turb +0.87% fwd); P(turb) fwd-vol IC 0.50.
      Market-level regime de-risking helped recent OOS half (Sharpe 0.64→0.73, maxDD
      −29%→−14%, cut 2022) — consistent w/ Task 2 (systemic not per-stock). Caveats:
      10y (no 2008), single index, edge period-dependent.
- [x] **4. Longer-term price factors** (`run_pricefactors.py`, 124 UK eq, 2004-2026):
      **12-1 MOMENTUM is a genuine winner** — rank IC +0.028 (t=2.0), market-neutral L/S
      net Sharpe **+0.34**, beta −0.13. The best market-neutral equity factor found in
      the whole programme, and DIRECTIONAL (reconciles "direction is a mirage": SHORT-
      horizon direction unpredictable, but 12-MONTH cross-sectional momentum is real).
      Long-term REVERSAL (5y-1y) FAILS: IC −0.028 (t=−2.2) — no De Bondt-Thaler; weak
      long-horizon momentum PERSISTENCE instead. Low-vol as L/S weak (net +0.06 — a
      TILT not an arb). COMBINING HURTS: MOM+LOWV net 0.19 < MOM 0.34; full combined
      −0.06 — weak factors dilute momentum. Caveats: Sharpe 0.34 modest, momentum has
      known crash risk. => TWO implementable equity edges total: low-vol TILT (long-only)
      + 12-1 MOMENTUM (market-neutral); they don't combine well.

## portfolio strategy (2026-08-20)
- [x] **Momentum-sleeve risk profile** (`run_momentum_profile.py`): 12-1 mom L/S Sharpe
      0.55 gross / 0.34 net, worst month −15% (crash real), sub-period 1.25/0.19/0.23
      (DECAYED post-2009). ≈uncorrelated w/ low-vol core (−0.03), −0.31 to market → good
      diversifier. Vol-scaling lifts Sharpe not crash.
- [x] **Full composed strategy** (`run_composed.py`, net 10bps, no look-ahead): low-vol
      core + momentum sleeve + vol-target = **Sharpe 0.58, maxDD −39%, beta 0.22** (market
      0.25/−105%/1.00). Layers compound 0.21→0.39→0.58. **Regime overlay HURTS once
      momentum present (0.58→0.31) → DROPPED** (momentum already hedges turbulence).
      Caveat: recent-third Sharpe −0.28 (momentum decay + crisis-concentration).
- [x] **Strategy write-up** — `strategy.html` Artifact "The Defensive Stack" (pitch:
      architecture, composed backtest, discipline, caveats).
- [x] **Operational strategy spec** — `STRATEGY.md` (repo root): the implementable
      definition — exact signal formulas, parameters table, rebalance procedure/pseudocode,
      risk mgmt, costs, governance. Grounded in run_composed.py.

## execution realism (2026-08-20)
- [x] **Execution model A/B/C** (`run_execution.py`, uses the Open column): the "trade
      at observed close" assumption is NOT flattering — A (instant close) CAGR 6.7% /
      Sharpe 0.55, B (T+1 close) 6.8%/0.56, C (T+1 open, realistic, splits overnight vs
      intraday) 7.0%/0.58 — within noise, A even lowest. Monthly diversified book: 1-day
      execution timing washes out (mirror of intraday where it dominated). Realism matters
      INVERSELY with holding period.
- [x] **Limit-at-decision-close execution** (`run_limit.py`, uses Low column): fill AT
      close C_d if next-day Low≤C_d, else skip the gap-up. Fill PRICE a non-issue (=C_d);
      effect is pure SELECTION — and it's badly ADVERSE. Skipped (gapped-up) names' fwd-20d
      return +2.08%(low-vol)/+2.93%(mom) vs filled +0.19%/+0.24% → you skip the WINNERS.
      Low-vol core per-cycle return 0.511%→0.391% (renorm, pure selection)→0.309% (cash on
      unfilled); Sharpe 0.47→0.36→0.30. Worse for momentum (wants strong names). A name
      that won't trade back to your price is signalling STRENGTH; limit orders avoid
      strength. Sell-side symmetric (sell into strength) compounds it. VERDICT: HURTS;
      cross the spread & take the fill for a continuation-content book.

## data / OOS refresh (2026-08-20)
- [x] **Refresh FTSE index 2021→2026** (Yahoo ^FTSE chart API; `run_ftse_extended.py`;
      `/Users/lance/Projects/FTSEData/all_extended.csv`, orig all.csv PRESERVED).
      Consistency check PERFECT (Close corr 1.00000, median |%diff| 0.000%) — index not
      div-adjusted so no artefact. Now 1990→2026-08 (9350 rows). Findings: (a) REGIME
      DETECTOR GENERALISES to OOS crises — fit pre-2015, causally flagged 2020 COVID (61%
      turbulent days) & 2022 bear (27%) it never trained on = robustness confirmed;
      (b) vol-targeting STILL doesn't help FTSE (full 0.24→0.21; 2022 bear 0.06→−0.24
      HURT) — premium- AND crisis-TYPE-dependent (helps sharp vol-SPIKE crashes 2008/2020,
      not slow GRINDS 2022; FTSE resilient in 2022 +0.9% via commodity weighting).
      NB constituents (yfinance) already reach 2026 — NOT re-fetched (survivorship +
      source-mixing risk > marginal gain).

## cross-cutting / new
- [x] **Cross-sectional volatility** strategy (low-vol anomaly, `vol-risk-experiment/
      run_lowvol.py`) — FIRST clearly implementable edge in the whole project. LONG-ONLY
      low-vol tilt: Sharpe 0.32 vs market 0.22, drawdown −56% vs −99% (same return, less
      risk) — real & usable. Market-neutral L/S: FAILS (alpha t=0.85 insignificant,
      break-even 6bps, net negative, mostly −0.38 disguised beta). Anomaly is a TILT,
      not an arb. Thesis (value on the vol axis) cashed out.
- [x] **Intraday realised volatility** (`vol-risk-experiment/run_intraday.py`) — computed
      daily RV from FX minute bars (~1400 bars/day) and forecast next-day vol. Intraday
      HAR-RV nearly DOUBLES OOS R² over the daily-return EWMA (0.151→0.274, +81%; beats
      it on all 4 pairs, GBPUSD EWMA even negative). Single-lag: RV predicts tomorrow's
      RV at IC 0.47 vs 0.09 for daily r² — 5× cleaner MEASUREMENT (RV averages 1400 bars
      vs r²'s 1). => the daily-r² EWMA used everywhere (FHS/calibration/regime/sizing)
      is a 5× noisier vol input than necessary wherever intraday data exists — the
      single biggest available upgrade to VOL FORECASTING per se. (Scope: FX 2010-17;
      HAR-RV documented, reproduced.)
- [x] **Does intraday RV improve FX envelope CALIBRATION?** (`run_fx_calib.py`, 50 FX
      pairs) — NO, slightly worse (mean calib error 0.042→0.045; tail 99% cov 0.940 vs
      EWMA 0.967). RV measures CONTINUOUS INTRADAY vol; the daily-return envelope must
      cover OVERNIGHT GAPS + JUMPS that RV excludes/smooths → too-smooth forecast
      under-covers the fat-tailed daily return. Better measurement ≠ better deliverable
      when it measures the wrong thing. (FX & equity kept strictly separate — user.)
- [x] **Overnight-gap fix for daily calibration** (`run_fx_daily_overnight.py`, per-day
      open/close → `scratchpad/fx_oc.csv`): total daily var = RV + overnight² RECOVERS
      the tail (99% cov 0.940→0.957) and slightly BEATS the daily EWMA overall (calib
      error 0.0410 vs 0.0418). Diagnosis confirmed: overnight was the missing piece.
      Residual 99%<nominal is irreducible jump risk. Intraday-informed forecast now
      competitive+ for daily returns.
- [x] **INTRADAY-return calibration** (`run_fx_intraday_calib.py`, 50 FX): user's insight
      CONFIRMED — daily EWMA OVER-covers the intraday return (95% cov 0.961 vs 0.95;
      calib err 0.073) because it carries overnight var an intraday trader never bears.
      Intraday-variance forecasts (RV or intraday-EWMA) calibrate well (err 0.053).
      NUANCE: HAR-RV ties intraday-EWMA on COVERAGE (0.0532 vs 0.0531) — RV's precision
      doesn't help calibration (coverage rewards LEVEL not precision — same as depth /
      per-series stack). RV's edge would show in SIZING stability, not coverage. Body
      over-covers (fat tails → want t/FHS intraday envelope).
- [x] **Intraday EDGE test** (minute-return moments → `scratchpad/fx_minute_moments.csv`):
      short-horizon REVERSAL is REAL — lag-1 ρ negative in 88% of liquid pairs (mean
      −0.04; EURGBP −0.073). But per-trade edge ~0.04–0.18 bp vs ~1 bp round-trip
      spread-bet cost → ~6× too small; loses ~0.9 bp/trade net. INTRADAY FX PICTURE
      COMPLETE: real structure both axes (vol forecastable + measurable; reversal real)
      but NEITHER tradeable after costs — spread eats the edge. Same "real signal, not
      tradeable" wall, brutal at HF.
- [ ] (residual intraday, low priority) fat-tailed intraday envelope; RV vol-targeting
      sizing-stability demo; intraday vol seasonality (U-shape); jump/continuous split.
- [ ] **Higher-premium assets** for vol-targeting (S&P500 via haf's FRED loader) —
      does the benefit show up where the equity premium is stronger?
- [x] **Cross-project SYNTHESIS** — `synthesis.html`, published as a private Artifact
      ("The Volatility Axis"): four-project story, three through-lines, FHS table,
      defensive-portfolio payoff, honest caveats. Covers all four at a higher level.
- [ ] Standalone write-ups of **vol-risk** and **haf** (only MC + pattern-discovery
      have their own WRITEUP.md so far).
