# NEXT STEPS — resume here

**Status (2026-09-08):** the momentum strategy is fully researched, specified, and documented.
Deployment is **paused pending Trading 212 account authorization (~2 weeks, so ~late Sept 2026).**
Nothing is blocked on analysis — the only blocker is the broker sign-up. Pick up at §"When the
T212 account is authorized" below.

---

## The strategy in one line
**12-1 cross-sectional momentum, tilted by quality and low-volatility, vol-targeted (unlevered),
long-only, monthly-rebalanced, on a survivorship-free UK top-350 universe — held in a Trading 212
Stocks & Shares ISA.**
Backtest (net of spread): CAGR ≈ 11.2%, Sharpe ≈ 0.89, max drawdown ≈ −40%; ≈ 10.1% / 0.81 after
UK stamp duty. Market buy-and-hold for comparison: ≈ 5.65%.

## Where everything lives
| What | File |
|---|---|
| Build-ready implementation spec (the source of truth) | `MOMENTUM_STRATEGY_SPEC.md` |
| Narrative writeup (with embedded charts) | `momentum.html` |
| Full experiment log / every result | `trend-channel-experiment/RESULTS.md` |
| All scripts | `trend-channel-experiment/*.py` |
| Data (survivorship-free, **local-only, gitignored, ~276 MB**) | `data/eodhd/` |
| Python interpreter with pandas | `heirarchical-adaptive-filter-experiment/bin/python3` |
| Persistent memory | `~/.claude/projects/-Users-lance-Projects-share-jupyter-experiments/memory/` |

Key spec sections: §0 recommended build · §5 signals · §6 vol-targeting · §7.4 fractional/NAV ·
§8 costs (stamp duty) · §16 tax wrapper · §17 risk posture & operating discipline · §18 Trading 212.

## Decisions already made (don't re-litigate)
- **Broker:** Trading 212 ISA — fractional + commission-free + REST API; confirmed API works in the
  ISA, fractional shares are ISA-eligible, limit orders by fractional quantity. Fund by **bank
  transfer** (free; avoids the 0.7% card deposit fee).
- **Unlevered.** Leverage adds return but not Sharpe and introduces margin-call ruin risk (§17.1).
- **Third factor is low-vol, not value** (currency-neutral, fully covered, higher Sharpe) (§5.5).
- **12-1, top-350, top-quintile** all sweep-confirmed; vol-*change* tested and rejected.
- **Fund via regular contributions** (accumulation discipline; not a return-enhancer) (§17.3).
- **Drawdowns:** pre-commit, hold, size for a −60% worse-than-backtest fall; reserve released only
  on a *market-driven* (beta) drawdown, frozen on an alpha one (§17.2, §17.4).

---

## When the T212 account is authorized — the pickup checklist

1. **Generate the DEMO API key** in the Trading 212 app; export it (direnv, as with EODHD):
   `export T212_API_KEY=...`  (optionally `export T212_HOST=https://demo.trading212.com`).
2. **Discover the real fractional universe** — run:
   `heirarchical-adaptive-filter-experiment/bin/python3 -u trend-channel-experiment/t212_instruments.py`
   It fetches `/equity/metadata/instruments` and flags fractional names via `minTradeQuantity`
   (<1 = fractional; the *value* is the granularity — a `0.1` min on a £345 share is still coarse).
   If it Cloudflare-403s despite the browser User-Agent, save the JSON from the browser docs
   explorer and re-run with `--from-file <file>` (both paths already tested).
3. **Reconcile the fractional set against our held names by ISIN.** This replaces the pessimistic
   "top-100 only" assumption in `wholeshare_10k.py`. Output: how many of the strategy's actual
   held names are fractional, and at what granularity → finalises the £10k configuration
   (fractional-filtered universe, and/or a concentrated ~20–30 name book — see §18.3, §5.5).
   (Need ISINs for our universe — check `data/eodhd/eodhd_uk_meta.csv` / the fundamentals JSON;
   T212 gives ISIN per instrument.)
4. **Build the rest of the T212 integration** (only `t212_instruments.py` exists so far): target
   weights → fractional-quantity **limit** orders in the ISA; read-back of balances/positions for
   reconciliation. All read-write testing on the DEMO account first.
5. **Demo go/no-go gate (§18.4):** run one full demo rebalance end-to-end and reconcile modelled vs
   actual holdings. A clean reconciliation is the gate before any real capital.

## Open items still to decide / build (not yet done)
- **Live data feed for ongoing signals.** `data/eodhd/` is a historical *snapshot* for the backtest.
  The live system needs *fresh* daily prices + point-in-time fundamentals each month to compute
  signals. Decide the recurring source (re-subscribe EODHD around rebalance dates? another feed?).
  This is the main un-built piece of the live pipeline besides the T212 order layer.
- **Initial capital & holding count.** £10k assumed for the tradeability test → points to a
  concentrated ~20–30 name book. Confirm actual NAV and pick the holding count (69 = max Sharpe;
  20–30 = more return, deeper drawdown, better small-NAV fit).
- **Contribution schedule** (fixed vs inflation-linked; amount) — see §17.3.
- **Signal-generation cadence/automation** — the monthly job that pulls data, computes weights, and
  calls the T212 API.

## Standing reminders
- **Back up `data/eodhd/`** — local-only, gitignored, not re-fetchable if the EODHD subscription lapses.
- Everything is committed on `main`; latest relevant commit `1fb2f83` (T212 instrument skeleton).
