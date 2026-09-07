# Share market prediction experiments

A quantitative research programme testing a single question: **can markets be
predicted well enough to trade?** Thirteen experiments attack it from different
angles — price/direction forecasting, volatility forecasting, cross-asset
rotation, sentiment, FX, and specific trading strategies — under one discipline:
*assume any good result is a bug until it survives walk-forward evaluation,
held-out data, permutation nulls, realistic costs, and every attempt to break
it.*

## Top-line conclusion

- **Direction is a mirage.** Every route to forecasting price direction hit a
  ~1% information-coefficient ceiling; the signals that were statistically real
  did not survive trading costs. Every strategy we traded *on a direction view*
  was beaten outright by buy-and-hold.
- **Volatility is genuinely forecastable** — but its direct value is risk
  management (calibrated tails, vol-targeted sizing, VaR), not alpha.
- **The only edges that survived are risk premia** — the variance premium and FX
  carry (both ~0.36 net Sharpe) — the *same* sell-insurance trade in two markets:
  a thin steady income, a violent loss in the crash, and untimeable for free.
- **Buy-and-hold is the benchmark nothing cleared on raw return.** One book (a
  defensive low-vol + momentum + vol-targeting stack) beat its market on a
  *risk-adjusted* basis (Sharpe 0.59 vs 0.22), but not on terminal wealth, and
  even that edge decayed across the sample.
- **Filtering a direction signal only steers *hit-rate*, never *payoff*.** Later
  studies fed the trend-line breakout extra information — line geometry, volume,
  breakout extent, relative line age, and the volatility regime — and every filter
  (and every combination) raised how *often* a breakout wins, never how *much*.
  Strong breakouts revert rather than continue; the method stays net-negative.
- **Trend-following adds value against sustained trends, destroys it against
  whipsaws.** A rising-channel strategy loses to buy-and-hold everywhere — as a
  *timer* (dip-buy / top-sell clips the right tail of big compounders; even
  ride-the-winner and capital rotation can't fix it) and as a *selector* (holding
  names in qualifying channels is just momentum — beats B&H pre-2013, loses after).
  The tell is the edge's correlation with volatility: **negative for equities**
  (V-shaped crashes whipsaw the trend exit) but **positive for bonds** (the
  sustained 2022 sell-off rewards it). So the driver is drawdown *shape*, not a
  2012 regime change or vol suppression — and the pattern holds across UK equities,
  US equities, and bonds.

The full write-ups are the `synthesis-*.html` / `strategy.html` documents at the
repo root (and `momentum.html` — the programme's one positive, survivorship-tested
edge), plus a `WRITEUP.md` / `RESULTS.md` inside most experiment folders.
`BACKLOG.md` and `STRATEGY.md` are the cross-project idea log and operational
spec.

## Experiments

| Folder | Axis / question |
| --- | --- |
| `monte-carlo-experiment` | Forward price envelopes (FHS calibration) + a mean-reversion direction test |
| `pattern-discovery` | Do data-discovered chart shapes predict returns? |
| `pattern-matching` | A non-linear (autoencoder) shape decomposition; a detection reframe |
| `heirarchical-adaptive-filter-experiment` | Self-tuning volatility forecasts vs a fixed expert stack |
| `vol-risk-experiment` | Sizing by risk, not by view — the defensive stack |
| `variance-premium-experiment` | Monetising the vol edge via the variance risk premium |
| `factor-rotation-experiment` | Cross-asset rotation: what reverts vs what trends |
| `sentiment-experiment` | Crowd fear as a contrarian signal |
| `fx-rotation-experiment` | FX value / momentum / carry |
| `swing-trading-trend-lines-experiment` | A support/resistance trend-line breakout method |
| `quick-flip-scalper-experiment` | An intraday opening-range strategy (fade, then follow) |
| `trend-channel-experiment` | Rising-channel trading vs buy-and-hold — timing vs selection, and the drawdown-shape mechanism across UK/US equities and bonds |

### Follow-up studies

Later work re-tested the two trading strategies with extra information, all as
cost-aware, stress-tested post-processing over the cached trades:

- **Volatility-regime gating** — use the forecastable vol regime to steer
  fade-vs-follow. `quick-flip-scalper-experiment/regime_and_costs.py` (also a
  futures-cost re-run) and `swing-trading-trend-lines-experiment/regime_gate_ftse.py`.
  Intraday it nudges the breakout to beta-neutral-positive but unproven; on daily
  FTSE it fails and the regime steers the *wrong* way.
- **The fade leg** — `swing-trading-trend-lines-experiment/fade_leg_ftse.py`: a
  daily mean-reversion entry gated to compression. The one real, significant,
  beta-neutral *gross* edge in the daily work — but dealing costs erase the net.
- **Line-duration & breakout anatomy** — `line_duration_study.py`,
  `duration_regime_cross.py`, `continuation_study.py`: relative action/safety line
  duration, breakout extent, volume, and their combinations each predict *win
  probability* but not *payoff*; strong breakouts revert rather than continue.

Conclusions live in each experiment's `RESULTS.md`, with raw output in the
`*_results.txt` files.

## Data

Market data (Yahoo 5-minute candles, cached `*.parquet` / `*.csv`) is **not**
committed — it is regenerable and large. Each experiment that needs data ships
its own download/loader scripts (e.g. `candle-data/download_sp500_5m.py` and
`candle-data/load.py`). Run those to repopulate the ignored `data/` directories.

**FTSE daily data.** The daily-bar experiments (notably
`swing-trading-trend-lines-experiment`) load ~120 FTSE stocks of daily OHLCV
from a **separate private repository** (`lancewalton/shares`, expected at
`~/Projects/shares/data/yfinance/`), not from this repo. That data is not
public, so these experiments will not run as-is for anyone without access to
that repository; substitute your own daily OHLCV source and point the loaders
at it to reproduce them.

## Environment

Experiments are Python (pandas / numpy / matplotlib / pyarrow / yfinance) in
Jupyter notebooks and plain scripts. The virtual environment is not committed;
recreate one and install the usual scientific-Python stack.

---

*A validated research programme, honestly caveated — not investment advice.*
