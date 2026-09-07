# Share market prediction experiments

A quantitative research programme testing a single question: **can markets be
predicted well enough to trade?** Twelve experiments attack it from different
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

The full write-ups are the `synthesis-*.html` / `strategy.html` documents at the
repo root, plus a `WRITEUP.md` / `RESULTS.md` inside most experiment folders.
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
