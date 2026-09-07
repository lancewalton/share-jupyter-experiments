# candle-data — S&P 500 5-minute candles

Five-minute OHLCV bars for the S&P 500, from Yahoo Finance.

Yahoo serves at most **~60 calendar days** of 5-minute data, so this is a
**wide cross-section over a short window** (~502 names × up to 4,680 bars),
not a long history. Good for cross-sectional / pattern work; not for
multi-regime backtests.

## Layout

- `download_sp500_5m.py` — downloader. Re-runnable; merges new bars into
  existing files (dedupes on timestamp). Yahoo's 60-day window slides
  forward, so running this on a schedule accumulates history **beyond**
  60 days over time.
- `data/<TICKER>.parquet` — one file per ticker; UTC-aware datetime index,
  columns `open, high, low, close, adj_close, volume`.
- `data/_combined.parquet` — tidy long form (`ticker` column added).
- `manifest.csv` — per-ticker bar count and date span from the last run.
- `load.py` — loader helpers.

## Refreshing

```bash
PY=../heirarchical-adaptive-filter-experiment/bin/python3
$PY download_sp500_5m.py --combine            # full universe
$PY download_sp500_5m.py --limit 20           # smoke test
```

To grow history past 60 days, run it regularly (cron / a daily loop);
each run appends the newest bars and keeps everything already saved.

## Loading (from the repo root)

```python
from candle_data.load import load_panel, load_ticker, load_long, universe

close = load_panel("close")          # wide: rows=datetime, cols=ticker
aapl  = load_ticker("AAPL")          # single-name OHLCV
long  = load_long()                  # tidy long form

# UTC by default; for session-local wall clock:
aapl.tz_convert("America/New_York")
```

(`candle_data` import name assumes a `candle_data` package alias or adding
this dir to `sys.path`; adjust to however the other experiments import.)

## Notes / caveats

- Timestamps are **UTC-aware**. Regular session is 13:30–19:55 UTC
  (09:30–15:55 New York), 78 bars/day.
- ~74% of names have the full 4,680 bars; the rest miss a few illiquid
  bars. A handful are genuinely thin (recent listings, low volume).
- `adj_close` here is Yahoo's; splits/dividends over a 60-day window are
  rare, so `close` and `adj_close` are usually identical.
- Yahoo is an unofficial feed. The downloader batches (50/call, 1.5s
  between batches) to stay under rate limits — don't crank it much higher.
- Universe is the **current** S&P 500 constituents (survivorship-biased);
  fine for a rolling 60-day window.
