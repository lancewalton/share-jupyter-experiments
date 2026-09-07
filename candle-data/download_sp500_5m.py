#!/usr/bin/env python3
"""Download 5-minute candle data for the S&P 500 from Yahoo Finance.

Yahoo serves at most ~60 calendar days of 5-minute bars, so this pulls a
wide cross-section rather than a long history. It writes one parquet file
per ticker under ``data/`` in tidy OHLCV form and is safe to re-run: each
run merges freshly fetched bars into the existing file, deduplicating on
timestamp. Because Yahoo's 60-day window slides forward, running this
regularly (e.g. daily) accumulates history *beyond* 60 days over time.

Usage
-----
    python download_sp500_5m.py                # full S&P 500, default settings
    python download_sp500_5m.py --limit 20     # first 20 names (a smoke test)
    python download_sp500_5m.py --combine      # also write combined parquet

Columns per file: datetime (UTC-aware index), open, high, low, close,
adj_close, volume.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import io

import pandas as pd
import requests
import yfinance as yf

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MANIFEST = HERE / "manifest.csv"
WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_tickers() -> list[str]:
    """Current S&P 500 constituents, as Yahoo-style symbols (dots -> dashes)."""
    # Wikipedia 403s pandas' default urllib UA, so fetch with a browser UA.
    resp = requests.get(WIKI_SP500, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    symbols = tables[0]["Symbol"].astype(str).str.strip()
    # Yahoo uses '-' where the index uses '.', e.g. BRK.B -> BRK-B.
    return sorted(s.replace(".", "-") for s in symbols)


def batched(seq: list[str], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def normalise(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Extract a single ticker's OHLCV frame from a yf.download result."""
    try:
        sub = raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw
    except KeyError:
        return None
    sub = sub.dropna(how="all")
    if sub.empty:
        return None
    sub = sub.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in sub.columns]
    sub = sub[keep]
    sub.index.name = "datetime"
    return sub


def merge_to_disk(ticker: str, fresh: pd.DataFrame) -> tuple[int, pd.Timestamp, pd.Timestamp]:
    """Merge fresh bars into the ticker's parquet, deduping on timestamp."""
    path = DATA_DIR / f"{ticker}.parquet"
    # Store timestamps in UTC so appended windows line up regardless of DST.
    fresh = fresh.copy()
    fresh.index = pd.to_datetime(fresh.index, utc=True)
    if path.exists():
        old = pd.read_parquet(path)
        old.index = pd.to_datetime(old.index, utc=True)
        combined = pd.concat([old, fresh])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = fresh.sort_index()
    combined.to_parquet(path)
    return len(combined), combined.index.min(), combined.index.max()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--period", default="60d", help="Yahoo lookback; 60d is the 5m ceiling")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=1.5, help="seconds between batches")
    ap.add_argument("--limit", type=int, default=None, help="only first N tickers (smoke test)")
    ap.add_argument("--combine", action="store_true", help="also write data/_combined.parquet (long form)")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching S&P 500 constituents...", flush=True)
    tickers = get_sp500_tickers()
    if args.limit:
        tickers = tickers[: args.limit]
    print(f"{len(tickers)} tickers; interval={args.interval} period={args.period}", flush=True)

    rows = []
    empties: list[str] = []
    t_start = time.time()
    for bi, batch in enumerate(batched(tickers, args.batch_size), 1):
        raw = yf.download(
            batch,
            period=args.period,
            interval=args.interval,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        got = 0
        for t in batch:
            sub = normalise(raw, t)
            if sub is None:
                empties.append(t)
                continue
            n, lo, hi = merge_to_disk(t, sub)
            rows.append({"ticker": t, "bars": n, "start": lo, "end": hi})
            got += 1
        print(
            f"batch {bi}: {got}/{len(batch)} ok "
            f"({time.time() - t_start:.0f}s elapsed)",
            flush=True,
        )
        if args.sleep and bi * args.batch_size < len(tickers):
            time.sleep(args.sleep)

    manifest = pd.DataFrame(rows).sort_values("ticker")
    manifest.to_csv(MANIFEST, index=False)
    print(f"\nWrote {len(manifest)} tickers to {DATA_DIR}")
    print(f"Manifest: {MANIFEST}")
    if empties:
        print(f"No data for {len(empties)}: {', '.join(empties)}")

    if args.combine and rows:
        print("Building combined long-form parquet...", flush=True)
        frames = []
        for t in manifest["ticker"]:
            df = pd.read_parquet(DATA_DIR / f"{t}.parquet")
            df.insert(0, "ticker", t)
            frames.append(df.reset_index())
        combined = pd.concat(frames, ignore_index=True)
        out = DATA_DIR / "_combined.parquet"
        combined.to_parquet(out, index=False)
        print(f"Combined: {out}  ({len(combined):,} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
