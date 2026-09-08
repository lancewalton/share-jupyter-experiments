"""Download EODHD fundamentals for the survivorship-free UK universe (same active +
delisted GBP/GBX common stocks). Saves raw fundamentals JSON per company (gzipped,
gitignored) so nothing is lost for future factor experiments, plus a curated
snapshot parquet (highlights/valuation/general) for immediate use.

Fundamentals is a separate endpoint (10 API units each). Reads EODHD_API_TOKEN.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u download_eodhd_fundamentals.py
"""
from __future__ import annotations

import concurrent.futures as cf
import gzip
import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "eodhd"
TOKEN = os.environ["EODHD_API_TOKEN"]
RATE = 12.0
WORKERS = 8
RAW_DIR = DATA / "eodhd_fundamentals"
OUT = DATA / "eodhd_uk_fundamentals.parquet"

_lock = threading.Lock()
_next = [time.monotonic()]


def throttle() -> None:
    with _lock:
        now = time.monotonic()
        wait = max(0.0, _next[0] - now)
        _next[0] = max(now, _next[0]) + 1.0 / RATE
    if wait > 0:
        time.sleep(wait)


def universe():
    act = json.load(open(DATA / "lse_active.json"))
    de = json.load(open(DATA / "lse_delisted.json"))
    out, seen = [], set()
    for d, flag in ((act, "active"), (de, "delisted")):
        for x in d:
            code = x.get("Code")
            if (x.get("Type") == "Common Stock" and x.get("Currency") in ("GBP", "GBX")
                    and code and code not in seen):
                seen.add(code)
                out.append((code, flag))
    return out


def curate(code: str, flag: str, f: dict) -> dict:
    g = f.get("General", {}) or {}
    h = f.get("Highlights", {}) or {}
    v = f.get("Valuation", {}) or {}
    row = {"code": code, "status": flag, "name": g.get("Name"), "sector": g.get("Sector"),
           "industry": g.get("Industry"), "isin": g.get("ISIN"), "currency": g.get("CurrencyCode"),
           "ipo": g.get("IPODate"), "delisted_flag": g.get("IsDelisted")}
    for k in ("MarketCapitalization", "PERatio", "PEGRatio", "BookValue", "DividendYield",
              "EPSEstimateCurrentYear", "ProfitMargin", "OperatingMarginTTM", "ReturnOnAssetsTTM",
              "ReturnOnEquityTTM", "RevenueTTM", "GrossProfitTTM", "EarningsShare"):
        row[k] = h.get(k)
    for k in ("TrailingPE", "ForwardPE", "PriceSalesTTM", "PriceBookMRQ", "EnterpriseValue",
              "EnterpriseValueEbitda"):
        row["Val_" + k] = v.get(k)
    return row


def fetch(code: str, flag: str):
    url = f"https://eodhd.com/api/fundamentals/{quote(code)}.LSE"
    params = {"api_token": TOKEN, "fmt": "json"}
    for attempt in range(4):
        throttle()
        try:
            r = requests.get(url, params=params, timeout=45)
            if r.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            if r.status_code == 404:
                return code, None
            r.raise_for_status()
            f = r.json()
            if not isinstance(f, dict) or not f:
                return code, None
            with gzip.open(RAW_DIR / f"{code}.json.gz", "wt", encoding="utf-8") as fh:
                json.dump(f, fh)
            return code, curate(code, flag, f)
        except Exception:
            time.sleep(1 + attempt)
    return code, None


def main() -> None:
    RAW_DIR.mkdir(exist_ok=True)
    uni = universe()
    print(f"Fundamentals for {len(uni)} names (10 units each ~ {10*len(uni)} units)...", flush=True)
    rows, fails, done = [], 0, 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(fetch, c, flag) for c, flag in uni]
        for fut in cf.as_completed(futs):
            code, row = fut.result()
            done += 1
            if row is not None:
                rows.append(row)
            else:
                fails += 1
            if done % 300 == 0:
                print(f"  {done}/{len(uni)} done, {len(rows)} with fundamentals", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT)
    print(f"\nSaved {OUT.name}: {len(df)} companies with fundamentals; raw JSON in {RAW_DIR.name}/")
    print(f"  with market cap: {df['MarketCapitalization'].notna().sum()}  "
          f"empty/failed: {fails}")


if __name__ == "__main__":
    main()
