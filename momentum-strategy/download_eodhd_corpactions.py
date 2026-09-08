"""Download dividends and splits (raw corporate-action events) for the survivorship-
free UK universe from EODHD. Separate endpoints from EOD (1 unit each). Saved
long-format to gitignored parquets. adjusted_close already bakes these in; these are
the raw events, useful for a dividend-yield factor or precise reconstruction.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u download_eodhd_corpactions.py
"""
from __future__ import annotations

import concurrent.futures as cf
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
RATE = 14.0
WORKERS = 8
OUT_DIV = DATA / "eodhd_uk_dividends.parquet"
OUT_SPLIT = DATA / "eodhd_uk_splits.parquet"

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
    for d in (act, de):
        for x in d:
            code = x.get("Code")
            if (x.get("Type") == "Common Stock" and x.get("Currency") in ("GBP", "GBX")
                    and code and code not in seen):
                seen.add(code)
                out.append(code)
    return out


def get(endpoint: str, code: str):
    url = f"https://eodhd.com/api/{endpoint}/{quote(code)}.LSE"
    params = {"api_token": TOKEN, "fmt": "json"}
    for attempt in range(4):
        throttle()
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            if r.status_code == 404:
                return []
            r.raise_for_status()
            d = r.json()
            return d if isinstance(d, list) else []
        except Exception:
            time.sleep(1 + attempt)
    return None


def fetch(code: str):
    divs = get("div", code)
    splits = get("splits", code)
    return code, divs, splits


def main() -> None:
    uni = universe()
    print(f"Dividends + splits for {len(uni)} names (2 calls each)...", flush=True)
    div_rows, split_rows, done = [], [], 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(fetch, c) for c in uni]
        for f in cf.as_completed(futs):
            code, divs, splits = f.result()
            done += 1
            for r in (divs or []):
                r = dict(r); r["code"] = code; div_rows.append(r)
            for r in (splits or []):
                r = dict(r); r["code"] = code; split_rows.append(r)
            if done % 400 == 0:
                print(f"  {done}/{len(uni)} done; {len(div_rows)} div rows, {len(split_rows)} split rows", flush=True)

    dv = pd.DataFrame(div_rows)
    if not dv.empty:
        dv.to_parquet(OUT_DIV)
    sp = pd.DataFrame(split_rows)
    if not sp.empty:
        sp.to_parquet(OUT_SPLIT)
    print(f"\nSaved {OUT_DIV.name} ({len(dv)} dividend events, "
          f"{dv['code'].nunique() if not dv.empty else 0} names)")
    print(f"Saved {OUT_SPLIT.name} ({len(sp)} split events, "
          f"{sp['code'].nunique() if not sp.empty else 0} names)")


if __name__ == "__main__":
    main()
