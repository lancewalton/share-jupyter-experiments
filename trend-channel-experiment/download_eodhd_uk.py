"""Download the survivorship-free UK equity universe from EODHD: all GBP/GBX common
stocks on LSE, active AND delisted, FULL OHLCV history (open/high/low/close/
adjusted_close/volume). One EOD call per symbol returns all fields, so we keep them
all — the dataset is then reusable for any experiment, not just momentum.

Saved long-format (one row per symbol-day) to a gitignored parquet, plus a metadata
CSV. Reads EODHD_API_TOKEN from the environment. Paced under 1000 req/min.

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u download_eodhd_uk.py
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
DATA = HERE.parent / "data" / "eodhd"          # symbol lists in + data out (gitignored)
TOKEN = os.environ["EODHD_API_TOKEN"]
RATE = 14.0
WORKERS = 8
COLS = ["date", "open", "high", "low", "close", "adjusted_close", "volume"]
OUT_OHLCV = DATA / "eodhd_uk_ohlcv.parquet"
OUT_META = DATA / "eodhd_uk_meta.csv"

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
                out.append((code, x.get("Name", ""), flag))
    return out


def fetch(code: str):
    url = f"https://eodhd.com/api/eod/{quote(code)}.LSE"
    params = {"api_token": TOKEN, "fmt": "json", "period": "d"}
    for attempt in range(4):
        throttle()
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or len(data) < 50:
                return code, None
            df = pd.DataFrame(data)
            if "adjusted_close" not in df.columns:
                return code, None
            df = df[[c for c in COLS if c in df.columns]].copy()
            df["code"] = code
            for c in ("open", "high", "low", "close", "adjusted_close", "volume"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
            return code, df
        except Exception:
            time.sleep(1 + attempt)
    return code, None


def main() -> None:
    uni = universe()
    meta = {c: (name, flag) for c, name, flag in uni}
    print(f"Universe: {len(uni)} GBP/GBX common stocks (active+delisted). "
          f"Downloading full OHLCV...", flush=True)
    frames, fails, done = [], [], 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(fetch, c, ) for c, _, _ in uni]
        for f in cf.as_completed(futs):
            code, df = f.result()
            done += 1
            if df is not None:
                frames.append(df)
            else:
                fails.append(code)
            if done % 300 == 0:
                print(f"  {done}/{len(uni)} done, {len(frames)} with data", flush=True)

    big = pd.concat(frames, ignore_index=True)
    big["date"] = pd.to_datetime(big["date"])
    big["status"] = big["code"].map(lambda c: meta.get(c, ("", ""))[1]).astype("category")
    big["code"] = big["code"].astype("category")
    big = big.sort_values(["code", "date"]).reset_index(drop=True)
    big.to_parquet(OUT_OHLCV)

    m = big.groupby("code", observed=True).agg(n_obs=("date", "size"),
                                               first=("date", "min"), last=("date", "max"))
    m["name"] = [meta.get(c, ("", ""))[0] for c in m.index]
    m["status"] = [meta.get(c, ("", ""))[1] for c in m.index]
    m.to_csv(OUT_META)

    print(f"\nSaved {OUT_OHLCV.name}: {len(big):,} rows, {big['code'].nunique()} names, "
          f"{big['date'].min().date()} -> {big['date'].max().date()}")
    print(f"  active with data: {(m.status=='active').sum()}  "
          f"delisted with data: {(m.status=='delisted').sum()}  failed/empty: {len(fails)}")
    print(f"  median history per name: {m.n_obs.median():.0f} rows; file size "
          f"{OUT_OHLCV.stat().st_size/1e6:.0f} MB")


if __name__ == "__main__":
    main()
