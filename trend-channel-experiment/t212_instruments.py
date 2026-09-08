"""Trading 212 — fetch the tradeable-instrument list and its fractional-share flags.

Answers the open question from MOMENTUM_STRATEGY_SPEC.md §18.3: which LSE names are fractionally
dealable (so we can restrict the eligible universe to them). This is step 1 of the demo-account
validation (§18.4). It does NOT trade — read-only metadata.

API facts (from docs.trading212.com/api, verified 2026-09-08):
  endpoint : GET /api/v0/equity/metadata/instruments
  hosts    : demo  https://demo.trading212.com     live  https://live.trading212.com
  auth     : the API key is sent directly as the `Authorization` header value (no "Bearer"/Basic)
  fractional: derived from `minTradeQuantity` -- < 1 means fractional dealing is allowed; the value
             itself is the granularity (e.g. 0.1 vs 1e-8), which matters for high-priced shares.

Two real gotchas (Trading 212 community + docs):
  * The demo instruments endpoint is often Cloudflare-403'd for scripts with a default user-agent,
    while it works in the browser docs explorer. We send a browser-like User-Agent to avoid that.
    If it still 403s, save the JSON from the docs explorer and pass --from-file instruments.json.
  * Metadata endpoints are heavily rate-limited -- fetch once and cache (this writes a CSV).

Setup: put the DEMO key in the environment (direnv, as with EODHD):  export T212_API_KEY=...
       optionally  export T212_HOST=https://demo.trading212.com   (default; use live host for live key)

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u t212_instruments.py
    ../heirarchical-adaptive-filter-experiment/bin/python3 -u t212_instruments.py --from-file instruments.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_CSV = HERE / "t212_instruments.csv"
ENDPOINT = "/api/v0/equity/metadata/instruments"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def fetch_live() -> list[dict]:
    key = os.environ.get("T212_API_KEY")
    if not key:
        sys.exit("Set T212_API_KEY in the environment (the DEMO key while validating).")
    host = os.environ.get("T212_HOST", "https://demo.trading212.com").rstrip("/")
    req = urllib.request.Request(host + ENDPOINT, headers={
        "Authorization": key,                 # T212: raw key as the Authorization value
        "User-Agent": UA,                     # dodge the Cloudflare 403 that hits default UAs
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        if e.code == 403:
            sys.exit("403 (likely Cloudflare). The demo metadata endpoint often blocks scripts.\n"
                     "Workaround: open the endpoint in the browser docs explorer, save the JSON,\n"
                     f"and re-run with --from-file <that file>.\nBody: {body}")
        if e.code == 401:
            sys.exit("401 unauthorized. Check the key, and that it has the 'metadata' scope enabled.\n"
                     f"Body: {body}")
        if e.code == 429:
            sys.exit(f"429 rate-limited — wait and retry (metadata is heavily throttled).\nBody: {body}")
        sys.exit(f"HTTP {e.code}: {body}")


def summarise(instruments: list[dict]) -> None:
    df = pd.DataFrame(instruments)
    # keep the fields we care about if present; tolerate schema drift
    keep = [c for c in ("ticker", "name", "shortName", "isin", "currencyCode", "type",
                        "minTradeQuantity", "maxOpenQuantity", "workingScheduleId", "addedOn") if c in df.columns]
    df = df[keep].copy()
    if "minTradeQuantity" in df:
        df["minTradeQuantity"] = pd.to_numeric(df["minTradeQuantity"], errors="coerce")
        df["fractional"] = df["minTradeQuantity"] < 1.0
    df.to_csv(OUT_CSV, index=False)

    print(f"instruments returned: {len(df):,}")
    if "type" in df:
        print("  by type:", df["type"].value_counts().head(6).to_dict())
    if "currencyCode" in df:
        lse = df[df["currencyCode"].isin(["GBX", "GBP"])]
        print(f"  GBP/GBX (LSE-ish): {len(lse):,}")
        if "fractional" in df:
            fr = int(lse["fractional"].sum())
            print(f"    fractional: {fr:,} ({100*fr/max(len(lse),1):.0f}%)  whole-share-only: {len(lse)-fr:,}")
            print("    minTradeQuantity distribution (GBP/GBX):",
                  lse["minTradeQuantity"].value_counts().sort_index().head(10).to_dict())
    print(f"\nsaved -> {OUT_CSV.name}  (reconcile to our universe by ISIN/ticker next)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-file", help="parse a saved instruments JSON instead of calling the API")
    args = ap.parse_args()
    if args.from_file:
        instruments = json.loads(Path(args.from_file).read_text())
    else:
        instruments = fetch_live()
    if not isinstance(instruments, list):
        sys.exit(f"expected a JSON array of instruments, got {type(instruments).__name__}")
    summarise(instruments)


if __name__ == "__main__":
    main()
