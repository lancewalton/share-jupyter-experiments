"""Parse the raw EODHD fundamentals JSON into a POINT-IN-TIME panel for factor work.

For each company, extract dated annual Balance Sheet + Income Statement fields, keyed
by the date the data was actually AVAILABLE (filing_date, or fiscal-end + 120 days if
missing) so a backtest never uses a figure before it was reported. Fields: book equity,
total assets, net income, gross profit, shares, statement & quote currency.

Output: data/eodhd/eodhd_uk_fundamentals_panel.parquet (gitignored).

    ../heirarchical-adaptive-filter-experiment/bin/python3 -u parse_fundamentals.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "eodhd"
RAW = DATA / "eodhd_fundamentals"
OUT = DATA / "eodhd_uk_fundamentals_panel.parquet"


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def shares_asof(shares_hist: dict, fiscal_date: str):
    """Latest reported shares outstanding on/before the fiscal date."""
    best = None
    for v in (shares_hist or {}).values():
        d = v.get("date") or v.get("dateFormatted")
        s = num(v.get("shares"))
        if d and s and d <= fiscal_date:
            if best is None or d > best[0]:
                best = (d, s)
    return best[1] if best else None


def parse_one(code: str, j: dict) -> list[dict]:
    quote_ccy = (j.get("General") or {}).get("CurrencyCode")
    fin = j.get("Financials") or {}
    bs = (fin.get("Balance_Sheet") or {}).get("yearly") or {}
    inc = (fin.get("Income_Statement") or {}).get("yearly") or {}
    shares_hist = (j.get("outstandingShares") or {}).get("annual") or {}
    rows = []
    for fdate, b in bs.items():
        i = inc.get(fdate, {})
        filing = b.get("filing_date") or i.get("filing_date")
        avail = filing if filing else str((pd.Timestamp(fdate) + pd.Timedelta(days=120)).date())
        rows.append({
            "code": code, "fiscal_date": fdate, "avail_date": avail,
            "book_equity": num(b.get("totalStockholderEquity")),
            "total_assets": num(b.get("totalAssets")),
            "net_income": num(i.get("netIncome")),
            "gross_profit": num(i.get("grossProfit")),
            "shares": shares_asof(shares_hist, fdate),
            "stmt_ccy": b.get("currency_symbol") or i.get("currency_symbol"),
            "quote_ccy": quote_ccy,
        })
    return rows


def main() -> None:
    files = sorted(RAW.glob("*.json.gz"))
    print(f"Parsing {len(files)} fundamentals files...", flush=True)
    all_rows, done = [], 0
    for f in files:
        try:
            j = json.load(gzip.open(f, "rt", encoding="utf-8"))
            all_rows.extend(parse_one(f.name.replace(".json.gz", ""), j))
        except Exception:
            pass
        done += 1
        if done % 800 == 0:
            print(f"  {done}/{len(files)}", flush=True)
    df = pd.DataFrame(all_rows)
    df["fiscal_date"] = pd.to_datetime(df["fiscal_date"], errors="coerce")
    df["avail_date"] = pd.to_datetime(df["avail_date"], errors="coerce")
    df = df.dropna(subset=["fiscal_date", "avail_date"]).sort_values(["code", "avail_date"])
    df.to_parquet(OUT)
    print(f"\nSaved {OUT.name}: {len(df):,} statement-rows, {df['code'].nunique()} names")
    print(f"  with book_equity: {df.book_equity.notna().sum():,}  net_income: {df.net_income.notna().sum():,}  "
          f"gross_profit: {df.gross_profit.notna().sum():,}  shares: {df.shares.notna().sum():,}")
    print(f"  statement currencies: {df.stmt_ccy.value_counts().head(6).to_dict()}")
    print(f"  avail_date range: {df.avail_date.min().date()} -> {df.avail_date.max().date()}")
    # sanity: one large-cap's book-equity history
    for c in ("AZN", "SHEL", "HSBA"):
        s = df[df.code == c].set_index("fiscal_date")["book_equity"].dropna()
        if len(s):
            print(f"  {c} book equity: {len(s)} yrs, {s.index.min().year}->{s.index.max().year}, "
                  f"latest {s.iloc[-1]:.3g} {df[df.code==c]['stmt_ccy'].iloc[-1]}")
            break


if __name__ == "__main__":
    main()
