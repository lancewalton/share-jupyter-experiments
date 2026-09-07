"""Reality check, done right: UK low-vol core TOTAL return (local prices + real
reinvested dividends) vs true-total-return global trackers, GBP, same window.

UK side  : local close + Yahoo dividend events -> per-name TR index -> rebuild core.
Global   : SWDA.L (accumulating MSCI World, GBP -> price IS total return)
           S&P 500 TR (^SP500TR) / GBPUSD  -> US-growth in GBP.
Common clean window 2010-2026 (SWDA history). Core net of 10 bps turnover.
"""
from __future__ import annotations

import csv, glob, json, os, pickle, time, urllib.request, datetime as dt
import numpy as np, pandas as pd
from mc.forecasters import _ewma_vol_series

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
SC = "/private/tmp/claude-501/-Users-lance-Projects-share-jupyter-experiments-monte-carlo-experiment/ecdd1cf6-87c2-423e-ad16-60786d5aca36/scratchpad"
REMAP = {"BT.A": "BT-A.L", "PNN": "PNN.L", "ADML": "ADM.L", "BAES": "BA.L",
         "BDEV": "BTRW.L", "BKGH": "BKG.L", "BRBY": "BRBY.L", "ABDN": "ABDN.L", "SHEL": "SHEL.L"}
H, QUINT, ANN, COST, V0 = 20, 0.2, 252, 0.001, 10_000.0
P1 = int(dt.datetime(2000, 1, 1).timestamp()); P2 = int(dt.datetime(2026, 8, 21).timestamp())


def yfetch(sym, events=False, interval="1d"):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1={P1}&period2={P2}&interval={interval}" + ("&events=div" if events else ""))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=30))["chart"]["result"][0]


def load_local_close():
    out = {}
    for pat in DIRS:
        for p in sorted(glob.glob(pat)):
            recs = []
            with open(p, encoding="utf-8-sig") as f:
                rd = csv.reader(f); next(rd, None)
                for r in rd:
                    if len(r) < 5: continue
                    try: recs.append((pd.to_datetime(r[0], dayfirst=True), float(r[1].replace(",", ""))))
                    except Exception: pass
            s = pd.Series(dict(recs)).sort_index()
            s = s[~s.index.duplicated(keep="last")]
            if len(s) < 1500 or (s <= 0).any() or np.abs(np.diff(np.log(s.values))).max() > 0.6:
                continue
            out[os.path.splitext(os.path.basename(p))[0]] = s
    return out


def get_divs():
    cache = os.path.join(SC, "tr_div_cache.pkl")
    close = load_local_close()
    divs = pickle.load(open(cache, "rb")) if os.path.exists(cache) else {}   # resume
    todo = [tk for tk in close if tk not in divs]
    for i, tk in enumerate(todo):
        got = None
        for attempt in range(3):
            try:
                r = yfetch(REMAP.get(tk, tk + ".L"), events=True, interval="1d")  # 1mo omits divs
                d = r.get("events", {}).get("dividends", {})
                got = pd.Series({pd.to_datetime(dt.date.fromtimestamp(v["timestamp"])): v["amount"]
                                 for v in d.values()}) if d else pd.Series(dtype=float)
                break
            except Exception:
                time.sleep(1.0 * (attempt + 1))
        divs[tk] = got if got is not None else pd.Series(dtype=float)
        if (i + 1) % 5 == 0:                              # incremental save -> resumable
            pickle.dump(divs, open(cache, "wb"))
        time.sleep(0.25)
    pickle.dump(divs, open(cache, "wb"))
    miss = [tk for tk in close if len(divs.get(tk, [])) == 0]
    if miss:
        print(f"  {len(miss)} names with no dividends found: {miss[:12]}{'...' if len(miss)>12 else ''}")
    return {tk: divs[tk] for tk in close}


def get_bench():
    cache = os.path.join(SC, "tr_bench_cache.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    def series(sym):
        r = yfetch(sym); c = r["indicators"]["quote"][0]["close"]
        idx = pd.to_datetime([dt.date.fromtimestamp(t) for t in r["timestamp"]])
        return pd.Series(c, index=idx).dropna()
    b = {"SWDA": series("SWDA.L"), "SPXTR": series("^SP500TR"), "GBPUSD": series("GBPUSD=X")}
    pickle.dump(b, open(cache, "wb"))
    return b


def tr_index(close, divs):
    """per-name total-return price index from close + reinvested dividends."""
    frame = {}
    for tk, C in close.items():
        d = divs.get(tk, pd.Series(dtype=float))
        D = d.reindex(C.index, method="nearest", tolerance=pd.Timedelta("3D")).fillna(0) if len(d) else pd.Series(0.0, index=C.index)
        r = (C + D) / C.shift(1) - 1
        frame[tk] = 100 * (1 + r.fillna(0)).cumprod()
    return pd.DataFrame(frame).sort_index()


def core_tr(TR):
    dates = TR.index; R = np.log(TR).diff()
    vol = R.apply(lambda c: pd.Series(np.sqrt(_ewma_vol_series(c.fillna(0).to_numpy(), lam=0.94, burn=60)), index=dates))
    rebal = list(dates[260::H]); W = pd.DataFrame(0.0, index=dates, columns=TR.columns)
    cost = pd.Series(0.0, index=dates); prevw = None
    for i, d in enumerate(rebal):
        s = vol.loc[d].dropna(); s = s[s.index.isin(R.loc[d].dropna().index)]
        if len(s) < 30: continue
        k = max(1, int(QUINT * len(s))); names = s.nsmallest(k).index
        w = pd.Series(0.0, index=s.index); w[names] = 1.0 / k
        end = rebal[i + 1] if i + 1 < len(rebal) else dates[-1]
        W.loc[(dates > d) & (dates <= end), w.index] = w.values
        if prevw is not None:
            u = prevw.index.union(w.index)
            cost.loc[d] = COST * (w.reindex(u).fillna(0) - prevw.reindex(u).fillna(0)).abs().sum()
        prevw = w
    return (R.fillna(0) * W).sum(axis=1) - cost


def stats(daily, dates):
    x = daily.reindex(dates).fillna(0).to_numpy()
    W = np.exp(np.cumsum(x)); dd = float(((W - np.maximum.accumulate(W)) / np.maximum.accumulate(W)).min())
    yrs = (dates[-1] - dates[0]).days / 365.25
    return W[-1] ** (1 / yrs) - 1, x.std() * np.sqrt(ANN), (x.mean()/x.std()*np.sqrt(ANN) if x.std()>0 else 0), dd, V0 * W[-1]


def main():
    close = load_local_close(); divs = get_divs(); bench = get_bench()
    TR = tr_index(close, divs)
    print(f"UK universe {TR.shape[1]} names, total return (prices + reinvested dividends)")
    npaid = sum(1 for t in divs if len(divs[t])); print(f"  {npaid} names have dividend history\n")

    core = core_tr(TR)
    ew = np.log(TR).diff().mean(axis=1)
    swda = np.log(bench["SWDA"]).diff().reindex(TR.index)               # MSCI World GBP (TR)
    spx_gbp = (bench["SPXTR"] / bench["GBPUSD"].reindex(bench["SPXTR"].index).ffill()).dropna()
    spx = np.log(spx_gbp).diff().reindex(TR.index)                       # S&P500 TR in GBP

    start = max(pd.Timestamp(2010, 1, 4), bench["SWDA"].index[0])
    dates = TR.index[(TR.index >= start) & (TR.index <= min(TR.index[-1], bench["SWDA"].index[-1]))]
    print(f"window {dates[0].date()} -> {dates[-1].date()}  ({(dates[-1]-dates[0]).days/365.25:.0f}y, GBP total return)\n")
    print(f"{'':<28}{'CAGR':>7}{'vol':>7}{'Sharpe':>8}{'maxDD':>8}{'£10k ->':>11}")
    for name, s in [("UK low-vol core (net)", core), ("UK equal-weight (gross)", ew),
                    ("MSCI World, GBP", swda), ("S&P 500 TR, GBP", spx)]:
        cagr, vol, sh, dd, fv = stats(s, dates)
        print(f"{name:<28}{cagr*100:>+6.1f}%{vol*100:>6.1f}%{sh:>+8.2f}{dd*100:>+7.0f}%   £{fv:>8,.0f}")

    print("\ncalendar-year total return:")
    cs = pd.Series(core.values, index=TR.index); ws = pd.Series(swda.values, index=TR.index)
    for y in range(2016, 2027):
        m = (TR.index >= pd.Timestamp(y, 1, 1)) & (TR.index < pd.Timestamp(y + 1, 1, 1))
        if m.sum() < 20: continue
        print(f"  {y}   core {(np.exp(cs[m].sum())-1)*100:+6.1f}%     MSCI World GBP {(np.exp(ws[m].fillna(0).sum())-1)*100:+6.1f}%")


if __name__ == "__main__":
    main()
