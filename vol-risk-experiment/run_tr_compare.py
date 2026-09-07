"""Reality check: our low-vol core TOTAL return vs a global tracker, GBP, same window.

Everything on dividends-reinvested total return (Yahoo adjusted close):
  - UK low-vol core   : rebuild the §3a core on TR (EWMA-vol lowest quintile, monthly)
  - UK equal-weight   : the same universe, held equally (our 'market')
  - S&P 500 TR in GBP : ^SP500TR / GBPUSD  (US-growth proxy a GBP investor would get)
Common GBP window starts 2004 (GBPUSD history limit). Net 10 bps on the core's turnover.
Caches Yahoo pulls to scratchpad so re-runs are instant.
"""
from __future__ import annotations

import json, os, glob, pickle, time, urllib.request, datetime as dt
import numpy as np, pandas as pd
from mc.forecasters import _ewma_vol_series

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
CACHE = "/private/tmp/claude-501/-Users-lance-Projects-share-jupyter-experiments-monte-carlo-experiment/ecdd1cf6-87c2-423e-ad16-60786d5aca36/scratchpad/tr_cache.pkl"
REMAP = {"BT.A": "BT-A.L", "PNN": "PNN.L", "ADML": "ADM.L", "BAES": "BA.L",
         "BDEV": "BTRW.L", "BKGH": "BKG.L", "BRBY": "BRBY.L", "ABDN": "ABDN.L",
         "SHEL": "SHEL.L"}
FROZEN = {"CRH", "FERG", "AHT", "BHPB", "CPG"}
H, QUINT, ANN, COST, V0 = 20, 0.2, 252, 0.001, 10_000.0
P1 = int(dt.datetime(2000, 1, 1).timestamp()); P2 = int(dt.datetime(2026, 8, 21).timestamp())


def fetch_adj(sym, interval="1d"):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1={P1}&period2={P2}&interval={interval}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    r = json.load(urllib.request.urlopen(req, timeout=30))["chart"]["result"][0]
    ts = r["timestamp"]; adj = r["indicators"]["adjclose"][0]["adjclose"]
    idx = [dt.date.fromtimestamp(t) for t in ts]
    s = pd.Series(adj, index=pd.to_datetime(idx)).dropna()
    return s[~s.index.duplicated(keep="last")]


def local_tickers():
    tks = []
    for pat in DIRS:
        for p in sorted(glob.glob(pat)):
            tk = os.path.splitext(os.path.basename(p))[0]
            if tk in FROZEN:
                continue
            tks.append(tk)
    return tks


def build_cache():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    uni = {}
    for tk in local_tickers():
        sym = REMAP.get(tk, tk + ".L")
        try:
            s = fetch_adj(sym)
            if len(s) > 800:
                uni[tk] = s
        except Exception:
            pass
        time.sleep(0.05)
    bench = {}
    for name, sym in [("SPX_TR", "^SP500TR"), ("GBPUSD", "GBPUSD=X")]:
        bench[name] = fetch_adj(sym)
    obj = {"uni": uni, "bench": bench}
    pickle.dump(obj, open(CACHE, "wb"))
    return obj


def core_tr(TR):
    dates = TR.index
    logC = np.log(TR); R = logC.diff()
    vol = R.apply(lambda c: pd.Series(np.sqrt(_ewma_vol_series(c.fillna(0).to_numpy(),
                  lam=0.94, burn=60)), index=dates))
    rebal = list(dates[260::H])
    W = pd.DataFrame(0.0, index=dates, columns=TR.columns); prevw = None
    cost = pd.Series(0.0, index=dates)
    for i, d in enumerate(rebal):
        s = vol.loc[d].dropna()
        act = R.loc[d].dropna().index                       # names trading now
        s = s[s.index.isin(act)]
        if len(s) < 30:
            continue
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
    W = np.exp(np.cumsum(x)); peak = np.maximum.accumulate(W)
    dd = float(((W - peak) / peak).min())
    yrs = (dates[-1] - dates[0]).days / 365.25
    cagr = W[-1] ** (1 / yrs) - 1
    sh = x.mean() / x.std() * np.sqrt(ANN) if x.std() > 0 else 0
    return cagr, x.std() * np.sqrt(ANN), sh, dd, V0 * W[-1]


def main():
    obj = build_cache(); uni, bench = obj["uni"], obj["bench"]
    TR = pd.DataFrame(uni).sort_index().ffill(limit=5)
    print(f"universe {TR.shape[1]} names with total-return history\n")

    core = core_tr(TR)
    ew = np.log(TR).diff().mean(axis=1)                      # UK equal-weight TR
    # S&P 500 TR in GBP
    spx = bench["SPX_TR"].reindex(TR.index).ffill()
    gu = bench["GBPUSD"].reindex(TR.index).ffill()
    spx_gbp = (spx / gu).dropna()
    spx_ret = np.log(spx_gbp).diff().reindex(TR.index)

    start = pd.Timestamp(2004, 1, 1)
    dates = TR.index[(TR.index >= start) & (TR.index >= spx_gbp.index[0])]
    print(f"window {dates[0].date()} -> {dates[-1].date()}  (GBP total return, net where noted)\n")
    print(f"{'strategy':<26}{'CAGR':>7}{'vol':>7}{'Sharpe':>8}{'maxDD':>8}{'£10k ->':>11}")
    for name, series in [("UK low-vol core (net)", core),
                         ("UK equal-weight (gross)", ew),
                         ("S&P 500 TR, in GBP", spx_ret)]:
        cagr, vol, sh, dd, fv = stats(series, dates)
        print(f"{name:<26}{cagr*100:>+6.1f}%{vol*100:>6.1f}%{sh:>+8.2f}{dd*100:>+7.0f}%   £{fv:>8,.0f}")

    # by-year total returns, core vs S&P-GBP
    print("\ncalendar-year total return  (core vs S&P-500-in-GBP):")
    cs = pd.Series(core, index=TR.index); ss = pd.Series(spx_ret.values, index=TR.index)
    for y in range(2015, 2027):
        m = (TR.index >= pd.Timestamp(y, 1, 1)) & (TR.index < pd.Timestamp(y + 1, 1, 1))
        if m.sum() < 20: continue
        print(f"  {y}   core {(np.exp(cs[m].sum())-1)*100:+6.1f}%     S&P-GBP {(np.exp(ss[m].fillna(0).sum())-1)*100:+6.1f}%")


if __name__ == "__main__":
    main()
