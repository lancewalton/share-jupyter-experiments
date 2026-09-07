"""How much did 'trade at the observed close' flatter the backtest?

Three execution models for the same monthly stack (low-vol core + 0.5x momentum
+ vol-target), net of 10 bps:
  A  instant close fill   -- weights active from the signal-date close (current)
  B  T+1 close            -- weights active one day later (standard conservative)
  C  T+1 open             -- execute at next open: entry day earns intraday only,
                            overnight gap carried on the OLD weights (realistic)
Uses the Open column we normally ignore. Decomposes r = overnight + intraday:
  overnight_t = log(Open_t/Close_{t-1}),  intraday_t = log(Close_t/Open_t).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from mc.forecasters import _ewma_vol_series

DIRS = ["/Users/lance/Projects/shares/data/yfinance/*.csv",
        "/Users/lance/Projects/shares/data/ukinvesting/*.csv"]
H, QUINT, ANN, COST = 20, 0.2, 252, 0.001
W_MOM, TARGET, MAXLEV, V0 = 0.5, 0.12, 2.5, 100_000.0


def _load_oc():
    close, opn = {}, {}
    for pat in DIRS:
        for path in sorted(glob.glob(pat)):
            try:
                df = pd.read_csv(path, thousands=",", encoding="utf-8-sig")
            except Exception:
                continue
            df.columns = [c.strip().strip('"') for c in df.columns]
            if "Open" not in df.columns or not ({"Price", "Close"} & set(df.columns)):
                continue
            pc = "Close" if "Close" in df.columns else "Price"
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            for c in (pc, "Open"):
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False),
                                      errors="coerce")
            df = df.dropna(subset=["Date", pc, "Open"]).sort_values("Date")
            df = df[~df["Date"].duplicated(keep="last")].set_index("Date")
            if len(df) < 1500 or (df[pc] <= 0).any() or (df["Open"] <= 0).any():
                continue
            tk = os.path.splitext(os.path.basename(path))[0]
            close[tk] = df[pc]; opn[tk] = df["Open"]
    C = pd.DataFrame(close).sort_index(); O = pd.DataFrame(opn).reindex_like(C)
    return C, O


def _book_W(signal, cols, rebal, dates, mode):
    W = pd.DataFrame(0.0, index=dates, columns=cols)
    for i, d in enumerate(rebal):
        if d not in signal.index:
            continue
        s = signal.loc[d].dropna()
        if len(s) < 30:
            continue
        k = max(1, int(QUINT * len(s))); rank = s.rank()
        w = pd.Series(0.0, index=s.index)
        if mode == "ls":
            w[rank > len(s) - k] = 0.5 / k; w[rank <= k] = -0.5 / k
        else:
            w[rank <= k] = 1.0 / k
        end = rebal[i + 1] if i + 1 < len(rebal) else dates[-1]
        W.loc[(dates > d) & (dates <= end), w.index] = w.values
    return W


def _stats(x):
    x = np.nan_to_num(x); ar = x.mean() * ANN; av = x.std() * np.sqrt(ANN)
    g = np.cumsum(x); dd = float((g - np.maximum.accumulate(g)).min())
    per = ANN
    return ar, av, (ar / av if av > 0 else 0.0), dd


def main():
    C, O = _load_oc()
    dates = C.index
    logC = np.log(C); logO = np.log(O)
    R = logC.diff()                                   # close-to-close
    ON = logO - logC.shift(1)                          # overnight gap
    ID = logC - logO                                   # intraday
    print(f"universe {C.shape[1]}; {dates[0].date()} -> {dates[-1].date()}\n")

    mom = R.rolling(231).sum().shift(21)
    vol = R.apply(lambda c: pd.Series(np.sqrt(_ewma_vol_series(c.fillna(0).to_numpy(),
                  lam=0.94, burn=60)), index=dates))
    rebal = list(dates[260::H])
    Wc = _book_W(vol, C.columns, rebal, dates, "low")
    Ws = _book_W(mom, C.columns, rebal, dates, "ls")
    W = Wc + W_MOM * Ws                                # combined stock weights

    Rf = R.fillna(0); ONf = ON.fillna(0); IDf = ID.fillna(0)
    base = (W * Rf).sum(axis=1)                        # for the vol-target sizing
    cvol = np.sqrt(_ewma_vol_series(base.to_numpy(), lam=0.94, burn=60)) * np.sqrt(ANN)
    m = np.nan_to_num(np.clip(TARGET / np.where(cvol > 0, cvol, np.nan), 0, MAXLEV))
    tcost = COST * np.abs(W.diff().fillna(W)).sum(axis=1).to_numpy()   # stock-trade cost
    oc = COST * np.abs(np.diff(m, prepend=m[0]))                        # resize cost

    bookA = m * (W * Rf).sum(axis=1).to_numpy() - tcost - oc
    bookB = m * (W.shift(1) * Rf).sum(axis=1).to_numpy() - tcost - oc
    bookC = m * ((W.shift(1) * ONf).sum(axis=1) + (W * IDf).sum(axis=1)).to_numpy() - tcost - oc

    print(f"{'execution model':<28}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'£100k -> ':>12}")
    for nm, b in [("A  instant close fill", bookA), ("B  T+1 close (conservative)", bookB),
                  ("C  T+1 open (realistic)", bookC)]:
        ar, av, sh, dd = _stats(b)
        yrs = (dates[-1] - dates[260]).days / 365.25
        cagr = np.exp(np.nansum(b)) ** (1 / yrs) - 1
        fv = V0 * np.exp(np.nansum(b))
        print(f"{nm:<28}{cagr*100:>+7.1f}%{sh:>+8.2f}{dd:>+8.2f}   £{fv:>10,.0f}")
    print("\n(difference A->C is the cost of the 'trade at the observed close' assumption)")


if __name__ == "__main__":
    main()
