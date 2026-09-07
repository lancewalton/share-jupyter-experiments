"""Data loading and volatility targets."""
import numpy as np
import pandas as pd

FTSE_PATH = "/Users/lance/Projects/FTSEData/all.csv"


def load_ftse(path=FTSE_PATH):
    """FTSE 100 daily OHLC, 1990-2021, oldest-first. Returns df with log 'ret'."""
    df = pd.read_csv(path, thousands=",")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    df["ret"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna(subset=["ret"]).reset_index(drop=True)


def load_sp500_fred():
    df = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500")
    df.columns = ["Date", "Close"]
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["ret"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna(subset=["ret"]).reset_index(drop=True)


def realized_vol_rolling(r, k):
    """OVERLAPPING trailing realized vol. NB: adjacent points share k-1 obs, which
    inflates one-step-ahead 'skill'. Kept only to demonstrate that artefact."""
    return np.sqrt(pd.Series(np.asarray(r) ** 2).rolling(k, min_periods=1).mean()).to_numpy()


def realized_vol_blocks(r, k):
    """NON-OVERLAPPING realized vol: one point per disjoint block of k days.
    RV_j = sqrt(mean r^2 over block j). Adjacent points share no observations, so
    predicting RV_j from prior blocks carries no overlap artefact."""
    r = np.asarray(r, float)
    n_blocks = len(r) // k
    r2 = r[:n_blocks * k].reshape(n_blocks, k) ** 2
    return np.sqrt(r2.mean(axis=1))


def block_inputs(r, k):
    """Non-overlapping k-day blocks -> (realized VARIANCE per block, sign of block
    net return). The variance target has SNR; the sign feeds the leverage expert."""
    r = np.asarray(r, float)
    nb = len(r) // k
    rr = r[:nb * k].reshape(nb, k)
    return (rr ** 2).mean(axis=1), np.sign(rr.sum(axis=1))
