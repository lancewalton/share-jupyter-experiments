"""Test A: is confidence-gated long-vs-short drift divergence directional?

At each origin: short_drift = mean of last Ns returns, long_drift = mean of last
Nl returns. Signal s = long_drift - short_drift (under MEAN-REVERSION, s>0 means
recent returns are below the long-run mean -> expect catch-up -> positive forward
return, i.e. positive IC; MOMENTUM would give negative IC). Confidence c =
|divergence| / SE(short_drift) -- the z-score of the divergence (its 'outside the
envelope'-ness). We measure the information coefficient overall and STRATIFIED by
confidence: the hypothesis predicts |IC| rises with confidence.

Non-overlapping forward windows (stride = H) so significance is honest.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.data import close_array
from mc.returns import log_returns

GRID_SHORT = [20, 60]
GRID_LONG = [500, 1250]
GRID_H = [10, 20]
N_BUCKETS = 5


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _run_combo(r, Ns, Nl, H):
    first, last = Nl - 1, len(r) - H - 1
    origins = range(first, last + 1, H)  # non-overlapping forward windows
    s, c, y = [], [], []
    for t in origins:
        short = r[t - Ns + 1 : t + 1]
        long = r[t - Nl + 1 : t + 1]
        sd, ld = short.mean(), long.mean()
        se = short.std() / np.sqrt(Ns)
        s.append(ld - sd)
        c.append(abs(sd - ld) / se if se > 0 else 0.0)
        y.append(r[t + 1 : t + 1 + H].sum())
    return np.array(s), np.array(c), np.array(y)


def main():
    r = log_returns(close_array())
    print(f"FTSE log returns: {len(r)}\n")
    print(f"{'Ns':>4}{'Nl':>6}{'H':>4}{'n':>6}{'IC_all':>9}{'hit%':>7}   IC by confidence quintile (low->high)")

    curves = {}
    for H in GRID_H:
        for Ns in GRID_SHORT:
            for Nl in GRID_LONG:
                s, c, y = _run_combo(r, Ns, Nl, H)
                ic_all = _spearman(s, y)
                hit = np.mean(np.sign(s) == np.sign(y))
                order = np.argsort(c)
                buckets = np.array_split(order, N_BUCKETS)
                ic_q = [_spearman(s[b], y[b]) for b in buckets]
                curves[(Ns, Nl, H)] = ic_q
                qs = "  ".join(f"{v:+.3f}" for v in ic_q)
                print(f"{Ns:>4}{Nl:>6}{H:>4}{len(s):>6}{ic_all:>+9.3f}{hit*100:>6.1f}   {qs}")
        print()

    _plot(curves)
    print("saved direction.png")


def _plot(curves):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    xs = np.arange(1, N_BUCKETS + 1)
    for (Ns, Nl, H), ic_q in curves.items():
        ax = axes[0] if H == GRID_H[0] else axes[1]
        ax.plot(xs, ic_q, "o-", label=f"Ns={Ns}, Nl={Nl}")
    for ax, H in zip(axes, GRID_H):
        ax.axhline(0, color="k", lw=0.7, ls=":")
        ax.set_title(f"forward H = {H} days")
        ax.set_xlabel("confidence quintile (low -> high divergence significance)")
        ax.set_xticks(xs)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Information Coefficient (Spearman signal vs forward return)")
    fig.suptitle("Mean-reversion hypothesis: does IC rise with divergence confidence?")
    fig.tight_layout()
    fig.savefig("direction.png", dpi=110)


if __name__ == "__main__":
    main()
