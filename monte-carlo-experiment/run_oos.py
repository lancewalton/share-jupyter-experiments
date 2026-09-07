"""Out-of-sample check (temporal): does FHS's calibration edge hold per era?

Forecasters use fixed trailing-history rules (nothing fitted to outcomes), so
bucketing origins by calendar period is a genuine generalisation test. We fix
the models chosen in-sample -- EWMA benchmark, fhs(gamma=0), fhs(gamma=1.0) --
and report per-period tail calibration and CRPS. No test-period peeking.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.data import load_close
from mc.returns import log_returns, realised_cumulative
from mc.forecasters import gaussian_ewma, fhs_ewma
from mc.scoring import crps_from_quantiles, interval_hit
from mc.walkforward import DENSE_LEVELS

T, MIN_HISTORY, STRIDE = 10, 1000, 5
PERIODS = [("1994-2003", 1994, 2003), ("2004-2013", 2004, 2013), ("2014-2021", 2014, 2021)]

forecasters = {
    "ewma(0.94)": gaussian_ewma(lam=0.94, min_obs=MIN_HISTORY),
    "fhs g=0.0": fhs_ewma(lam=0.94, gamma=0.0, n_sims=10_000, seed=0),
    "fhs g=1.0": fhs_ewma(lam=0.94, gamma=1.0, n_sims=10_000, seed=0),
}


def _idx(level):
    return int(np.argmin(np.abs(DENSE_LEVELS - level)))


def main() -> None:
    close = load_close()
    r = log_returns(close.to_numpy())
    dates = close.index[1:]  # r[i] ends on date dates[i]
    lev = DENSE_LEVELS
    i99 = (_idx(0.005), _idx(0.995))
    i90 = (_idx(0.05), _idx(0.95))

    first, last = MIN_HISTORY - 1, len(r) - T - 1
    origins = list(range(first, last + 1, STRIDE))
    years = np.array([dates[t].year for t in origins])

    print(f"origins: {len(origins)}   {dates[origins[0]].date()} -> {dates[origins[-1]].date()}\n")
    table = {name: {} for name in forecasters}

    for name, f in forecasters.items():
        crps = np.empty(len(origins))
        hit99 = np.empty(len(origins), bool)
        hit90 = np.empty(len(origins), bool)
        for k, t in enumerate(origins):
            q = f(r[: t + 1], T, lev)
            y = realised_cumulative(r, t, T)
            crps[k] = crps_from_quantiles(q, y, lev).mean()
            hit99[k] = interval_hit(q[i99[0], T - 1], q[i99[1], T - 1], y[T - 1])
            hit90[k] = interval_hit(q[i90[0], T - 1], q[i90[1], T - 1], y[T - 1])
        for label, y0, y1 in PERIODS:
            m = (years >= y0) & (years <= y1)
            table[name][label] = (
                crps[m].mean(), hit90[m].mean(), hit99[m].mean(), int(m.sum())
            )

    for label, _, _ in PERIODS:
        n = table["ewma(0.94)"][label][3]
        print(f"=== {label}   (n_origins={n}) ===")
        print(f"  {'model':<12}{'CRPS h10':>10}{'cov90':>8}{'cov99':>8}")
        for name in forecasters:
            c, c90, c99, _ = table[name][label]
            print(f"  {name:<12}{c:>10.5f}{c90:>8.3f}{c99:>8.3f}")
        print()

    _plot(table)
    print("saved oos.png")


def _plot(table) -> None:
    labels = [p[0] for p in PERIODS]
    names = list(forecasters)
    x = np.arange(len(labels))
    w = 0.25
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, name in enumerate(names):
        breach = [1 - table[name][lab][2] for lab in labels]
        a1.bar(x + (i - 1) * w, breach, w, label=name)
        crps = [table[name][lab][0] for lab in labels]
        a2.bar(x + (i - 1) * w, crps, w, label=name)
    a1.axhline(0.01, color="k", ls=":", label="nominal 1%")
    a1.set_xticks(x); a1.set_xticklabels(labels)
    a1.set_ylabel("99% breach rate (h=10)"); a1.set_title("tail breaches per era")
    a1.legend(fontsize=8)
    a2.set_xticks(x); a2.set_xticklabels(labels)
    a2.set_ylabel("mean CRPS (h=10)"); a2.set_title("CRPS per era")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("oos.png", dpi=110)


if __name__ == "__main__":
    main()
