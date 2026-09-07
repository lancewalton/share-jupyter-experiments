"""Beyond fixed pairs: a finer (Ns,Nl) sweep, and a >2 drift *term structure*.

Part 1: IC heatmap over a grid of short/long history lengths -- is the reversion
        signal robust across the plane or an artefact of a few pairs?
Part 2: use the whole drift-vs-lookback curve instead of one difference:
        - SLOPE of drift on log(lookback)  (positive slope = recent below long-run)
        - ROBUST ANCHOR: mean of long-window drifts minus mean of short-window
          drifts, with cross-timescale AGREEMENT as confidence.
        Compared, with confidence-stratified IC, against the best single pair.

Signal sign convention: positive => recent drift below long-run => under
mean-reversion expect a positive forward return (positive IC). Non-overlapping
forward windows (stride = H).
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.data import close_array
from mc.returns import log_returns

H = 20
SHORTS = [10, 20, 30, 40, 60, 90, 125, 180, 250]
LONGS = [250, 375, 500, 750, 1000, 1250, 1500]
LADDER = [20, 40, 60, 90, 125, 180, 250, 375, 500, 750, 1000, 1250]
SHORT_SET = [20, 40, 60]
LONG_SET = [750, 1000, 1250]
N_BUCKETS = 5


def _spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _drift(csum, t, N):
    """mean of r[t-N+1 : t+1] via prefix sums (csum[i] = sum of r[:i])."""
    return (csum[t + 1] - csum[t + 1 - N]) / N


def _origins(n, warm):
    return range(warm, n - H - 1 + 1, H)


def main():
    r = log_returns(close_array())
    csum = np.concatenate([[0.0], np.cumsum(r)])
    fsum = np.concatenate([np.cumsum(r), [0.0]])  # not used; keep explicit fwd below

    # ---- Part 1: (Ns, Nl) IC heatmap ----
    warm = max(LONGS) - 1
    origins = list(_origins(len(r), warm))
    fwd = np.array([r[t + 1 : t + 1 + H].sum() for t in origins])
    print(f"origins: {len(origins)}  (H={H}, non-overlapping)\n")

    IC = np.full((len(SHORTS), len(LONGS)), np.nan)
    for i, Ns in enumerate(SHORTS):
        for j, Nl in enumerate(LONGS):
            if Ns >= Nl:
                continue
            s = np.array([_drift(csum, t, Nl) - _drift(csum, t, Ns) for t in origins])
            IC[i, j] = _spearman(s, fwd)
    best = np.unravel_index(np.nanargmax(IC), IC.shape)
    print(f"Part 1 — best pair Ns={SHORTS[best[0]]}, Nl={LONGS[best[1]]}: "
          f"IC={IC[best]:+.3f}   (grid range {np.nanmin(IC):+.3f}..{np.nanmax(IC):+.3f}, "
          f"positive fraction {np.mean(IC[np.isfinite(IC)]>0):.0%})\n")

    # ---- Part 2: term-structure signals ----
    logN = np.log(np.array(LADDER, dtype=float))
    logN_c = logN - logN.mean()
    denom = (logN_c ** 2).sum()

    sig_pair, sig_slope, sig_ra = [], [], []
    conf_pair, conf_slope, conf_ra = [], [], []
    Ns_b, Nl_b = SHORTS[best[0]], LONGS[best[1]]
    for t in origins:
        drifts = np.array([_drift(csum, t, N) for N in LADDER])
        # best pair
        sd = _drift(csum, t, Ns_b); ld = _drift(csum, t, Nl_b)
        se = r[t - Ns_b + 1 : t + 1].std() / np.sqrt(Ns_b)
        sig_pair.append(ld - sd); conf_pair.append(abs(ld - sd) / se if se > 0 else 0)
        # slope of drift on log(N): positive => drift rises with lookback
        beta = float((logN_c * (drifts - drifts.mean())).sum() / denom)
        resid = drifts - (drifts.mean() + beta * logN_c)
        se_b = np.sqrt((resid ** 2).sum() / (len(LADDER) - 2) / denom)
        sig_slope.append(beta); conf_slope.append(abs(beta) / se_b if se_b > 0 else 0)
        # robust anchor vs short, with agreement across short windows
        shorts = np.array([_drift(csum, t, N) for N in SHORT_SET])
        anchor = np.mean([_drift(csum, t, N) for N in LONG_SET])
        sig_ra.append(anchor - shorts.mean())
        agree = np.mean(shorts < anchor)              # fraction of short windows below anchor
        conf_ra.append(abs(agree - 0.5) * 2)          # 0 (split) .. 1 (unanimous)

    def report(name, sig, conf):
        sig, conf = np.array(sig), np.array(conf)
        ic = _spearman(sig, fwd)
        order = np.argsort(conf)
        ic_q = [_spearman(sig[b], fwd[b]) for b in np.array_split(order, N_BUCKETS)]
        print(f"  {name:<16} IC={ic:+.3f}   by confidence quintile: "
              + "  ".join(f"{v:+.3f}" for v in ic_q))
        return ic_q

    print("Part 2 — whole-curve signals vs the best pair:")
    q_pair = report(f"pair({Ns_b},{Nl_b})", sig_pair, conf_pair)
    q_slope = report("slope~logN", sig_slope, conf_slope)
    q_ra = report("robust-anchor", sig_ra, conf_ra)

    _plot(IC, {"pair": q_pair, "slope~logN": q_slope, "robust-anchor": q_ra})
    print("\nsaved termstructure.png")


def _plot(IC, qcurves):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    im = a1.imshow(IC, origin="lower", aspect="auto", cmap="RdBu_r",
                   vmin=-np.nanmax(np.abs(IC)), vmax=np.nanmax(np.abs(IC)))
    a1.set_xticks(range(len(LONGS))); a1.set_xticklabels(LONGS)
    a1.set_yticks(range(len(SHORTS))); a1.set_yticklabels(SHORTS)
    a1.set_xlabel("long window Nl"); a1.set_ylabel("short window Ns")
    a1.set_title("Part 1: reversion IC over (Ns, Nl)")
    for i in range(len(SHORTS)):
        for j in range(len(LONGS)):
            if np.isfinite(IC[i, j]):
                a1.text(j, i, f"{IC[i,j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=a1, shrink=0.8, label="IC")

    xs = np.arange(1, N_BUCKETS + 1)
    for name, q in qcurves.items():
        a2.plot(xs, q, "o-", label=name)
    a2.axhline(0, color="k", lw=0.7, ls=":")
    a2.set_xlabel("confidence quintile (low -> high)"); a2.set_ylabel("IC")
    a2.set_xticks(xs); a2.set_title("Part 2: whole-curve vs best pair")
    a2.legend(fontsize=9)
    fig.tight_layout(); fig.savefig("termstructure.png", dpi=110)


if __name__ == "__main__":
    main()
