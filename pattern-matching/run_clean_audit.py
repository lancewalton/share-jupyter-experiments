"""Data-quality audit: do the pattern-matching positives survive on clean data?

The universe contains a few corrupt series (SPX index; DCC/ITV scrambled) and
possible unadjusted splits. We re-run the two AT-RISK positive findings on three
universes and compare:

  * full        -- everything (122), as originally used;
  * clean       -- minus indices and bad prints (spike-and-reversal);
  * clean+split -- also minus isolated split-like steps (conservative).

At-risk findings:
  A. reversion IC  -- pooled OOS rank-IC of -drift vs the forward return
     (bad prints fabricate a "bounce", so this is the most exposed);
  B. detector enrichment -- real vs sign-flip tail enrichment@5% (Part II's
     "directional structure exists").

If both hold across universes, the findings are robust to the data issues.
"""
from __future__ import annotations

import numpy as np

from matching.surrogate import sign_flip_surrogate
from matching.universe import clean_universe, load_universe
from run_phase2_direction import _collect as collect_fwd, _spearman, STRIDE, TRAIN_FRAC
from run_phase5_detector import _collect as collect_sf, _fit_score, _labelled

L, H = 80, 20


def reversion_ic(uni):
    W, y, d = collect_fwd(uni, L, H)
    tr = d.astype("int64") <= np.quantile(d.astype("int64"), TRAIN_FRAC)
    te = np.where(~tr)[0][::max(1, round(H / STRIDE))]
    return _spearman(-W[te, -1], y[te]), len(te)


def detector(uni, rng):
    Wr, dr = collect_sf(uni, L)
    Ws, ds = collect_sf(uni, L, transform=lambda lp: sign_flip_surrogate(lp, rng))
    r = _fit_score(*_labelled(Wr, dr, Ws, ds))
    return r["auc"], r["e5"]


def main():
    full = load_universe()
    clean, drop1 = clean_universe(max_bad=1)
    claggr, drop2 = clean_universe(max_bad=1, max_steps=2)
    print(f"full        : {len(full)} series")
    print(f"clean       : {len(clean)} series  (dropped {len(drop1)}: "
          f"{', '.join(sorted(drop1))})")
    print(f"clean+split : {len(claggr)} series  (dropped {len(drop2)})\n")

    rng = np.random.default_rng(0)
    print(f"{'universe':12s}  {'n':>4s}   reversion-IC (OOS)   detector AUC / enrich@5%")
    for tag, uni in [("full", full), ("clean", clean), ("clean+split", claggr)]:
        ic, nte = reversion_ic(uni)
        auc, e5 = detector(uni, rng)
        print(f"{tag:12s}  {len(uni):>4d}   {ic:+.4f}              {auc:.3f} / {e5:+.3f}")

    print("\n=> if reversion-IC and enrich@5% are stable across rows, the positives")
    print("   are robust to the data corruption; if they collapse on clean data,")
    print("   the 'faint structure' was partly artefact.")


if __name__ == "__main__":
    main()
