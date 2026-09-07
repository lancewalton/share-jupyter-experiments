"""A structurally-different hierarchy: mix diverse volatility experts.

Instead of nesting learning rates (which self-cancel), each level controls a
different, non-substitutable thing:
  Level 1  experts   -- diverse fixed predictors of next-block realized variance
  Level 2  mixer     -- adaptive convex weights deciding which expert to trust now
  Level 3  meta-mixer-- adapts the mixer's responsiveness (eta)

Experts are built to capture DIFFERENT structure, so no single one dominates and
the mixer has something real to do:
  - short EWMA        : recent volatility level (persistence)
  - long mean         : slow reversion to the unconditional level
  - momentum          : extrapolate the recent trend in volatility
  - leverage          : recent variance conditioned on down- vs up-moves
All predictions are one-step-ahead and causal (use only data strictly before j).
"""
import numpy as np


def _ewma_state(v, halflife):
    a = 1 - np.exp(-np.log(2) / halflife)
    pred = np.full(len(v), np.nan)
    s = v[0]
    for j in range(1, len(v)):
        pred[j] = s
        s = (1 - a) * s + a * v[j]
    return pred


def build_experts(rv_var, block_sign):
    """rv_var: block realized-variance series. block_sign: mean sign of returns in
    each block (for the leverage expert). Returns dict name -> one-step preds."""
    v = np.asarray(rv_var, float)
    n = len(v)
    experts = {}
    experts["ewma_short"] = _ewma_state(v, 1.5)
    experts["ewma_mid"] = _ewma_state(v, 6.0)

    # long-run mean reversion: expanding mean of past variance
    csum = np.concatenate([[0], np.cumsum(v)])
    idx = np.arange(n)
    longmean = np.full(n, np.nan)
    longmean[1:] = csum[1:n] / idx[1:]           # mean of v[0:j]
    experts["long_mean"] = longmean

    # momentum: last value + recent change (short EWMA slope)
    short = _ewma_state(v, 1.5)
    mom = np.full(n, np.nan)
    mom[2:] = short[1:-1] + (short[1:-1] - short[:-2])
    experts["momentum"] = np.clip(mom, 0, None)

    # leverage: EWMA of variance but only counting down-move blocks, blended
    down = (np.asarray(block_sign) < 0).astype(float)
    lev = np.full(n, np.nan)
    s_all, s_down = v[0], v[0]
    a = 1 - np.exp(-np.log(2) / 3.0)
    for j in range(1, n):
        lev[j] = 0.5 * s_all + 0.5 * s_down
        s_all = (1 - a) * s_all + a * v[j]
        if down[j]:
            s_down = (1 - a) * s_down + a * v[j]
    experts["leverage"] = lev
    return experts


class Mixer:
    """Level 2: multiplicative-weights (Hedge) convex mixture of experts."""
    def __init__(self, n_experts, eta=5.0):
        self.p = np.ones(n_experts) / n_experts
        self.eta = eta

    def predict(self, preds):
        return float(np.dot(self.p, preds))

    def update(self, preds, actual):
        out = self.predict(preds)
        loss = (preds - actual) ** 2
        loss = loss / (np.mean(loss) + 1e-18)          # scale-free losses
        self.p *= np.exp(-self.eta * loss)
        self.p /= self.p.sum()
        return out


class MetaMixer:
    """Level 3: same Hedge mixer, but eta itself adapts. eta rises when the mix
    has been sluggish (recent loss trending up), falls when stable -- a different
    knob from Level 2's weights, not a nested learning rate on the same quantity."""
    def __init__(self, n_experts, eta=5.0, meta=0.1, eta_min=0.2, eta_max=40.0):
        self.p = np.ones(n_experts) / n_experts
        self.eta = eta
        self.meta, self.eta_min, self.eta_max = meta, eta_min, eta_max
        self.loss_ewma = None

    def predict(self, preds):
        return float(np.dot(self.p, preds))

    def update(self, preds, actual):
        out = self.predict(preds)
        mix_loss = (out - actual) ** 2
        if self.loss_ewma is None:
            self.loss_ewma = mix_loss
        trend = (mix_loss - self.loss_ewma) / (self.loss_ewma + 1e-18)
        self.loss_ewma = 0.9 * self.loss_ewma + 0.1 * mix_loss
        self.eta = float(np.clip(self.eta * np.exp(self.meta * np.tanh(trend)),
                                 self.eta_min, self.eta_max))
        loss = (preds - actual) ** 2
        loss = loss / (np.mean(loss) + 1e-18)
        self.p *= np.exp(-self.eta * loss)
        self.p /= self.p.sum()
        return out


def run_mixer(experts, actual, mixer):
    names = list(experts)
    P = np.array([experts[k] for k in names])           # (E, n)
    n = P.shape[1]
    pred = np.full(n, np.nan)
    weights = np.full((n, len(names)), np.nan)
    for j in range(n):
        col = P[:, j]
        if not np.all(np.isfinite(col)) or not np.isfinite(actual[j]):
            continue
        pred[j] = mixer.update(col, actual[j])
        weights[j] = mixer.p
    return pred, weights, names
