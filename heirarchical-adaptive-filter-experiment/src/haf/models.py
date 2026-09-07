"""Filter models for the hierarchical adaptive FIR experiment.

Level 0  FixedFIR         least-squares FIR, fit once (the stationary baseline)
Level 1  AdaptiveFIR      NLMS: weights adapt, fixed learning rate
Level 2  MetaAdaptiveFIR  learning rate adapts from smoothed-error trend (notebook rule)
Level 2  IDBD             learning rate adapts from gradient-trace correlation (Sutton 1992)
Level 3  MetaIDBD         the meta-rate itself adapts, self-similarly

All adaptive models carry a bias/intercept term, because volatility targets have
a large non-zero level that an intercept-free FIR would waste capacity chasing.
The IDBD variants divide gradients by input energy (NLMS-style) so per-weight
step sizes stay in a stable range regardless of the number of lags.
"""
import numpy as np


class FixedFIR:
    def __init__(self, n_lags):
        self.n_lags = n_lags
        self.w = np.zeros(n_lags + 1)

    def fit(self, v):
        v = np.asarray(v, float)
        X = np.array([v[i - self.n_lags:i][::-1] for i in range(self.n_lags, len(v))])
        y = v[self.n_lags:]
        Xb = np.column_stack([np.ones(len(X)), X])
        self.w = np.linalg.lstsq(Xb, y, rcond=None)[0]

    def predict(self, x):
        return float(self.w[0] + np.dot(self.w[1:], x))


class AdaptiveFIR:
    """Level 1: NLMS with intercept, fixed learning rate."""
    def __init__(self, n_lags, lr=0.1, eps=1e-6):
        self.n_lags = n_lags
        self.lr, self.eps = lr, eps
        self.w = np.zeros(n_lags + 1)

    def predict(self, x):
        return float(self.w[0] + np.dot(self.w[1:], x))

    def update(self, x, actual):
        pred = self.predict(x)
        err = actual - pred
        xb = np.concatenate([[1.0], x])
        self.w += self.lr * err * xb / (self.eps + np.dot(xb, xb))
        return {"pred": pred, "err": err, "lr": self.lr}


class MetaAdaptiveFIR:
    """Level 2 (notebook rule): learning rate follows the trend in smoothed error energy."""
    def __init__(self, n_lags, lr=0.1, meta_rate=0.05, smooth=0.02,
                 lr_min=1e-4, lr_max=1.0, eps=1e-6):
        self.n_lags = n_lags
        self.lr, self.meta_rate, self.smooth = lr, meta_rate, smooth
        self.lr_min, self.lr_max, self.eps = lr_min, lr_max, eps
        self.w = np.zeros(n_lags + 1)
        self.energy = 0.0

    def predict(self, x):
        return float(self.w[0] + np.dot(self.w[1:], x))

    def update(self, x, actual):
        pred = self.predict(x)
        err = actual - pred
        prev = self.energy
        self.energy = (1 - self.smooth) * self.energy + self.smooth * err ** 2
        trend = (self.energy - prev) / (prev + self.eps)
        self.lr = float(np.clip(self.lr * np.exp(self.meta_rate * trend),
                                self.lr_min, self.lr_max))
        xb = np.concatenate([[1.0], x])
        self.w += self.lr * err * xb / (self.eps + np.dot(xb, xb))
        return {"pred": pred, "err": err, "lr": self.lr, "energy": self.energy}


class IDBD:
    """Level 2 (principled, Sutton 1992): per-weight step size, energy-normalised."""
    def __init__(self, n_in, theta=0.1, init_beta=None, eps=1e-6):
        self.n = n_in
        self.theta, self.eps = theta, eps
        self.beta = np.full(n_in, np.log(0.1) if init_beta is None else init_beta)
        self.w = np.zeros(n_in)
        self.h = np.zeros(n_in)

    def predict(self, x):
        return float(np.dot(self.w, x))

    def update(self, x, actual):
        pred = self.predict(x)
        delta = actual - pred
        xn = x / (self.eps + np.dot(x, x))
        self.beta = np.clip(self.beta + self.theta * delta * xn * self.h, -15, 2)
        alpha = np.exp(self.beta)
        self.w += alpha * delta * xn
        decay = np.maximum(0.0, 1.0 - alpha * x * xn)
        self.h = self.h * decay + alpha * delta * xn
        return {"pred": pred, "err": delta, "alpha": alpha}


class MetaIDBD:
    """Level 3 (per-weight theta). Kept for reference; its per-weight meta-meta
    signal is too weak to move theta on noisy data -- see MetaIDBDScalar."""
    def __init__(self, n_in, meta_meta=0.1, init_beta=-2.3, init_gamma=None, eps=1e-6):
        self.n = n_in
        self.mm, self.eps = meta_meta, eps
        self.beta = np.full(n_in, init_beta)
        self.gamma = np.full(n_in, np.log(0.05) if init_gamma is None else init_gamma)
        self.w = np.zeros(n_in)
        self.h = np.zeros(n_in)
        self.hb = np.zeros(n_in)

    def predict(self, x):
        return float(np.dot(self.w, x))

    def update(self, x, actual):
        pred = self.predict(x)
        delta = actual - pred
        xn = x / (self.eps + np.dot(x, x))
        grad = delta * xn
        beta_drive = grad * self.h
        self.gamma = np.clip(self.gamma + self.mm * beta_drive * self.hb, -12, 1)
        theta = np.exp(self.gamma)
        self.beta = np.clip(self.beta + theta * beta_drive, -15, 2)
        alpha = np.exp(self.beta)
        self.w += alpha * grad
        decay = np.maximum(0.0, 1.0 - alpha * x * xn)
        self.h = self.h * decay + alpha * grad
        self.hb = self.hb * decay + theta * beta_drive
        return {"pred": pred, "err": delta, "alpha": alpha, "theta": theta}


class MetaIDBDScalar:
    """Level 3 done to actually move: a SINGLE global meta-rate theta, adapted
    from the meta-meta signal POOLED across all weights (which raises its SNR),
    with the same IDBD correlation logic one level up.

    Pooling turns 21 near-zero-mean per-weight signals into one usable scalar, so
    theta stops being inert. gamma = log theta; H is a scalar trace of the pooled
    beta-driving signal; when the current pooled drive agrees with that trace the
    step sizes are persistently trending (we are chasing) and theta rises, when it
    disagrees (jitter) theta falls.
    """
    def __init__(self, n_in, meta_meta=0.05, init_beta=-2.3, init_gamma=None,
                 gamma_clip=(-8.0, 2.5), eps=1e-6):
        self.n = n_in
        self.mm, self.eps = meta_meta, eps
        self.gclip = gamma_clip
        self.beta = np.full(n_in, init_beta)
        self.gamma = np.log(0.1) if init_gamma is None else init_gamma  # scalar
        self.w = np.zeros(n_in)
        self.h = np.zeros(n_in)
        self.H = 0.0                                                    # scalar trace

    def predict(self, x):
        return float(np.dot(self.w, x))

    def update(self, x, actual):
        pred = self.predict(x)
        delta = actual - pred
        xn = x / (self.eps + np.dot(x, x))
        grad = delta * xn
        beta_drive = grad * self.h
        pooled = float(np.sum(beta_drive))               # scalar meta-meta signal
        theta = np.exp(self.gamma)
        # Level 3: one scalar gamma, IDBD logic on the pooled signal
        self.gamma = float(np.clip(self.gamma + self.mm * pooled * self.H,
                                   self.gclip[0], self.gclip[1]))
        theta = np.exp(self.gamma)
        # Level 2: per-weight beta at the (now adaptive, shared) rate theta
        self.beta = np.clip(self.beta + theta * beta_drive, -15, 2)
        alpha = np.exp(self.beta)
        # Level 1: weights
        self.w += alpha * grad
        decay = np.maximum(0.0, 1.0 - alpha * x * xn)
        self.h = self.h * decay + alpha * grad
        # scalar trace of the pooled drive (bounded decay for stability)
        self.H = 0.99 * self.H + theta * pooled
        return {"pred": pred, "err": delta, "alpha": alpha, "theta": theta}


# ------------------------------------------------------------------- runners
def run_online(v, model):
    """Strict one-step-ahead for bias models (predict then update)."""
    v = np.asarray(v, float)
    n = len(v)
    pred = np.full(n, np.nan)
    lr = np.full(n, np.nan)
    for t in range(model.n_lags, n):
        out = model.update(v[t - model.n_lags:t][::-1], v[t])
        pred[t], lr[t] = out["pred"], out.get("lr", np.nan)
    return pred, lr


def run_idbd(v, model_factory, n_lags, split):
    """One-step-ahead for IDBD models on inputs standardised with training-half
    stats (causal for the held-out region); predictions mapped back to raw scale."""
    v = np.asarray(v, float)
    mu, sd = v[:split].mean(), v[:split].std() + 1e-12
    z = (v - mu) / sd
    n = len(z)
    model = model_factory(n_lags + 1)          # +1 for the bias feature
    pred = np.full(n, np.nan)
    step = np.full(n, np.nan)
    aux = np.full(n, np.nan)
    for t in range(n_lags, n):
        x = np.concatenate([[1.0], z[t - n_lags:t][::-1]])
        out = model.update(x, z[t])
        pred[t] = out["pred"] * sd + mu
        step[t] = np.mean(out["alpha"][1:])    # step size on the lag weights
        if "theta" in out:
            th = out["theta"]
            aux[t] = float(np.mean(np.atleast_1d(th)))
    return pred, step, aux
