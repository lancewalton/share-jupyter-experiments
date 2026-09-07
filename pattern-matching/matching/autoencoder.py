"""A compact numpy autoencoder and a deflationary (growing) trainer.

Each *component* is a 1-D-bottleneck non-linear autoencoder: it compresses a
window to a single scalar code and reconstructs it. Because the encoder and
decoder are non-linear (tanh), one component can trace a *curved* 1-D manifold
through window-space -- e.g. "a dip whose position slides", which a linear PCA
component cannot represent (PCA would need several orthogonal sinusoids). This
is the whole reason for going non-linear.

The trainer grows components one at a time in the spirit of the brief: fit
component 1 on the data, FREEZE it, fit component 2 on the *residual*, and so on
(deflation). A component's activation is its code computed on the residual left
by all earlier, frozen components.

Deliberately small and regularised: at this signal-to-noise, capacity is the
enemy (the sibling projects showed complexity loses on direction).
"""
from __future__ import annotations

import numpy as np


class Autoencoder1D:
    """P -> h -tanh-> 1 (code) -> h -tanh-> P, trained by Adam on MSE + L2."""

    def __init__(self, P: int, h: int = 16, l2: float = 1e-5, seed: int = 0):
        rng = np.random.default_rng(seed)
        s = 0.1
        self.W1 = rng.normal(0, s, (h, P)); self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, s, (1, h)); self.b2 = np.zeros(1)
        self.W3 = rng.normal(0, s, (h, 1)); self.b3 = np.zeros(h)
        self.W4 = rng.normal(0, s, (P, h)); self.b4 = np.zeros(P)
        self.l2 = l2

    # --- forward pieces -------------------------------------------------
    def encode(self, X):
        h1 = np.tanh(X @ self.W1.T + self.b1)
        return h1 @ self.W2.T + self.b2  # (N,1) code

    def decode(self, Z):
        h2 = np.tanh(Z @ self.W3.T + self.b3)
        return h2 @ self.W4.T + self.b4  # (N,P)

    def reconstruct(self, X):
        return self.decode(self.encode(X))

    # --- loss + gradients ----------------------------------------------
    def _forward_cache(self, X):
        h1 = np.tanh(X @ self.W1.T + self.b1)
        z = h1 @ self.W2.T + self.b2
        h2 = np.tanh(z @ self.W3.T + self.b3)
        xr = h2 @ self.W4.T + self.b4
        return h1, z, h2, xr

    def loss_and_grads(self, X):
        N = len(X)
        h1, z, h2, xr = self._forward_cache(X)
        diff = xr - X
        mse = np.mean(np.sum(diff ** 2, axis=1))
        reg = self.l2 * sum(np.sum(W ** 2) for W in (self.W1, self.W2, self.W3, self.W4))
        d_xr = 2.0 * diff / N
        gW4 = d_xr.T @ h2 + 2 * self.l2 * self.W4
        gb4 = d_xr.sum(0)
        d_h2 = d_xr @ self.W4
        d_pre2 = d_h2 * (1 - h2 ** 2)
        gW3 = d_pre2.T @ z + 2 * self.l2 * self.W3
        gb3 = d_pre2.sum(0)
        d_z = d_pre2 @ self.W3
        gW2 = d_z.T @ h1 + 2 * self.l2 * self.W2
        gb2 = d_z.sum(0)
        d_h1 = d_z @ self.W2
        d_pre1 = d_h1 * (1 - h1 ** 2)
        gW1 = d_pre1.T @ X + 2 * self.l2 * self.W1
        gb1 = d_pre1.sum(0)
        grads = dict(W1=gW1, b1=gb1, W2=gW2, b2=gb2,
                     W3=gW3, b3=gb3, W4=gW4, b4=gb4)
        return mse + reg, grads

    def params(self):
        return dict(W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                    W3=self.W3, b3=self.b3, W4=self.W4, b4=self.b4)

    def fit(self, X, epochs=60, batch=2048, lr=2e-3, seed=0, verbose=False):
        rng = np.random.default_rng(seed)
        p = self.params()
        m = {k: np.zeros_like(v) for k, v in p.items()}
        v = {k: np.zeros_like(v) for k, v in p.items()}
        b1c, b2c, eps, t = 0.9, 0.999, 1e-8, 0
        N = len(X)
        for ep in range(epochs):
            idx = rng.permutation(N)
            for s in range(0, N, batch):
                xb = X[idx[s:s + batch]]
                _, g = self.loss_and_grads(xb)
                t += 1
                for k in p:
                    m[k] = b1c * m[k] + (1 - b1c) * g[k]
                    v[k] = b2c * v[k] + (1 - b2c) * g[k] ** 2
                    mh = m[k] / (1 - b1c ** t)
                    vh = v[k] / (1 - b2c ** t)
                    p[k] -= lr * mh / (np.sqrt(vh) + eps)
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                loss, _ = self.loss_and_grads(X)
                print(f"    epoch {ep:3d}  loss {loss:.5f}")
        return self


class MLPAutoencoder:
    """General autoencoder: P -> enc_hidden... -> code -> dec_hidden... -> P.

    Every layer is tanh EXCEPT the final (linear) output, so the bottleneck
    itself is tanh -- a ``code_dim``-unit non-linear code. Used for the capacity
    robustness check: does a wider/deeper net with a multi-unit code find
    structure the small 1-D-per-component stack could not?
    """

    def __init__(self, P, enc_hidden, code_dim, l2=1e-5, seed=0):
        rng = np.random.default_rng(seed)
        enc = [P] + list(enc_hidden) + [code_dim]
        dec = [code_dim] + list(reversed(enc_hidden)) + [P]
        dims = list(zip(enc[:-1], enc[1:])) + list(zip(dec[:-1], dec[1:]))
        self.n_enc = len(enc) - 1                       # encoder weight layers
        self.W, self.b, self.act = [], [], []
        for i, (fin, fout) in enumerate(dims):
            self.W.append(rng.normal(0, np.sqrt(1.0 / fin), (fout, fin)))
            self.b.append(np.zeros(fout))
            self.act.append("tanh")
        self.act[-1] = "linear"                          # linear reconstruction
        self.l2 = l2

    def _forward(self, X, upto=None):
        a = X; acts = [a]; pre = []
        n = len(self.W) if upto is None else upto
        for i in range(n):
            z = a @ self.W[i].T + self.b[i]
            a = np.tanh(z) if self.act[i] == "tanh" else z
            pre.append(z); acts.append(a)
        return a, acts, pre

    def encode(self, X):
        a, _, _ = self._forward(X, upto=self.n_enc)
        return a

    def reconstruct(self, X):
        a, _, _ = self._forward(X)
        return a

    def loss_and_grads(self, X):
        N = len(X)
        recon, acts, _ = self._forward(X)
        diff = recon - X
        mse = np.mean(np.sum(diff ** 2, axis=1))
        reg = self.l2 * sum(np.sum(W ** 2) for W in self.W)
        d = 2.0 * diff / N
        gW = [None] * len(self.W); gb = [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            a_out, a_in = acts[i + 1], acts[i]
            dz = d * (1 - a_out ** 2) if self.act[i] == "tanh" else d
            gW[i] = dz.T @ a_in + 2 * self.l2 * self.W[i]
            gb[i] = dz.sum(0)
            d = dz @ self.W[i]
        return mse + reg, (gW, gb)

    def fit(self, X, epochs=120, batch=2048, lr=2e-3, seed=0, verbose=False):
        rng = np.random.default_rng(seed)
        mW = [np.zeros_like(W) for W in self.W]; vW = [np.zeros_like(W) for W in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        N = len(X)
        for ep in range(epochs):
            idx = rng.permutation(N)
            for s in range(0, N, batch):
                xb = X[idx[s:s + batch]]
                _, (gW, gb) = self.loss_and_grads(xb)
                t += 1
                for i in range(len(self.W)):
                    mW[i] = b1 * mW[i] + (1 - b1) * gW[i]
                    vW[i] = b2 * vW[i] + (1 - b2) * gW[i] ** 2
                    self.W[i] -= lr * (mW[i] / (1 - b1 ** t)) / (np.sqrt(vW[i] / (1 - b2 ** t)) + eps)
                    mb[i] = b1 * mb[i] + (1 - b1) * gb[i]
                    vb[i] = b2 * vb[i] + (1 - b2) * gb[i] ** 2
                    self.b[i] -= lr * (mb[i] / (1 - b1 ** t)) / (np.sqrt(vb[i] / (1 - b2 ** t)) + eps)
            if verbose and (ep % 20 == 0 or ep == epochs - 1):
                loss, _ = self.loss_and_grads(X)
                print(f"    epoch {ep:3d}  loss {loss:.5f}")
        return self


def deflationary_fit(X, k, h=16, l2=1e-5, epochs=60, seed=0, verbose=False):
    """Grow ``k`` frozen 1-D components on residuals. Returns the list of AEs."""
    P = X.shape[1]
    aes, resid = [], X.copy()
    for i in range(k):
        ae = Autoencoder1D(P, h=h, l2=l2, seed=seed + i)
        ae.fit(resid, epochs=epochs, seed=seed + i, verbose=verbose)
        resid = resid - ae.reconstruct(resid)
        aes.append(ae)
        if verbose:
            print(f"  component {i+1}: residual var {np.var(resid):.5f}")
    return aes


def activations(aes, X):
    """Per-window codes ``(N, k)``, computing the residual chain causally."""
    resid = X.copy()
    cols = []
    for ae in aes:
        cols.append(ae.encode(resid).ravel())
        resid = resid - ae.reconstruct(resid)
    return np.column_stack(cols)


def final_residual(aes, X):
    resid = X.copy()
    for ae in aes:
        resid = resid - ae.reconstruct(resid)
    return resid
