"""Phase-0 baselines: drift-zero predictive quantiles of cumulative log return.

A forecaster is a callable ``f(history, T, levels) -> ndarray(len(levels), T)``
where ``history`` is the log-return history available at the origin. Drift is
pinned to zero throughout: the predictive distribution of cumulative return is
centred on zero, so we are testing *width* (a volatility forecast), not the
centre.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np

_N = NormalDist()


def _z(levels: np.ndarray) -> np.ndarray:
    """Standard-normal inverse CDF for each level (stdlib, no scipy)."""
    return np.array([_N.inv_cdf(float(p)) for p in levels], dtype=float)


def gaussian_ewma(lam: float = 0.94, min_obs: int = 20):
    """RiskMetrics-style baseline: iid Normal increments with EWMA volatility.

    Daily variance is estimated by an exponentially weighted mean of squared
    returns (drift ~ 0, so raw squares). Under iid N(0, sigma^2) increments the
    cumulative return over ``h`` days is N(0, h*sigma^2), giving quantiles
    ``z(level) * sqrt(h) * sigma``.
    """

    def forecaster(history: np.ndarray, T: int, levels: np.ndarray) -> np.ndarray:
        r = np.asarray(history, dtype=float)
        if len(r) < min_obs:
            raise ValueError("insufficient history for EWMA volatility")
        # EWMA recursion seeded with the sample variance of the first block.
        var = float(np.var(r[:min_obs]))
        for x in r[min_obs:]:
            var = lam * var + (1.0 - lam) * x * x
        sigma = np.sqrt(var)
        z = _z(levels)  # shape (L,)
        h = np.arange(1, T + 1, dtype=float)  # shape (T,)
        return sigma * np.outer(z, np.sqrt(h))  # (L, T)

    return forecaster


def bootstrap_iid(N: int, n_sims: int = 10_000, seed: int = 0):
    """Phase-1 core: IID resampling of the last ``N`` demeaned daily returns.

    For each simulated path we draw ``T`` returns with replacement from the
    demeaned window and cumulatively sum them, giving an ensemble of cumulative
    log-return paths (shape ``n_sims x T``) centred on zero. Predictive
    quantiles are read directly off the ensemble.

    Unlike ``empirical_scaled`` this does *not* assume the sqrt-T spread law:
    the multi-day distribution is the honest IID convolution of daily returns.
    It still assumes returns are IID (no volatility clustering) — the gap that
    Phase 2's block bootstrap and recency kernels exist to close.
    """
    rng = np.random.default_rng(seed)

    def forecaster(history: np.ndarray, T: int, levels: np.ndarray) -> np.ndarray:
        r = np.asarray(history, dtype=float)
        if len(r) < N:
            raise ValueError("insufficient history for bootstrap window")
        window = r[-N:]
        window = window - window.mean()  # drift pinned to zero
        idx = rng.integers(0, N, size=(n_sims, T))
        paths = np.cumsum(window[idx], axis=1)  # (n_sims, T) cumulative returns
        return np.quantile(paths, levels, axis=0)  # (len(levels), T)

    return forecaster


def _paths_to_quantiles(paths: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Pin drift to zero per horizon (subtract column means) then read quantiles."""
    paths = paths - paths.mean(axis=0, keepdims=True)
    return np.quantile(paths, levels, axis=0)


def _recency_weights(N: int, half_life: float) -> np.ndarray:
    """Exponential-decay weights over a window; newest (last) index heaviest."""
    age = np.arange(N - 1, -1, -1, dtype=float)  # index 0 oldest .. N-1 newest
    w = 0.5 ** (age / half_life)
    return w / w.sum()


def bootstrap_age_weighted(N: int, half_life: float, n_sims: int = 10_000, seed: int = 0):
    """IID bootstrap with recency-weighted resampling (age-weighted historical sim).

    Recent returns are drawn more often (exponential decay with ``half_life``
    days), so the envelope tracks the *current* volatility state. Targets the
    Phase-1 body over-coverage caused by a regime-blind static window.
    """
    rng = np.random.default_rng(seed)
    w = _recency_weights(N, half_life)

    def forecaster(history: np.ndarray, T: int, levels: np.ndarray) -> np.ndarray:
        r = np.asarray(history, dtype=float)
        if len(r) < N:
            raise ValueError("insufficient history for bootstrap window")
        window = r[-N:]
        idx = rng.choice(N, size=(n_sims, T), p=w)
        paths = np.cumsum(window[idx], axis=1)
        return _paths_to_quantiles(paths, levels)

    return forecaster


def bootstrap_stationary(
    N: int,
    mean_block: float,
    half_life: float | None = None,
    n_sims: int = 10_000,
    seed: int = 0,
):
    """Stationary (Politis-Romano) block bootstrap over the last ``N`` returns.

    Each path is built by walking consecutive returns (wrapping at the window
    edge) and, with probability ``1/mean_block`` at each step, jumping to a new
    random start. Geometric block lengths (mean ``mean_block``) preserve
    short-range dependence and volatility clustering -> fatter multi-day tails.
    ``mean_block=1`` recovers the IID bootstrap. If ``half_life`` is given, the
    restart index is drawn with recency weights (age-weighted blocks: both
    upgrades at once).
    """
    rng = np.random.default_rng(seed)
    p_restart = 1.0 / mean_block
    w = None if half_life is None else _recency_weights(N, half_life)

    def _starts(size: int) -> np.ndarray:
        return rng.integers(0, N, size=size) if w is None else rng.choice(N, size=size, p=w)

    def forecaster(history: np.ndarray, T: int, levels: np.ndarray) -> np.ndarray:
        r = np.asarray(history, dtype=float)
        if len(r) < N:
            raise ValueError("insufficient history for bootstrap window")
        window = r[-N:]
        draws = np.empty((n_sims, T))
        pos = _starts(n_sims)
        for t in range(T):
            draws[:, t] = window[pos]
            cont = rng.random(n_sims) >= p_restart
            new = _starts(n_sims)
            pos = np.where(cont, (pos + 1) % N, new)
        paths = np.cumsum(draws, axis=1)
        return _paths_to_quantiles(paths, levels)

    return forecaster


def bootstrap_vol_scaled(
    N_shape: int,
    B_vol: int,
    alpha: float = 1.0,
    half_life: float | None = None,
    n_sims: int = 10_000,
    seed: int = 0,
):
    """Long-window bootstrap for tail *shape*, rescaled to *current* volatility.

    Motivated by the Phase-3 finding that tail breaches concentrate where recent
    volatility diverges from the window. A long ``N_shape`` window supplies the
    fat-tailed return *shape* (it contains crashes); the envelope is then scaled
    by ``(sigma_recent / sigma_window) ** alpha`` where ``sigma_recent`` is the
    std of the last ``B_vol`` returns. ``alpha=0`` recovers the plain bootstrap;
    ``alpha=1`` fully rescales to the recent regime. ``half_life`` optionally
    age-weights the shape resampling.
    """
    rng = np.random.default_rng(seed)
    w = None if half_life is None else _recency_weights(N_shape, half_life)

    def forecaster(history: np.ndarray, T: int, levels: np.ndarray) -> np.ndarray:
        r = np.asarray(history, dtype=float)
        if len(r) < N_shape:
            raise ValueError("insufficient history for shape window")
        window = r[-N_shape:]
        window = window - window.mean()  # drift pinned to zero
        sigma_win = window.std()
        sigma_recent = r[-B_vol:].std()
        scale = 1.0 if sigma_win == 0 else (sigma_recent / sigma_win) ** alpha
        if w is None:
            idx = rng.integers(0, N_shape, size=(n_sims, T))
        else:
            idx = rng.choice(N_shape, size=(n_sims, T), p=w)
        paths = np.cumsum(window[idx], axis=1)
        return scale * _paths_to_quantiles(paths, levels)

    return forecaster


def _gjr_innovation(ret: np.ndarray, gamma: float) -> np.ndarray:
    """Squared-return innovation with GJR leverage: down moves weighted (1+gamma)."""
    return ret * ret * (1.0 + gamma * (ret < 0.0))


def _ewma_vol_series(r: np.ndarray, lam: float, burn: int, gamma: float = 0.0) -> np.ndarray:
    """Causal (GJR-)EWMA conditional-variance forecast for each day.

    ``sig2[s]`` is the variance forecast for return ``r[s]`` using only returns
    before ``s``. Seeded at ``burn`` with the sample variance of ``r[:burn]``.
    ``gamma>0`` adds GJR leverage: a negative previous return raises variance by
    the factor ``(1+gamma)``. Entries before ``burn`` are the seed (unused).
    """
    sig2 = np.empty(len(r))
    seed = float(np.var(r[:burn]))
    sig2[:burn] = seed
    for s in range(burn, len(r)):
        innov = r[s - 1] * r[s - 1] * (1.0 + gamma * (r[s - 1] < 0.0))
        sig2[s] = lam * sig2[s - 1] + (1.0 - lam) * innov
    return sig2


def fhs_ewma(
    lam: float = 0.94,
    burn: int = 100,
    gamma: float = 0.0,
    n_sims: int = 10_000,
    seed: int = 0,
):
    """Filtered Historical Simulation with a (GJR-)EWMA volatility filter.

    Standardise each historical return by its causal conditional vol to get
    homoscedastic, fat-tailed residuals ``z``; bootstrap those and re-inflate by
    a volatility path that is propagated forward through the same recursion using
    the *simulated* returns. This reproduces volatility clustering in the forward
    paths (a large drawn shock raises subsequent vol), the mechanism that fattens
    multi-day tails. ``gamma>0`` adds GJR leverage so *down* shocks raise vol more
    -- both in the historical filter and in the forward propagation. Drift pinned
    to zero (residual pool demeaned and ensemble recentred).
    """
    rng = np.random.default_rng(seed)

    def forecaster(history: np.ndarray, T: int, levels: np.ndarray) -> np.ndarray:
        r = np.asarray(history, dtype=float)
        if len(r) < burn + 50:
            raise ValueError("insufficient history for FHS burn-in")
        sig2 = _ewma_vol_series(r, lam, burn, gamma)
        z = r[burn:] / np.sqrt(sig2[burn:])
        z = z - z.mean()  # drift pinned to zero
        # current one-step-ahead variance forecast for t+1
        var_next = lam * sig2[-1] + (1.0 - lam) * _gjr_innovation(r[-1:], gamma)[0]

        var = np.full(n_sims, var_next)
        cum = np.zeros(n_sims)
        paths = np.empty((n_sims, T))
        for h in range(T):
            zt = z[rng.integers(0, len(z), size=n_sims)]
            ret = np.sqrt(var) * zt
            cum = cum + ret
            paths[:, h] = cum
            var = lam * var + (1.0 - lam) * _gjr_innovation(ret, gamma)  # leverage
        return _paths_to_quantiles(paths, levels)

    return forecaster


_HL = (3.0, 10.0, 40.0, 10.0)  # halflives: ewma short/mid/long, leverage-down


def _expert_var_features(r: np.ndarray, hl=_HL) -> np.ndarray:
    """Causal expert-variance features F[t] (uses only r[:t]) -> predicts r[t]^2.

    Columns: [const, ewma_var_short, ewma_var_mid, ewma_var_long, long_run_mean,
    leverage_down_var]. This is the `haf` volatility expert stack in variance
    space: a fixed signed combination of these forecasts conditional variance far
    better than a single EWMA at multi-day horizons.
    """
    r = np.asarray(r, float)
    r2 = r * r
    n = len(r)

    def ewma(h):
        a = 1.0 - np.exp(-np.log(2) / h)
        v = np.empty(n); v[0] = r2[0]
        for i in range(1, n):
            v[i] = (1 - a) * v[i - 1] + a * r2[i - 1]
        return v

    v_s, v_m, v_l = ewma(hl[0]), ewma(hl[1]), ewma(hl[2])
    csum = np.concatenate([[0.0], np.cumsum(r2)])
    lm = np.empty(n); lm[0] = r2[0]
    lm[1:] = csum[1:n] / np.arange(1, n)               # mean of r2[:i]
    dn = r2 * 2.0 * (r < 0)
    a = 1.0 - np.exp(-np.log(2) / hl[3])
    vlev = np.empty(n); vlev[0] = dn[0]
    for i in range(1, n):
        vlev[i] = (1 - a) * vlev[i - 1] + a * dn[i - 1]
    return np.column_stack([np.ones(n), v_s, v_m, v_l, lm, vlev])


def fit_stack_coeffs(r: np.ndarray, hl=_HL, burn: int = 60) -> np.ndarray:
    """Least-squares coefficients predicting r[t]^2 from the expert-variance stack."""
    F = _expert_var_features(r, hl)
    y = np.asarray(r, float) ** 2
    coeffs, *_ = np.linalg.lstsq(F[burn:], y[burn:], rcond=None)
    return coeffs


def fhs_stack(coeffs, hl=_HL, burn: int = 60, n_sims: int = 10_000, seed: int = 0):
    """FHS whose conditional-vol filter is the haf expert stack (not a single EWMA).

    Standardise history by the stack's conditional vol, bootstrap the residuals,
    and propagate every expert variance forward through its own recursion on the
    simulated returns -- recombining via the fixed ``coeffs`` at each step. This is
    the drop-in upgrade of ``fhs_ewma``'s vol filter to the multi-expert stack.
    """
    coeffs = np.asarray(coeffs, float)
    a_s = 1 - np.exp(-np.log(2) / hl[0]); a_m = 1 - np.exp(-np.log(2) / hl[1])
    a_l = 1 - np.exp(-np.log(2) / hl[2]); a_v = 1 - np.exp(-np.log(2) / hl[3])
    rng = np.random.default_rng(seed)

    def _pv(vs, vm, vl, lm, vlev):
        return np.maximum(coeffs[0] + coeffs[1] * vs + coeffs[2] * vm
                          + coeffs[3] * vl + coeffs[4] * lm + coeffs[5] * vlev, 1e-10)

    def forecaster(history, T, levels):
        r = np.asarray(history, float)
        if len(r) < burn + 50:
            raise ValueError("insufficient history for stack FHS")
        F = _expert_var_features(r, hl)
        sig1 = np.sqrt(_pv(F[:, 1], F[:, 2], F[:, 3], F[:, 4], F[:, 5]))
        z = r[burn:] / sig1[burn:]
        z = z - z.mean()
        # advance expert states by the final observed return to forecast t+1
        last2 = r[-1] ** 2
        vs = np.full(n_sims, (1 - a_s) * F[-1, 1] + a_s * last2)
        vm = np.full(n_sims, (1 - a_m) * F[-1, 2] + a_m * last2)
        vl = np.full(n_sims, (1 - a_l) * F[-1, 3] + a_l * last2)
        vlev = np.full(n_sims, (1 - a_v) * F[-1, 5] + a_v * (last2 * 2 * (r[-1] < 0)))
        lm = float(F[-1, 4])
        cum = np.zeros(n_sims); paths = np.empty((n_sims, T))
        for h in range(T):
            sig = np.sqrt(_pv(vs, vm, vl, lm, vlev))
            ret = sig * z[rng.integers(0, len(z), size=n_sims)]
            cum = cum + ret; paths[:, h] = cum
            r2 = ret * ret
            vs = (1 - a_s) * vs + a_s * r2
            vm = (1 - a_m) * vm + a_m * r2
            vl = (1 - a_l) * vl + a_l * r2
            vlev = (1 - a_v) * vlev + a_v * (r2 * 2 * (ret < 0))
        return _paths_to_quantiles(paths, levels)

    return forecaster


def combine_average(forecasters: list):
    """Combine forecasters by averaging their predictive quantiles (equal weight).

    Averaging quantiles (a horizontal mixture / Vincentisation) keeps the result
    a valid non-decreasing quantile function and blends the history lengths.
    """

    def forecaster(history: np.ndarray, T: int, levels: np.ndarray) -> np.ndarray:
        qs = [f(history, T, levels) for f in forecasters]
        return np.mean(qs, axis=0)

    return forecaster


def empirical_scaled(N: int):
    """Empirical baseline: quantiles of the last ``N`` demeaned returns, sqrt-scaled.

    Takes the last ``N`` daily returns, removes their mean (drift pinned to
    zero), reads off empirical quantiles, and scales by ``sqrt(h)`` for the
    ``h``-day horizon (the iid random-walk spread law). No distributional
    assumption on shape, but still assumes iid via the sqrt-scaling.
    """

    def forecaster(history: np.ndarray, T: int, levels: np.ndarray) -> np.ndarray:
        r = np.asarray(history, dtype=float)
        if len(r) < N:
            raise ValueError("insufficient history for empirical window")
        window = r[-N:]
        window = window - window.mean()  # drift pinned to zero
        q1 = np.quantile(window, levels)  # one-day quantiles, shape (L,)
        h = np.arange(1, T + 1, dtype=float)  # (T,)
        return np.outer(q1, np.sqrt(h))  # (L, T)

    return forecaster
