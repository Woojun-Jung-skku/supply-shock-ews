"""dgp.py — simulation data-generating process (manuscript §5.1, Table 4; §3.4 and Supplementary Note 16 for Hawkes).

Time convention (used everywhere in this package, 0-based arrays):
  N[:, j]  counts of month j+1, j = 0..T-1
  X[:, s]  indicators observable at the end of month s, s = 0..T  (s = 0 is the pre-sample month)
  count N[:, j] depends on X[:, j] = x_{t-1} for month t = j+1  (eq. 11, one-period lead)
"""
import hashlib
import math
import numpy as np
from scipy.stats import norm
import config as C


def seed_of(*parts):
    """Stable 32-bit seed from any labels (cell id, replication, ...)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def _centre(v, group, G):
    means = np.array([v[group == g].mean(0) for g in range(G)])
    return v - means[group]


def hawkes_counts(mu, n, kd, rng, burn=24):
    """Monthly counts of an exponential-kernel Hawkes process (cluster representation,
    Hawkes & Oakes 1974). Immigrant rate nu_it = mu_it (1 - n) so that the stationary mean per
    month is mu_it; offspring per event ~ Poisson(n), delay ~ Exp(b), b = kd / (1 - n) (Delta = 1).
    A burn-in of `burn` months at the first month's rate removes the start-up deficit."""
    I, T = mu.shape
    if n <= 0:
        return rng.poisson(mu)
    b = kd / (1.0 - n)
    out = np.zeros((I, T), int)
    for i in range(I):
        rate = np.concatenate([np.full(burn, mu[i, 0]), mu[i]]) * (1.0 - n)
        k = rng.poisson(rate)
        gen = np.repeat(np.arange(T + burn), k) + rng.random(k.sum())
        events = [gen]
        while gen.size:
            kids = np.repeat(gen, rng.poisson(n, gen.size))
            gen = kids + rng.exponential(1.0 / b, kids.size)
            gen = gen[gen < T + burn]
            events.append(gen)
        tt = np.concatenate(events) - burn
        tt = tt[tt >= 0]
        out[i] = np.bincount(tt.astype(int), minlength=T)[:T]
    return out


def alpha_match(mu, n, kd):
    """Moment-matched dispersion (15) for window counts with mean mu (Delta = 1)."""
    if n <= 0:
        return math.inf
    g = 1.0 - (1.0 - math.exp(-kd)) / kd
    return mu * (1 - n) ** 2 / (n * (2 - n) * g)


def _trunc_lognormal(rng, size, m, s):
    u = rng.random(size) * norm.cdf((0.0 - m) / s)          # log S < 0  <=>  S < 1
    return np.exp(m + s * norm.ppf(np.clip(u, 1e-12, 1 - 1e-12)))


def _trunc_pareto(rng, size, a, xm):
    u = rng.random(size) * (1.0 - xm ** a)                   # F(s) = 1 - (xm/s)^a on [xm, 1]
    return xm / (1.0 - u) ** (1.0 / a)


def simulate(I, T, mu, alpha, regime, seed, G=C.G, K=C.K, meas_err=0.0, miss=C.MISS_DEFAULT,
             cold=0.0, hawkes=None, severity="lognormal", x_lag=1, frailty="gamma"):
    """One simulated panel. alpha: float (np.inf = Poisson) or a sequence of G values, one per group. hawkes: (n, kd) replaces the
    gamma-Poisson counts. x_lag: lag of the indicators in the truth (1 = eq. 11). frailty: 'gamma' (the multiplier of eq. 2) or
    'lognormal' (v1.25, B3: a log-normal multiplier with the same mean 1 and variance 1/alpha, Section 3.5)."""
    rng = np.random.default_rng(seed)
    group = np.arange(I) % G
    extra = x_lag - 1                                        # pre-sample months needed for the lag
    Xall = np.zeros((I, T + 1 + extra, K))
    Xall[:, 0] = rng.standard_normal((I, K))                 # stationary start, unit variance
    for s in range(1, T + 1 + extra):
        Xall[:, s] = C.RHO_X * Xall[:, s - 1] + math.sqrt(1 - C.RHO_X ** 2) * rng.standard_normal((I, K))
    X = Xall[:, extra:]                                      # months 0..T
    Xlag = Xall[:, :T]                                       # x_{t - x_lag} for months 1..T
    # exposure (Table 4): z standard normal, centred within group; u ~ N(0, 0.3^2) with sum-to-zero
    z = _centre(rng.standard_normal((I, 2)), group, G)
    u = _centre(C.SIGMA_U * rng.standard_normal(I), group, G)
    w = np.array(C.W_TRUE)
    logC = z @ w + u
    zobs = _centre(z + meas_err * rng.standard_normal(z.shape), group, G)
    # group trend (§5.1): base log mu, change from t0 = T/2
    t0 = T // 2                                              # first changed count column
    m = np.full((G, T), math.log(mu))
    delta = np.zeros((G, T))
    if regime == "gradual":
        m[:, t0:] += C.GRADUAL_SLOPE * np.arange(1, T - t0 + 1)
        delta[:, t0 - 1:] = C.GRADUAL_SLOPE
    elif regime == "abrupt":
        m[:, t0:] += C.ABRUPT_JUMP
    beta = np.array(C.BETA)
    mu_it = np.exp(m[group] + np.einsum("itk,k->it", Xlag, beta) + logC[:, None])
    if hawkes is not None:
        N = hawkes_counts(mu_it, hawkes[0], hawkes[1], rng)
        a_true = alpha_match(mu, *hawkes)
    elif np.ndim(alpha) == 0:
        if not np.isfinite(alpha):
            xi = 1.0
        elif frailty == "lognormal":                         # same mean and variance as the gamma multiplier
            s2 = math.log1p(1.0 / alpha)
            xi = np.exp(math.sqrt(s2) * rng.standard_normal((I, T)) - 0.5 * s2)
        else:
            xi = rng.gamma(alpha, 1.0 / alpha, size=(I, T))
        N = rng.poisson(mu_it * xi)
        a_true = alpha
    else:                                                    # v1.24 (B2): dispersion differs by group, alpha = (G,)
        ag = np.asarray(alpha, float)[group][:, None]
        xi = rng.gamma(ag, 1.0 / ag, size=(I, T))
        N = rng.poisson(mu_it * xi)
        a_true = np.asarray(alpha, float)
    # severity S given at least one shock (Table 4)
    if severity == "pareto":
        S = _trunc_pareto(rng, (I, T), *C.SEV_PARETO)
    else:
        S = _trunc_lognormal(rng, (I, T), *C.SEV_LOGNORMAL)
    S = np.where(N >= 1, S, np.nan)
    # missing indicators (MCAR) and cold start
    Xobs = X.copy()
    Xobs[rng.random(X.shape) < miss] = np.nan
    Nmask = np.ones((I, T), bool)
    cold_items = np.zeros(I, bool)
    if cold > 0:
        idx = rng.choice(I, int(round(cold * I)), replace=False)
        cold_items[idx] = True
        start = T - T // 4                                   # observed only in the last T/4 periods
        Nmask[idx, :start] = False
        Xobs[idx, :start] = np.nan                           # x unobserved before the item enters
    N = np.where(Nmask, N, 0)
    S = np.where(Nmask, S, np.nan)
    truth = dict(beta=beta, w=w, kappa_w=C.KAPPA_TRUE, logC=logC, alpha=a_true, m=m, delta=delta,
                 t0=t0, mu_bar=mu, cold_items=cold_items)
    return dict(N=N.astype(np.int32), Nmask=Nmask, X=Xobs, Z=zobs, group=group, S=S, truth=truth)


def locf(X):
    """Last observation carried forward along time, first missing values set to 0 (the stationary
    mean); used by the baselines, by the LOCF robustness arm and inside the particle filter."""
    X = X.copy()
    I, T1, K = X.shape
    last = np.zeros((I, K))
    for s in range(T1):
        nan = np.isnan(X[:, s])
        X[:, s] = np.where(nan, last, X[:, s])
        last = X[:, s]
    return X
