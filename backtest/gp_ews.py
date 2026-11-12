"""gp_ews — gamma–Poisson early-warning core.

Self-contained implementation of the model, used by the backtest on public trade data (Section 5.5;
backtest_multi.py); the simulation studies use the package in code/.
Implements: the M0/M1/M2 models of Table 3, NUTS fitting, particle
propagation between refits (Algorithm 1, step 4), the two-channel alarm
(p with cost thresholds, q with bootstrap-calibrated ARL0), the simulation
DGP of Table 4, and evaluation metrics.
"""
import numpy as np
import jax, jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median, init_to_uniform

numpyro.set_host_device_count(2)
PI_LEVELS = (0.048, 0.167, 0.5)          # cost ratios 1:20, 1:5, 1:1
RHO_SHIFT = 1.5                          # level-shift factor for q


# ------------------------------------------------------------------ model (M2; M0/M1 by flags)
def m2_model(N, X, Z, group, T, mu0_log, variant="M2"):
    """N: (I,T) counts; X: (I,T,K) indicators already lagged/standardised;
    Z: (I,M) exposure measurements centred within group; group: (I,) ints."""
    I, K, M, G = N.shape[0], X.shape[2], Z.shape[1], int(group.max()) + 1
    beta = numpyro.sample("beta", dist.Normal(0, 1).expand([K]))
    # exposure (18)-(19): equal prior weights when no AHP elicitation is supplied
    kap = numpyro.sample("kappa_w", dist.HalfNormal(1.0))
    e = numpyro.sample("e_w", dist.Normal(0, 0.3).expand([M]))
    w = kap * (jnp.ones(M) / M + e)
    tauC = numpyro.sample("tau_C", dist.HalfNormal(0.5))
    u_raw = numpyro.sample("u_raw", dist.Normal(0, 1).expand([I]))
    u = tauC * (u_raw - u_raw.mean())
    logC = Z @ w + u
    # dispersion (20): hierarchical across groups; collapses cleanly at G=1
    if variant == "M0":
        alpha_g = jnp.full(G, 1e4)
    elif variant == "M1":
        alpha_g = jnp.full(G, numpyro.sample("alpha1", dist.LogNormal(jnp.log(2.0), 1.0)))
    else:
        m_a = numpyro.sample("m_alpha", dist.Normal(jnp.log(2.0), 1.0))
        t_a = numpyro.sample("tau_alpha", dist.HalfNormal(0.5))
        a_raw = numpyro.sample("a_raw", dist.Normal(0, 1).expand([G]))
        alpha_g = jnp.exp(m_a + t_a * a_raw)
    numpyro.deterministic("alpha_g", alpha_g)
    # trend (12)-(13), non-centred; deterministic log-linear for M0/M1
    m1 = numpyro.sample("m1", dist.Normal(mu0_log, 1.0).expand([G]))
    d1 = numpyro.sample("d1", dist.Normal(0, 0.05).expand([G]))
    if variant in ("M0", "M1"):
        t_idx = jnp.arange(T)
        m_t = m1[:, None] + d1[:, None] * t_idx[None, :]
        dlt = jnp.broadcast_to(d1[:, None], (G, T))
        sig_e = jnp.zeros(()); tau_z = jnp.zeros(())
    else:
        sig_e = numpyro.sample("sigma_eta", dist.HalfNormal(0.1))
        tau_z = numpyro.sample("tau_zeta", dist.HalfNormal(0.1))
        eta = numpyro.sample("eta_raw", dist.Normal(0, 1).expand([G, T - 1]))
        zet = numpyro.sample("zeta_raw", dist.Normal(0, 1).expand([G, T - 1]))
        dlt = jnp.concatenate([d1[:, None], d1[:, None] + tau_z * jnp.cumsum(zet, 1)], 1)
        m_t = jnp.concatenate([m1[:, None],
                               m1[:, None] + jnp.cumsum(dlt[:, :-1] + sig_e * eta, 1)], 1)
    numpyro.deterministic("m_t", m_t); numpyro.deterministic("delta_t", dlt)
    numpyro.deterministic("logC", logC)
    logR = m_t[group] + jnp.einsum("itk,k->it", X, beta)          # (11)
    mu = jnp.exp(jnp.clip(logR + logC[:, None], -20.0, 10.0))
    numpyro.sample("N", dist.GammaPoisson(alpha_g[group][:, None],
                                          alpha_g[group][:, None] / mu), obs=N)


def fit(N, X, Z, group, mu0_log, variant="M2", warmup=800, samples=800, chains=2, seed=0, init="median"):
    """NUTS fit; `init`: "median" starts each chain near the prior medians (the median of 15 prior draws per site, as
    in Supplementary Note 1), "uniform" uses NumPyro's default (uniform in (-2, 2) on the unconstrained scale), which
    can start a chain on the plateau where the rate is clipped at exp(10) and leave it frozen there."""
    strat = init_to_median(num_samples=15) if init == "median" else init_to_uniform()
    kern = NUTS(m2_model, max_tree_depth=8, target_accept_prob=0.9, init_strategy=strat)
    mc = MCMC(kern, num_warmup=warmup, num_samples=samples, num_chains=chains,
              chain_method="sequential", progress_bar=False)
    mc.run(jax.random.PRNGKey(seed), jnp.array(N), jnp.array(X), jnp.array(Z),
           np.asarray(group), N.shape[1], float(mu0_log), variant)
    s = {k: np.asarray(v) for k, v in mc.get_samples().items()}
    extra = mc.get_extra_fields() if hasattr(mc, "get_extra_fields") else {}
    return s


def draw_pack(s, group):
    """Flatten posterior draws into the per-particle state used downstream."""
    S = s["beta"].shape[0]
    return dict(S=S, beta=s["beta"], logC=s["logC"], alpha=s["alpha_g"],
                m=s["m_t"][:, :, -1], d=s["delta_t"][:, :, -1],
                m_path=s["m_t"], sig=s.get("sigma_eta", np.zeros(S)),
                tau=s.get("tau_zeta", np.zeros(S)), group=np.asarray(group))


# ------------------------------------------------------------------ probabilities and propagation
def sri(mu, a):
    return (1.0 + mu / a) ** (-a)


def propagate(pack, N_new, X_new, m_base, rng, ess_frac=0.5):
    """One-month particle update: predict, reweight on observed counts, resample.
    N_new: (I,), X_new: (I,K) = indicators dated this month (used next month)."""
    S = pack["S"]; g = pack["group"]
    pack["m"] = pack["m"] + pack["d"] + pack["sig"][:, None] * rng.standard_normal((S, pack["m"].shape[1]))
    pack["d"] = pack["d"] + pack["tau"][:, None] * rng.standard_normal((S, pack["d"].shape[1]))
    mu = np.exp(pack["m"][:, g] + (pack["x_prev"] @ pack["beta"].T).T + pack["logC"])
    a = pack["alpha"][:, g]
    from scipy.stats import nbinom
    logw = nbinom.logpmf(N_new[None, :], a, a / (a + mu)).sum(1)
    logw -= logw.max(); w = np.exp(logw); w /= w.sum()
    ess = 1.0 / (w ** 2).sum()
    if ess < ess_frac * S:
        idx = systematic(w, rng)
        for k in ("m", "d", "beta", "logC", "alpha", "sig", "tau"):
            pack[k] = pack[k][idx]
        w = np.full(S, 1.0 / S)
    pack["w"] = w; pack["x_prev"] = X_new
    q = (w[:, None] * ((pack["m"] - m_base[None, :]) > np.log(RHO_SHIFT))).sum(0)
    return np.asarray(q)


def systematic(w, rng):
    S = len(w); pos = (rng.random() + np.arange(S)) / S
    return np.searchsorted(np.cumsum(w), pos).clip(0, S - 1)


def weighted_p(pack, h=3):
    S = pack["S"]; g = pack["group"]; w = pack.get("w", np.full(S, 1 / S))
    keep = np.ones((S, pack["logC"].shape[1]))
    for step in range(1, h + 1):
        mu = np.exp(pack["m"][:, g] + step * pack["d"][:, g] + (pack["x_prev"] @ pack["beta"].T).T + pack["logC"])
        keep *= sri(mu, pack["alpha"][:, g])
    return 1.0 - (w[:, None] * keep).sum(0)


def alarm_levels(p_series, pis=PI_LEVELS):
    """2-of-2 persistence; p_series: (T,I) -> levels (T,I)."""
    T, I = p_series.shape
    lev = np.zeros((T, I), int)
    for k, pi in enumerate(pis, 1):
        hit = p_series >= pi
        pers = hit & np.vstack([np.zeros((1, I), bool), hit[:-1]])
        lev = np.maximum(lev, k * pers)
    return lev


def calibrate_q(pack0, x_hold, months, target_arl=24, B=200, grid=None, seed=1):
    """Bootstrap ARL0 calibration: simulate stable paths (delta=0) from the posterior
    at the first origin, run the same filter, choose the smallest threshold whose
    2-of-2 alarm rate implies ARL0 >= target."""
    rng = np.random.default_rng(seed)
    grid = grid if grid is not None else np.arange(0.50, 0.96, 0.05)
    S = pack0["S"]; g = pack0["group"]; G = pack0["m"].shape[1]
    qs = np.zeros((B, months))
    for b in range(B):
        j = rng.integers(S)
        m = pack0["m"][j].copy(); sig = pack0["sig"][j]
        beta, logC, a = pack0["beta"][j], pack0["logC"][j], pack0["alpha"][j]
        sub = {k: pack0[k].copy() for k in ("m", "d", "beta", "logC", "alpha", "sig", "tau")}
        sub.update(S=S, group=g, x_prev=x_hold.copy(), w=np.full(S, 1 / S))
        m_base = pack0["m"].mean(0)
        for t in range(months):
            m = m + sig * rng.standard_normal(G)          # truth: stable level walk
            mu = np.exp(m[g] + x_hold @ beta + logC)
            N = rng.negative_binomial(a[g], a[g] / (a[g] + mu))
            qs[b, t] = propagate(sub, N, x_hold, m_base, rng).max()
    for c in grid:
        hit = qs >= c
        al = (hit[:, 1:] & hit[:, :-1]).sum()
        arl0 = qs.size / max(al, 1)
        if arl0 >= target_arl:
            return float(c), float(arl0)
    return float(grid[-1]), float(qs.size / max((qs >= grid[-1]).sum(), 1))


# ------------------------------------------------------------------ simulation DGP (Table 4)
def simulate(I=30, T=120, G=3, mu_bar=0.2, alpha=2.0, regime="gradual", K=3,
             rho_x=0.8, beta=(0.4, 0.3, 0.2), sigma_u=0.3, meas_err=0.0,
             hawkes=None, seed=0, cold_frac=0.0, miss=0.05):
    rng = np.random.default_rng(seed)
    group = np.arange(I) % G
    X = np.zeros((I, T, K))
    for t in range(1, T):
        X[:, t] = rho_x * X[:, t - 1] + np.sqrt(1 - rho_x ** 2) * rng.standard_normal((I, K))
    Z = rng.standard_normal((I, 2))
    logC = 0.5 * Z[:, 0] + 0.3 * Z[:, 1] + sigma_u * rng.standard_normal(I)
    logC -= np.array([logC[group == g].mean() for g in range(G)])[group]
    Zobs = Z + meas_err * rng.standard_normal(Z.shape)
    Zobs -= np.array([[Zobs[group == g, m].mean() for m in range(2)] for g in range(G)])[group]
    t0 = T // 2
    m = np.full((G, T), np.log(mu_bar))
    if regime == "gradual":
        for t in range(t0, T): m[:, t] = m[:, t - 1] + 0.02
    elif regime == "abrupt":
        m[:, t0:] += np.log(3.0)
    logR = m[group] + np.einsum("itk,k->it", X[:, :, :], np.array(beta))
    # indicators enter lagged: shift X by one when building mu
    Xl = np.concatenate([np.zeros((I, 1, K)), X[:, :-1]], 1)
    logR = m[group] + np.einsum("itk,k->it", Xl, np.array(beta))
    mu = np.exp(logR + logC[:, None])
    if hawkes is None:
        lam = rng.gamma(alpha, 1 / alpha, size=(I, T)) if np.isfinite(alpha) else 1.0
        N = rng.poisson(mu * lam)
    else:
        n_br, kD = hawkes
        N = hawkes_counts(mu, n_br, kD, rng)
    Xo = X.copy()
    Xo[rng.random((I, T, K)) < miss] = np.nan
    obs_start = np.zeros(I, int)
    if cold_frac > 0:
        cold = rng.choice(I, int(cold_frac * I), replace=False)
        obs_start[cold] = 3 * T // 4
    return dict(N=N, X=Xo, X_true=X, Z=Zobs, Z_true=Z, group=group, logC=logC,
                m=m, t0=t0, mu=mu, obs_start=obs_start)


def hawkes_counts(mu, n, kD, rng):
    I, T = mu.shape; b = kD / (1 - n); out = np.zeros((I, T), int)
    for i in range(I):
        nu = mu[i] * (1 - n)
        times = []
        parents = np.concatenate([np.full(rng.poisson(nu[t]), t + rng.random()) for t in range(T)]) \
            if T else np.array([])
        while parents.size:
            times.append(parents)
            kids = np.repeat(parents, rng.poisson(n, parents.size))
            kids = kids + rng.exponential(1 / b, kids.size)
            parents = kids[kids < T]
        if times:
            tt = np.concatenate(times); out[i] = np.bincount(tt.astype(int), minlength=T)[:T]
    return out


# ------------------------------------------------------------------ metrics
def recovery_metrics(s, truth):
    out = {}
    for name, tr in [("beta", truth.get("beta")), ("logC", truth.get("logC"))]:
        if tr is None: continue
        d = s[name]; lo, hi = np.percentile(d, [5, 95], 0); med = np.median(d, 0)
        out[name] = dict(rmse=float(np.sqrt(((med - tr) ** 2).mean())),
                         cover=float(((tr >= lo) & (tr <= hi)).mean()))
    a = s["alpha_g"]; lo, hi = np.percentile(a, [5, 95], 0)
    out["alpha"] = dict(lo=lo.tolist(), hi=hi.tolist(), med=np.median(a, 0).tolist())
    return out
