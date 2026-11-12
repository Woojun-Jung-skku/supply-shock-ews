"""pfilter.py — online part of Algorithm 1 in JAX (§4.4, §5.3).

Between refits the posterior draws are particles; each month they are propagated through the
trend (12)-(13), reweighted on the new counts and resampled when the effective sample size falls
below half. At every decision month t the filter emits
  p_it(3)      eq. (22), R_{i,t+s} drawn from trend extrapolation, x held at its last value
  p_ar_it(3)   the same with an AR(1) forecast of the indicators (robustness variant, §4.4)
  q_t^g        eq. (24), P(m_t^g - m_base^g > log rho)
  pmf_it(n)    posterior predictive pmf of N_{i,t+1}, n = 0..NMAX (conformal sets, §4.3)
  mu_hat, alpha_hat, logmu_hat   plug-in quantities for the sigma-rule and the severity model
A pack is a dict of particle arrays: m (S,G), d (S,G), beta (S,K), logC (S,I), alpha (S,I),
sig (S,), tau (S,), rho (S,K). `poisson=True` for M0 (alpha ignored).
"""
import math
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
import config as C

LOG_RHO = math.log(C.RHO_SHIFT)


def nb_logpmf(n, mu, a, poisson):
    if poisson:
        return n * jnp.log(mu) - mu - gammaln(n + 1.0)
    return (gammaln(n + a) - gammaln(a) - gammaln(n + 1.0) + a * jnp.log(a / (a + mu))
            + n * jnp.log(mu / (a + mu)))


def _mu(pack, m, x, group):
    logmu = m[:, group] + jnp.einsum("ik,sk->si", x, pack["beta"]) + pack["logC"]
    return jnp.exp(jnp.clip(logmu, *C.LOGMU_CLIP)), logmu


def _systematic(key, w):
    S = w.shape[0]
    pos = (jax.random.uniform(key) + jnp.arange(S)) / S
    return jnp.clip(jnp.searchsorted(jnp.cumsum(w), pos), 0, S - 1)


def _step(pack, logw, key, n_t, mask_t, x_prev, group, poisson):
    """Advance one month and reweight on counts n_t (I,) observed with mask_t (I,)."""
    S, G = pack["m"].shape
    k1, k2, k3 = jax.random.split(key, 3)
    m = pack["m"] + pack["d"] + pack["sig"][:, None] * jax.random.normal(k1, (S, G))
    d = pack["d"] + pack["tau"][:, None] * jax.random.normal(k2, (S, G))
    mu, _ = _mu(pack, m, x_prev, group)
    ll = jnp.where(mask_t[None, :], nb_logpmf(n_t[None, :], mu, pack["alpha"], poisson), 0.0).sum(1)
    ll = jnp.where(jnp.isfinite(ll), ll, -jnp.inf)
    logw = logw + ll
    logw = logw - jax.scipy.special.logsumexp(logw)
    logw = jnp.where(jnp.isfinite(logw).any(), logw, jnp.full(S, -jnp.log(S)))   # no usable particle
    w = jnp.exp(logw)
    ess = 1.0 / (w ** 2).sum()
    new = dict(pack, m=m, d=d)
    idx = _systematic(k3, w)
    res = {k: v[idx] for k, v in new.items()}
    do = ess < 0.5 * S
    new = jax.tree.map(lambda a, b: jnp.where(do, a, b), res, new)
    logw = jnp.where(do, jnp.full(S, -jnp.log(S)), logw)
    return new, logw


def _decide(pack, logw, key, x_t, group, m_base, poisson, h=C.HORIZON, nmax=C.CONF_NMAX):
    S, G = pack["m"].shape
    w = jnp.exp(logw)
    ks = jax.random.split(key, 2 * h)
    m, d = pack["m"], pack["d"]
    keep = jnp.ones((S, x_t.shape[0]))
    keep_ar = jnp.ones((S, x_t.shape[0]))
    for s in range(h):
        m = m + d + pack["sig"][:, None] * jax.random.normal(ks[2 * s], (S, G))
        d = d + pack["tau"][:, None] * jax.random.normal(ks[2 * s + 1], (S, G))
        mu, logmu = _mu(pack, m, x_t, group)                     # x held at its last value
        x_ar = x_t[None] * (pack["rho"] ** s)[:, None, :]        # E[x_{t+s} | x_t] = rho^s x_t
        logmu_ar = m[:, group] + jnp.einsum("sik,sk->si", x_ar, pack["beta"]) + pack["logC"]
        mu_ar = jnp.exp(jnp.clip(logmu_ar, *C.LOGMU_CLIP))
        if poisson:
            keep, keep_ar = keep * jnp.exp(-mu), keep_ar * jnp.exp(-mu_ar)
        else:
            a = pack["alpha"]
            keep = keep * (1.0 + mu / a) ** (-a)
            keep_ar = keep_ar * (1.0 + mu_ar / a) ** (-a)
        if s == 0:
            mu1, logmu1 = mu, logmu
    p = 1.0 - w @ keep
    p_ar = 1.0 - w @ keep_ar
    q = w @ ((pack["m"] - m_base[None, :]) > LOG_RHO).astype(jnp.float32)
    n = jnp.arange(nmax + 1.0)
    lp = nb_logpmf(n[None, None, :], mu1[:, :, None], pack["alpha"][:, :, None], poisson)
    pmf = jnp.einsum("s,sin->in", w, jnp.exp(lp))
    mu_hat = w @ mu1
    a_hat = w @ pack["alpha"] if not poisson else jnp.full(x_t.shape[0], jnp.inf)
    return dict(p=p, p_ar=p_ar, q=q, pmf=pmf, mu_hat=mu_hat, alpha_hat=a_hat, logmu_hat=w @ logmu1)


def run_segment(pack, key, n_seg, mask_seg, x_seg, group, m_base, poisson):
    """Decisions at months t_o, t_o+1, ..., t_o+L-1 (L = n_seg.shape[0] + 1).
    n_seg (L-1, I): counts of months t_o+1..; x_seg (L, I, K): indicators of months t_o..t_o+L-1."""
    S = pack["m"].shape[0]
    logw = jnp.full(S, -jnp.log(S))
    k0, key = jax.random.split(key)
    out0 = _decide(pack, logw, k0, x_seg[0], group, m_base, poisson)

    def body(carry, inp):
        pk, lw, k = carry
        n_t, mk, x_prev, x_t = inp
        k, k1, k2 = jax.random.split(k, 3)
        pk, lw = _step(pk, lw, k1, n_t, mk, x_prev, group, poisson)
        return (pk, lw, k), _decide(pk, lw, k2, x_t, group, m_base, poisson)
    if n_seg.shape[0] == 0:
        return jax.tree.map(lambda a: a[None], out0)
    _, outs = jax.lax.scan(body, (pack, logw, key), (n_seg, mask_seg, x_seg[:-1], x_seg[1:]))
    return jax.tree.map(lambda a, b: jnp.concatenate([a[None], b], 0), out0, outs)


_SEG = {}


def segment_batch(poisson):
    """run_segment vmapped over replications (pack, key, n_seg, mask_seg, x_seg, m_base batched)."""
    if poisson not in _SEG:
        f = partial(run_segment, poisson=poisson)
        _SEG[poisson] = jax.jit(jax.vmap(lambda pk, k, n, mk, x, g, mb: f(pk, k, n, mk, x, g, mb),
                                         in_axes=(0, 0, 0, 0, 0, None, 0)))
    return _SEG[poisson]


# ------------------------------------------------------------------ bootstrap calibration of q (§4.4)
def _boot_path(pack, k, x_hold, group, m_base, months, poisson):
    """One stable-regime path from key k (see boot_q): returns q (months, G)."""
    S, G = pack["m"].shape
    kj, ks = jax.random.split(k)
    j = jax.random.randint(kj, (), 0, S)
    m_true = pack["m"][j]
    sig_j, beta_j, logC_j, a_j = pack["sig"][j], pack["beta"][j], pack["logC"][j], pack["alpha"][j]
    lw = jnp.full(S, -jnp.log(S))

    def body(carry, k):
        pk, lw, mt = carry
        k1, k2, k3, k4 = jax.random.split(k, 4)
        if C.BOOT_LEVEL_WALK:
            mt = mt + sig_j * jax.random.normal(k1, (G,))
        mu = jnp.exp(jnp.clip(mt[group] + x_hold @ beta_j + logC_j, *C.LOGMU_CLIP))
        if poisson:
            n = jax.random.poisson(k2, mu).astype(jnp.float32)
        else:
            lam = jax.random.gamma(k2, a_j) / a_j
            n = jax.random.poisson(k3, mu * lam).astype(jnp.float32)
        pk, lw = _step(pk, lw, k4, n, jnp.ones_like(n, bool), x_hold, group, poisson)
        q = jnp.exp(lw) @ ((pk["m"] - m_base[None, :]) > LOG_RHO).astype(jnp.float32)
        return (pk, lw, mt), q
    _, qs = jax.lax.scan(body, (pack, lw, m_true), jax.random.split(ks, months))
    return qs


@partial(jax.jit, static_argnames=("months", "paths", "poisson"))
def boot_q(pack, key, x_hold, group, m_base, months, paths, poisson):
    """Stable-regime paths from the posterior at the first origin: pick a draw, hold its level fixed
    (stable regime of Table 4: delta = 0 and no level shift; with config.BOOT_LEVEL_WALK the level
    also walks with that draw's sigma_eta), draw counts, run the same filter; returns q (paths, months, G)."""
    one = lambda k: _boot_path(pack, k, x_hold, group, m_base, months, poisson)
    return jax.lax.map(one, jax.random.split(key, paths), batch_size=min(paths, 25))


@partial(jax.jit, static_argnames=("months", "poisson"))
def boot_q_keys(pack, keys, x_hold, group, m_base, months, poisson):
    """v1.4: boot_q for an explicit block of per-path keys, so that the bootstrap can be split into
    checkpointed blocks (segmented runs): boot_q(pack, key, ..., paths) equals the concatenation of
    boot_q_keys(pack, jax.random.split(key, paths)[a:b], ...) over consecutive blocks."""
    one = lambda k: _boot_path(pack, k, x_hold, group, m_base, months, poisson)
    return jax.lax.map(one, keys, batch_size=min(keys.shape[0], 25))


@partial(jax.jit, static_argnames=("months", "paths", "poisson"))
def boot_q_batch(packs, keys, x_hold, group, m_base, months, paths, poisson):
    """boot_q for a chunk of replications (leading axis of packs, keys, x_hold, m_base)."""
    return jax.vmap(lambda pk, k, x, mb: boot_q(pk, k, x, group, mb, months, paths, poisson))(packs, keys, x_hold, m_base)


def persistent(hit, r):
    """r-of-r persistence along axis 0 of a boolean array."""
    out = hit.copy()
    for k in range(1, r):
        sh = np.zeros_like(hit)
        sh[k:] = hit[:-k]
        out &= sh
    return out


def calibrate(qpaths, target=C.ARL0_TARGET, grid=C.Q_GRID, r0=C.PERSIST, rmax=C.PERSIST_MAX):
    """Smallest threshold (then stronger persistence) whose per-group ARL0 >= target.
    ARL0 = monitored group-months / alarm months. qpaths (paths, months, G)."""
    q = np.moveaxis(np.asarray(qpaths), 1, 0)                     # (months, paths, G)
    for r in range(r0, rmax + 1):
        for c in grid:
            al = persistent(q >= c, r)[r - 1:]
            arl0 = al.size / max(al.sum(), 1)
            if arl0 >= target:
                return dict(q_thr=float(c), q_persist=r, q_arl0_boot=round(float(arl0), 1), q_calibrated=1)
    return dict(q_thr=float(grid[-1]), q_persist=rmax, q_arl0_boot=round(float(arl0), 1), q_calibrated=0)
