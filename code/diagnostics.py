"""diagnostics.py — rank-normalised split R-hat and bulk / tail ESS (Vehtari et al. 2021) in JAX,
so the convergence check runs on the GPU for a whole batch of replications. Matches
arviz.rhat(method="rank") and arviz.ess(method="bulk" / "tail") (checked in tests/test_diag.py).

All functions take x of shape (chains, draws) and are vmapped over parameters and replications.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri


def _split(x):
    c, n = x.shape
    h = n // 2
    return jnp.concatenate([x[:, :h], x[:, n - h:]], 0)


def _ranks(x):
    """Average ranks (ties get the mean rank, as scipy.stats.rankdata(method='average'))."""
    flat = x.reshape(-1)
    srt = jnp.sort(flat)
    lo = jnp.searchsorted(srt, flat, side="left")
    hi = jnp.searchsorted(srt, flat, side="right")
    r = (lo + hi + 1) / 2.0
    return r.reshape(x.shape)


def _z(x):
    n = x.size
    return ndtri((_ranks(x) - 0.375) / (n + 0.25))


def _rhat_basic(x):
    m, n = x.shape
    cm = x.mean(1)
    b = n * cm.var(ddof=1)
    w = x.var(1, ddof=1).mean()
    vh = (n - 1) / n * w + b / n
    return jnp.sqrt(vh / w)


def rhat(x):
    """max(bulk, folded) rank-normalised split R-hat."""
    xs = _split(x)
    bulk = _rhat_basic(_z(xs))
    fold = jnp.abs(xs - jnp.median(xs))
    tail = _rhat_basic(_z(fold))
    return jnp.maximum(bulk, tail)


def _autocov(x):
    n = x.shape[-1]
    x = x - x.mean(-1, keepdims=True)
    f = jnp.fft.rfft(x, n=2 * n)
    ac = jnp.fft.irfft(f * jnp.conj(f), n=2 * n)[..., :n]
    return ac / n


def _ess(x):
    """ESS of (chains, draws) with Geyer's initial monotone sequence (ArviZ _ess)."""
    m, n = x.shape
    ac = _autocov(x)
    cm = x.mean(1)
    mean_var = ac[:, 0].mean() * n / (n - 1)
    var_plus = mean_var * (n - 1) / n + cm.var(ddof=1)
    rho = 1.0 - (mean_var - ac.mean(0)) / var_plus
    rho = rho.at[0].set(1.0)
    npair = (n - 1) // 2
    ev, od = rho[0:2 * npair:2], rho[1:2 * npair:2]
    P = ev + od
    # a pair is kept while every earlier pair sum was > 0 and its own sum is >= 0
    prev_pos = jnp.concatenate([jnp.array([True]), jnp.cumprod(P[:-1] > 0).astype(bool)])
    keep = prev_pos & (P >= 0)
    keep = jnp.cumprod(keep).astype(bool)
    nkeep = keep.sum()
    Pm = jax.lax.cummin(jnp.where(keep, P, jnp.inf))            # initial monotone sequence
    Pm = jnp.where(keep, Pm, 0.0)
    # first dropped pair: ArviZ adds its even term when positive
    idx = jnp.clip(nkeep, 0, npair - 1)
    extra = jnp.where((nkeep < npair) & (ev[idx] > 0), ev[idx], 0.0)
    tau = -1.0 + 2.0 * Pm.sum() + extra
    total = m * n
    tau = jnp.maximum(tau, 1.0 / jnp.log10(total))
    return total / tau


def ess_bulk(x):
    return _ess(_z(_split(x)))


def ess_tail(x):
    xs = _split(x)
    q05, q95 = jnp.quantile(x, 0.05), jnp.quantile(x, 0.95)
    e1 = _ess(_split((x <= q05).astype(x.dtype)))
    e2 = _ess(_split((x <= q95).astype(x.dtype)))
    return jnp.minimum(e1, e2)


def _param_stats(x):
    return rhat(x), ess_bulk(x), ess_tail(x)


# x: (chains, draws, P) -> three (P,) vectors
def batch_stats(x):
    xt = jnp.moveaxis(x, -1, 0)
    r, eb, et = jax.vmap(_param_stats)(xt)
    return r, eb, et


def rhat_only(x):
    return jax.vmap(rhat)(jnp.moveaxis(x, -1, 0))
