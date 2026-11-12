"""models.py — NumPyro models of §4.1 (eqs. 16-21), the model ladder of Table 3 and the severity
submodel (8)-(9). All data enter as one dict of arrays with shapes fixed within a batch, so that
replications — of one cell or of many cells with the same (I, T) — can be stacked and sampled
together (nuts_gpu.py).

Variants (argument `spec`):
  variant  "M2" (dynamic local linear trend) | "M0" (Poisson, deterministic log-linear trend,
           C = exp(w'z)) | "M1" (M0 + one pooled alpha')
  pooling  "hier" (eq. 20) | "none" (alpha'_i and u_i free) | "complete" (one alpha', u = 0)
  innov    "gauss" | "horseshoe" (regularised horseshoe scale mixture on the level innovations)
           | "horseshoe2" (the same prior with the half-Cauchy local scale written as z sqrt(nu),
           z ~ half-N(0, 1), nu ~ InverseGamma(1/2, 1/2); follow-up arm A6, followup/mac_arms.py)
  xmode    "latent" (missing indicators are latent with an AR(1) prior; Table 4) | "locf"

Information time (v1.2/v1.3): a fit "at origin t" uses the arrays of the full series padded to T
with every month after t masked out of the likelihood, of the AR(1) indicator prior and of the
trend innovations' influence — so that every origin has the same shapes and one compiled sampler.
The particle state at origin t is (m_t[..., t-1], delta_t[..., t-1]).
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import config as C
from dgp import locf

P = C.PRIOR


def mcap(I, T, K, miss, cold=0.0):
    """Fixed number of latent slots for missing indicators of a full series (padded; unused slots
    carry a N(0, 1) prior and touch nothing). Cold-start items have no indicators before month
    T - T // 4, which removes those slots (they are not latent, v1.1)."""
    n = I * (T + 1) * K
    start = T - T // 4
    cold_part = int(round(cold * I)) * start * K if cold > 0 else 0
    rest = n - cold_part
    return int(math.ceil(miss * rest + 8 * math.sqrt(miss * (1 - miss) * rest) + 16))


def innov_basis(T, t_obs=None):
    """Exact orthonormal split of a group's T-1 innovations into one 'mean' coordinate and T-2
    orthogonal ones (level: mean level over the observed months, a_s = (t_obs - s)/t_obs; slope:
    average slope, a_s = (t_obs - 1 - s)/(t_obs - 1)). Sampling s ~ N(0, 1), v ~ N(0, I) and setting
    innovations = a s + Q v leaves the model unchanged (INNOV_SPLIT)."""
    t_obs = T if t_obs is None else t_obs
    s = np.arange(1, T)
    out = []
    for a in (np.where(s < t_obs, (t_obs - s) / t_obs, 0.0),
              np.where(s < t_obs - 1, (t_obs - 1 - s) / max(t_obs - 1, 1), 0.0)):
        a = a / np.linalg.norm(a)
        q, _ = np.linalg.qr(np.column_stack([a, np.eye(T - 1)]))
        q = q[:, :T - 1]
        if q[:, 0] @ a < 0:
            q[:, 0] = -q[:, 0]
        out += [q[:, 0].astype(np.float32), q[:, 1:].astype(np.float32)]
    return out


def prep(sim, t_obs, xmode="latent", cap=None):
    """Model data for a fit at origin t_obs (months 1..t_obs enter the likelihood), arrays padded to
    the full length T of the series. cap = number of latent slots for missing indicators (fixed
    across the replications of a batch: models.mcap with the largest missingness of the batch)."""
    N, Nmask, X = sim["N"], sim["Nmask"].copy(), sim["X"]
    I, T = N.shape
    K = X.shape[2]
    Nmask[:, t_obs:] = False                                # information-time discipline
    d = dict(N=N.astype(np.float32), Nmask=Nmask, Z=sim["Z"].astype(np.float32),
             group=sim["group"].astype(np.int32), wahp=np.array(C.W_AHP, np.float32),
             t_obs=np.int32(t_obs))
    first = Nmask[:, :12]
    obs_mean = N[:, :12][first].mean() if first.any() else N[Nmask].mean() if Nmask.any() else 0.02
    d["mu0_log"] = np.float32(math.log(max(float(obs_mean), 0.02)))
    # indicators observed in information time: x at the end of month s is observed when the item has a
    # count in month s or in month s + 1 (count column j uses X[:, j]); and s <= t_obs (cold start,
    # information-time discipline)
    entered = np.zeros((I, T + 1), bool)
    entered[:, 1:] = Nmask                                  # month s has a count observation
    entered[:, :T] |= Nmask                                 # or the next month has one (its x_{t-1})
    xvalid = entered & (np.arange(T + 1)[None, :] <= t_obs)
    d["xvalid"] = xvalid
    if xmode == "latent":
        nan = np.isnan(X) & xvalid[:, :, None]
        idx = np.flatnonzero(nan.ravel()).astype(np.int32)
        cap = cap if cap is not None else idx.size
        if idx.size > cap:
            raise ValueError(f"{idx.size} missing indicators exceed the cap {cap}")
        pad = np.full(cap - idx.size, I * (T + 1) * K, np.int32)            # dummy position
        d["xmis_idx"] = np.concatenate([idx, pad])
        d["xmis_valid"] = np.arange(cap) < idx.size
        d["X"] = np.where(xvalid[:, :, None], np.nan_to_num(X, nan=0.0), 0.0).astype(np.float32)
    else:
        d["X"] = np.where(xvalid[:, :, None], locf(np.where(xvalid[:, :, None], X, np.nan)), 0.0).astype(np.float32)
    if C.INNOV_SPLIT:
        d["a_eta"], d["Q_eta"], d["a_zeta"], d["Q_zeta"] = innov_basis(T, t_obs)
    fast_stats(d, xmode, G=C.G)
    return d


# ---------------------------------------------------------------- v1.4: statistics of the fast log density
def fast_stats(d, xmode, G=C.G):
    """Adds (in place) the data summaries used by ews_model (v1.4). All shapes are fixed by (I, T, K,
    latent-slot cap), so replications and origins keep one compiled sampler.
      counts: per item, the number of observed months with count > k (k < TB), the count total, and an
              overflow list for counts >= TB (exact lgamma there);
      indicators (latent mode): per-k sums over pairs of two observed values, and the pairs that touch
              a missing value (at most two per latent slot);
      shear: window of the least-squares level / slope of the trend path = the observed months t < t_obs."""
    TB, OV = C.COUNT_TABLE, C.COUNT_OVERFLOW
    N, Nmask = d["N"], d["Nmask"]
    I, T = N.shape
    n = np.rint(N).astype(np.int64)
    n_obs = np.where(Nmask, n, 0)
    d["Nf"] = n_obs.astype(np.float32)
    d["Nmf"] = Nmask.astype(np.float32)
    d["nb_tail_i"] = ((n_obs[:, :, None] > np.arange(TB)[None, None, :]) & Nmask[:, :, None]).sum(1).astype(np.float32)
    d["nb_ntot_i"] = n_obs.sum(1).astype(np.float32)
    big_i, big_t = np.nonzero(Nmask & (n >= TB))
    if big_i.size > OV:
        raise ValueError(f"{big_i.size} counts >= {TB} exceed the overflow capacity {OV}")
    pad = OV - big_i.size
    d["ov_i"] = np.concatenate([big_i, np.zeros(pad, np.int64)]).astype(np.int32)
    d["ov_n"] = np.concatenate([n[big_i, big_t], np.full(pad, TB)]).astype(np.float32)
    d["ov_valid"] = np.arange(OV) < big_i.size
    if xmode == "latent":
        X, xvalid = d["X"], d["xvalid"]
        K = X.shape[2]
        idx, valid = d["xmis_idx"], d["xmis_valid"]
        cap = idx.shape[0]
        miss = np.zeros(X.shape, bool)
        slot = np.zeros(X.shape, np.int64)
        miss.reshape(-1)[idx[valid]] = True
        slot.reshape(-1)[idx[valid]] = np.flatnonzero(valid)
        pair = np.repeat((xvalid[:, 1:] & xvalid[:, :-1])[:, :, None], K, 2)       # (s-1, s), s = 1..T
        cm, pm = miss[:, 1:], miss[:, :-1]
        oo = pair & ~cm & ~pm
        Xc, Xp = X[:, 1:], X[:, :-1]
        d["ar_n"] = pair.sum((0, 1)).astype(np.float32)
        d["ar_scc"] = np.where(oo, Xc * Xc, 0.0).sum((0, 1)).astype(np.float32)
        d["ar_scp"] = np.where(oo, Xc * Xp, 0.0).sum((0, 1)).astype(np.float32)
        d["ar_spp"] = np.where(oo, Xp * Xp, 0.0).sum((0, 1)).astype(np.float32)
        ii, ss, kk = np.nonzero(pair & (cm | pm))
        mcap = 2 * cap
        if ii.size > mcap:
            raise ValueError(f"{ii.size} indicator pairs with a missing value exceed {mcap}")
        pad = mcap - ii.size
        z_i = lambda a, dt: np.concatenate([a, np.zeros(pad, a.dtype)]).astype(dt)
        for tag, s_ in (("c", ss + 1), ("p", ss)):
            m_ = miss[ii, s_, kk]
            d[f"mx_{tag}s"] = z_i(np.where(m_, slot[ii, s_, kk], 0), np.int32)
            d[f"mx_{tag}m"] = z_i(m_, bool)
            d[f"mx_{tag}v"] = z_i(np.where(m_, 0.0, X[ii, s_, kk]), np.float32)
        d["mx_k"] = z_i(kk, np.int32)
        d["mx_valid"] = np.arange(mcap) < ii.size
        start = np.zeros(X.shape, bool)
        start[:, 0] = xvalid[:, 0][:, None]
        start[:, 1:] = (xvalid[:, 1:] & ~xvalid[:, :-1])[:, :, None]
        xs = np.zeros(cap, bool)
        xs[slot[start & miss]] = True
        d["xm_start"] = xs
    t_obs = int(d["t_obs"])
    # data-based start of the trend coordinates (nuts_gpu._init): least-squares line through the log
    # monthly count rate of each group over the observed months (the jittered prior-median start of
    # v1.3 put the slope at +-0.2 per month, i.e. the trend path into the clipped plateau of exp)
    group = d["group"]
    tt = np.arange(T)
    im, idl = np.zeros(G, np.float32), np.zeros(G, np.float32)
    for g in range(G):
        sel = group == g
        c = n_obs[sel].sum(0)
        k = Nmask[sel].sum(0)
        ok = (k > 0) & (tt < t_obs)
        if ok.sum() >= 3:
            y = np.log((c[ok] + 0.5) / k[ok])
            x = tt[ok] - tt[ok].mean()
            b = float((x * (y - y.mean())).sum() / max((x * x).sum(), 1e-9))
            idl[g] = np.clip(b, -0.05, 0.05)
            im[g] = np.clip(y.mean() + b * (np.arange(t_obs).mean() - tt[ok].mean()), -8.0, 3.0)
        else:
            im[g] = d["mu0_log"]
    d["init_mbar"], d["init_dbar"] = im, idl
    w = np.tile((np.arange(T) < t_obs).astype(np.float64), (G, 1))
    w = w / w.sum(1, keepdims=True)
    tbar = (w * np.arange(T)).sum(1)
    d["sh_w"] = w.astype(np.float32)
    d["sh_tbar"] = tbar.astype(np.float32)
    d["sh_stt"] = (w * (np.arange(T)[None] - tbar[:, None]) ** 2).sum(1).astype(np.float32)
    return d


def _group_centre(v, group, G):
    s = jax.ops.segment_sum(v, group, G)
    n = jax.ops.segment_sum(jnp.ones_like(v), group, G)
    return v - (s / n)[group]


def ews_model_ref(d, spec):
    """Eqs. (16)-(21), written directly (v1.3 reference implementation; tests compare the fast
    model below against it). `spec` is a hashable tuple (variant, pooling, innov, xmode, G)."""
    variant, pooling, innov, xmode, G = spec
    N, Nmask, X, Z, group = d["N"], d["Nmask"], d["X"], d["Z"], d["group"]
    I, T = N.shape
    K, M = X.shape[2], Z.shape[1]
    xvalid = d["xvalid"]
    # ---- indicators: latent AR(1) for the missing entries (Table 4, "latent AR(1) in the model");
    #      only pairs of information-time-observed months enter the AR(1) factor
    if xmode == "latent":
        rho = numpyro.sample("rho_x", dist.Uniform(-1.0, 1.0).expand([K]))
        xm = numpyro.sample("x_mis", dist.Normal(0.0, 1.0).expand([d["xmis_idx"].shape[0]]))
        flat = jnp.concatenate([X.reshape(-1), jnp.zeros(1)])
        flat = flat.at[d["xmis_idx"]].set(xm)
        X = flat[:-1].reshape(X.shape)
        sd = jnp.sqrt(jnp.clip(1.0 - rho ** 2, 1e-6))
        pair = (xvalid[:, 1:] & xvalid[:, :-1])[:, :, None]
        start = (xvalid[:, 1:] & ~xvalid[:, :-1])[:, :, None]           # first observed month: N(0, 1)
        lp = jnp.where(xvalid[:, 0][:, None], dist.Normal(0.0, 1.0).log_prob(X[:, 0]), 0.0).sum()
        lp = lp + jnp.where(start, dist.Normal(0.0, 1.0).log_prob(X[:, 1:]), 0.0).sum()
        lp = lp + jnp.where(pair, dist.Normal(rho * X[:, :-1], sd).log_prob(X[:, 1:]), 0.0).sum()
        # the N(0,1) sample statement above is only a carrier: remove it for real slots
        lp = lp - jnp.where(d["xmis_valid"], dist.Normal(0.0, 1.0).log_prob(xm), 0.0).sum()
        numpyro.factor("x_ar1", lp)
    beta = numpyro.sample("beta", dist.Normal(0.0, P["beta_sd"]).expand([K]))
    # ---- exposure (18)-(19)
    kap = numpyro.sample("kappa_w", dist.HalfNormal(P["kappa_sd"]))
    e = numpyro.sample("e_w", dist.Normal(0.0, C.S_W).expand([M]))
    w = numpyro.deterministic("w", kap * (d["wahp"] + e))
    if variant in ("M0", "M1") or pooling == "complete":
        u = jnp.zeros(I)
    else:
        u_raw = numpyro.sample("u_raw", dist.Normal(0.0, 1.0).expand([I]))
        scale = C.U_NOPOOL_SD if pooling == "none" else numpyro.sample("tau_C", dist.HalfNormal(P["tauC_sd"]))
        u = scale * _group_centre(u_raw, group, G)
    logC = numpyro.deterministic("logC", Z @ w + u)
    # ---- dispersion (20)
    if variant == "M0":
        alpha_i = None
    elif variant == "M1" or pooling == "complete":
        a = numpyro.sample("alpha1", dist.LogNormal(P["m_alpha_mean"], P["m_alpha_sd"]))
        numpyro.deterministic("alpha", a[None])
        alpha_i = jnp.full(I, a)
    elif pooling == "none":
        a = numpyro.sample("alpha_i", dist.LogNormal(P["m_alpha_mean"], P["m_alpha_sd"]).expand([I]))
        numpyro.deterministic("alpha", a)
        alpha_i = a
    else:
        m_a = numpyro.sample("m_alpha", dist.Normal(P["m_alpha_mean"], P["m_alpha_sd"]))
        t_a = numpyro.sample("tau_alpha", dist.HalfNormal(P["tau_alpha_sd"]))
        a_raw = numpyro.sample("a_raw", dist.Normal(0.0, 1.0).expand([G]))
        a = numpyro.deterministic("alpha", jnp.exp(m_a + t_a * a_raw))
        alpha_i = a[group]
    # ---- trend (12)-(13), non-centred
    m1 = numpyro.sample("m1", dist.Normal(d["mu0_log"], P["m1_sd"]).expand([G]))
    d1 = numpyro.sample("d1", dist.Normal(0.0, P["d1_sd"]).expand([G]))
    if variant in ("M0", "M1"):
        tt = jnp.arange(T)
        m_t = m1[:, None] + d1[:, None] * tt[None, :]
        dlt = jnp.broadcast_to(d1[:, None], (G, T))
    else:
        sig = numpyro.sample("sigma_eta", dist.HalfNormal(P["sigma_eta_sd"]))
        tau = numpyro.sample("tau_zeta", dist.HalfNormal(P["tau_zeta_sd"]))
        if "a_eta" in d:                                     # split coordinates (same N(0, I) prior)
            s_e = numpyro.sample("s_eta", dist.Normal(0.0, 1.0).expand([G]))
            s_z = numpyro.sample("s_zeta", dist.Normal(0.0, 1.0).expand([G]))
            v_e = numpyro.sample("eta_raw", dist.Normal(0.0, 1.0).expand([G, T - 2]))
            v_z = numpyro.sample("zeta_raw", dist.Normal(0.0, 1.0).expand([G, T - 2]))
            eta = s_e[:, None] * d["a_eta"][None, :] + v_e @ d["Q_eta"].T
            zet = s_z[:, None] * d["a_zeta"][None, :] + v_z @ d["Q_zeta"].T
        else:
            eta = numpyro.sample("eta_raw", dist.Normal(0.0, 1.0).expand([G, T - 1]))
            zet = numpyro.sample("zeta_raw", dist.Normal(0.0, 1.0).expand([G, T - 1]))
        dlt = jnp.concatenate([d1[:, None], d1[:, None] + tau * jnp.cumsum(zet, 1)], 1)
        scale = sig
        if innov in ("horseshoe", "horseshoe2"):
            if innov == "horseshoe":
                lam = numpyro.sample("lam_hs", dist.HalfCauchy(1.0).expand([G, T - 1]))
            else:
                # A6 (follow-up): identical prior — |N(0,1)| / |N(0,1)| is half-Cauchy(0, 1), and 1/nu ~ chi2_1 —
                # in coordinates without the heavy-tailed funnel of log lambda (Piironen & Vehtari 2017, App. C)
                z_hs = numpyro.sample("z_hs", dist.HalfNormal(1.0).expand([G, T - 1]))
                nu_hs = numpyro.sample("nu_hs", dist.InverseGamma(0.5, 0.5).expand([G, T - 1]))
                lam = z_hs * jnp.sqrt(nu_hs)
            c2 = numpyro.sample("c2_hs", dist.InverseGamma(C.HS_SLAB_DF / 2,
                                                          C.HS_SLAB_DF * C.HS_SLAB_SCALE ** 2 / 2))
            lam_t = jnp.sqrt(c2 * lam ** 2 / (c2 + sig ** 2 * lam ** 2))
            scale = sig * lam_t
        m_t = jnp.concatenate([m1[:, None], m1[:, None] + jnp.cumsum(dlt[:, :-1] + scale * eta, 1)], 1)
    numpyro.deterministic("m_t", m_t)
    numpyro.deterministic("delta_t", dlt)
    # ---- likelihood (16)-(17): N[:, j] uses x at month j (= x_{t-1} for month t = j + 1)
    logmu = m_t[group] + jnp.einsum("itk,k->it", X[:, :T], beta) + logC[:, None]
    mu = jnp.exp(jnp.clip(logmu, *C.LOGMU_CLIP))
    if alpha_i is None:
        numpyro.sample("N", dist.Poisson(mu).mask(Nmask), obs=N)
    else:
        numpyro.sample("N", dist.GammaPoisson(alpha_i[:, None], alpha_i[:, None] / mu).mask(Nmask), obs=N)


def ews_model(d, spec):
    """Eqs. (16)-(21) (v1.4): the same model as ews_model_ref — same priors, same likelihood — with
    (a) the log density evaluated through the sufficient statistics of models.fast_stats (it differs
        from ews_model_ref by a constant: lgamma(n + 1) and the Gaussian normalising constants of
        observed indicator pairs are dropped), and
    (b) the sampler moving in the coordinates (m_bar, d_bar) = level and slope of the least-squares line
        through the trend path over the observed months instead of (m1, d1): m1 and d1 are computed from
        them and the innovations (a unit-Jacobian shear) and keep their N(mu0, 1) and N(0, 0.05^2)
        priors. The initial slope d1 and the slope innovations otherwise form a ridge that forced
        NUTS to 511-1,023 leapfrog steps per iteration."""
    variant, pooling, innov, xmode, G = spec
    X, Z, group = d["X"], d["Z"], d["group"]
    Nf, Nmf = d["Nf"], d["Nmf"]
    I, T = Nf.shape
    K, M = X.shape[2], Z.shape[1]
    if xmode == "latent":
        rho = numpyro.sample("rho_x", dist.Uniform(-1.0, 1.0).expand([K]))
        xm = numpyro.sample("x_mis", dist.Normal(0.0, 1.0).expand([d["xmis_idx"].shape[0]]))
        flat = jnp.concatenate([X.reshape(-1), jnp.zeros(1)])
        flat = flat.at[d["xmis_idx"]].set(xm)
        X = flat[:-1].reshape(X.shape)
        sd = jnp.sqrt(jnp.clip(1.0 - rho ** 2, 1e-6))
        v2 = sd ** 2
        q = d["ar_scc"] - 2.0 * rho * d["ar_scp"] + rho ** 2 * d["ar_spp"]
        lp = (-d["ar_n"] * jnp.log(sd) - q / (2.0 * v2)).sum()
        xc = jnp.where(d["mx_cm"], xm[d["mx_cs"]], d["mx_cv"])
        xp = jnp.where(d["mx_pm"], xm[d["mx_ps"]], d["mx_pv"])
        lp = lp - jnp.where(d["mx_valid"], (xc - rho[d["mx_k"]] * xp) ** 2 / (2.0 * v2[d["mx_k"]]), 0.0).sum()
        # the N(0, 1) carrier of x_mis stays for padding slots and for missing values at a series start
        lp = lp + jnp.where(d["xmis_valid"] & ~d["xm_start"], 0.5 * xm ** 2, 0.0).sum()
        numpyro.factor("x_ar1", lp)
    beta = numpyro.sample("beta", dist.Normal(0.0, P["beta_sd"]).expand([K]))
    kap = numpyro.sample("kappa_w", dist.HalfNormal(P["kappa_sd"]))
    e = numpyro.sample("e_w", dist.Normal(0.0, C.S_W).expand([M]))
    w = numpyro.deterministic("w", kap * (d["wahp"] + e))
    if variant in ("M0", "M1") or pooling == "complete":
        u = jnp.zeros(I)
    else:
        u_raw = numpyro.sample("u_raw", dist.Normal(0.0, 1.0).expand([I]))
        scale = C.U_NOPOOL_SD if pooling == "none" else numpyro.sample("tau_C", dist.HalfNormal(P["tauC_sd"]))
        u = scale * _group_centre(u_raw, group, G)
    logC = numpyro.deterministic("logC", Z @ w + u)
    # alpha' by unit: groups (hier), items (none) or one (complete, M1)
    if variant == "M0":
        a_u = unit = None
    elif variant == "M1" or pooling == "complete":
        a = numpyro.sample("alpha1", dist.LogNormal(P["m_alpha_mean"], P["m_alpha_sd"]))
        numpyro.deterministic("alpha", a[None])
        a_u, unit, n_u = a[None], jnp.zeros(I, jnp.int32), 1
    elif pooling == "none":
        a = numpyro.sample("alpha_i", dist.LogNormal(P["m_alpha_mean"], P["m_alpha_sd"]).expand([I]))
        numpyro.deterministic("alpha", a)
        a_u, unit, n_u = a, jnp.arange(I), I
    else:
        m_a = numpyro.sample("m_alpha", dist.Normal(P["m_alpha_mean"], P["m_alpha_sd"]))
        t_a = numpyro.sample("tau_alpha", dist.HalfNormal(P["tau_alpha_sd"]))
        a_raw = numpyro.sample("a_raw", dist.Normal(0.0, 1.0).expand([G]))
        a = numpyro.deterministic("alpha", jnp.exp(m_a + t_a * a_raw))
        a_u, unit, n_u = a, group, G
    # trend: level / slope of the least-squares line over the observed months as coordinates
    mbar = numpyro.sample("m_bar", dist.Normal(d["mu0_log"], 10.0).expand([G]))      # carriers: their log
    dbar = numpyro.sample("d_bar", dist.Normal(0.0, 1.0).expand([G]))                # density is removed below
    tt = jnp.arange(T, dtype=jnp.float32)
    if variant in ("M0", "M1"):
        dl_in = jnp.zeros((G, T))
        p_in = jnp.zeros((G, T))
    else:
        sig = numpyro.sample("sigma_eta", dist.HalfNormal(P["sigma_eta_sd"]))
        tau = numpyro.sample("tau_zeta", dist.HalfNormal(P["tau_zeta_sd"]))
        eta = numpyro.sample("eta_raw", dist.Normal(0.0, 1.0).expand([G, T - 1]))
        zet = numpyro.sample("zeta_raw", dist.Normal(0.0, 1.0).expand([G, T - 1]))
        scale = sig
        if innov in ("horseshoe", "horseshoe2"):
            if innov == "horseshoe":
                lam = numpyro.sample("lam_hs", dist.HalfCauchy(1.0).expand([G, T - 1]))
            else:
                # A6 (follow-up): identical prior — |N(0,1)| / |N(0,1)| is half-Cauchy(0, 1), and 1/nu ~ chi2_1 —
                # in coordinates without the heavy-tailed funnel of log lambda (Piironen & Vehtari 2017, App. C)
                z_hs = numpyro.sample("z_hs", dist.HalfNormal(1.0).expand([G, T - 1]))
                nu_hs = numpyro.sample("nu_hs", dist.InverseGamma(0.5, 0.5).expand([G, T - 1]))
                lam = z_hs * jnp.sqrt(nu_hs)
            c2 = numpyro.sample("c2_hs", dist.InverseGamma(C.HS_SLAB_DF / 2,
                                                          C.HS_SLAB_DF * C.HS_SLAB_SCALE ** 2 / 2))
            lam_t = jnp.sqrt(c2 * lam ** 2 / (c2 + sig ** 2 * lam ** 2))
            scale = sig * lam_t
        dl_in = jnp.concatenate([jnp.zeros((G, 1)), tau * jnp.cumsum(zet, 1)], 1)                  # delta_t - d1
        p_in = jnp.concatenate([jnp.zeros((G, 1)), jnp.cumsum(dl_in[:, :-1] + scale * eta, 1)], 1)  # m_t - m1 - t d1
    wv, tbar = d["sh_w"], d["sh_tbar"]
    mean_p = (p_in * wv).sum(1)
    slope_p = (p_in * wv * (tt[None, :] - tbar[:, None])).sum(1) / d["sh_stt"]
    d1 = dbar - slope_p
    m1 = mbar - tbar * d1 - mean_p
    numpyro.factor("m1_d1_prior", dist.Normal(d["mu0_log"], P["m1_sd"]).log_prob(m1).sum()
                   + dist.Normal(0.0, P["d1_sd"]).log_prob(d1).sum()
                   - dist.Normal(d["mu0_log"], 10.0).log_prob(mbar).sum() - dist.Normal(0.0, 1.0).log_prob(dbar).sum())
    numpyro.deterministic("m1", m1)
    numpyro.deterministic("d1", d1)
    m_t = m1[:, None] + d1[:, None] * tt[None, :] + p_in
    dlt = d1[:, None] + dl_in
    numpyro.deterministic("m_t", m_t)
    numpyro.deterministic("delta_t", dlt)
    # likelihood (16)-(17): N[:, j] uses x at month j (= x_{t-1} for month t = j + 1)
    lm = jnp.clip(m_t[group] + jnp.einsum("itk,k->it", X[:, :T], beta) + logC[:, None], *C.LOGMU_CLIP)
    mu = jnp.exp(lm)
    if a_u is None:
        ll = (Nmf * (Nf * lm - mu)).sum()
    else:
        ai = a_u[unit][:, None]
        # log NB(n | mu, a) = sum_{k<n} log(a + k) - lgamma(n + 1) - (a + n) log1p(mu / a) + n log mu - n log a
        ll = (Nmf * (Nf * lm - (ai + Nf) * jnp.log1p(mu / ai))).sum()
        ll = ll - (jax.ops.segment_sum(d["nb_ntot_i"], unit, n_u) * jnp.log(a_u)).sum()
        tail = jax.ops.segment_sum(d["nb_tail_i"], unit, n_u)                               # (units, TB)
        kk = jnp.arange(C.COUNT_TABLE, dtype=jnp.float32)
        ll = ll + (tail * jnp.log(a_u[:, None] + kk[None, :])).sum()
        ao = a_u[unit[d["ov_i"]]]
        ll = ll + jnp.where(d["ov_valid"], jax.scipy.special.gammaln(d["ov_n"] + ao)
                            - jax.scipy.special.gammaln(C.COUNT_TABLE + ao), 0.0).sum()
    numpyro.factor("N_ll", ll)


def severity_model(d, spec=None):
    """Severity submodel (8)-(9): S | N >= 1 ~ Beta(phi m, phi (1 - m)),
    logit m = gamma0 + gamma1 log mu + gamma2' x_{t-1}; phi ~ Gamma(2, 0.1), gamma ~ N(0, 1).
    d: S (n,), smask (n,), logmu (n,), x (n, K) — item-periods flattened."""
    K = d["x"].shape[1]
    g = numpyro.sample("gamma", dist.Normal(0.0, P["gamma_sd"]).expand([2 + K]))
    phi = numpyro.sample("phi", dist.Gamma(P["phi_shape"], P["phi_rate"]))
    eta = g[0] + g[1] * d["logmu"] + d["x"] @ g[2:]
    m = jnp.clip(jax.nn.sigmoid(eta), 1e-6, 1 - 1e-6)
    phi = jnp.clip(phi, 1e-6)
    s = jnp.clip(d["S"], 1e-6, 1 - 1e-6)
    numpyro.sample("S", dist.Beta(phi * m, phi * (1 - m)).mask(d["smask"]), obs=s)


# which sites a variant reports (for ESS) — alpha is absent from M0
def ess_sites(spec):
    return tuple(s for s in C.ESS_SITES if not (s == "alpha" and spec[0] == "M0"))
