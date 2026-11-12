"""alarm.py — the rolling-origin protocol of §5.3 for one model and a batch of replications
(v1.3: the batch may hold replications of several cells with the same (I, T)).

For every origin t of the model's refit schedule (M2: 36, 42, ..., 114; the model-based baselines
every config.B_BASELINE_STRIDE-th origin) the model is refitted on data up to t — all replications
of the batch in one GPU call, warm-started from the previous origin's draws — and the posterior draws
become particles that are propagated month by month until the next refit (pfilter.run_segment).
At the first origin the reference level m_base is fixed and the q threshold is calibrated by
parametric bootstrap (200 stable-regime paths, ARL0 >= 24). For M2 the severity submodel (8)-(9)
is fitted at every origin as well, and a GPD tail to the observed loss fractions (§4.4).
Progress is saved after every origin (resumable) with a .json sidecar that `run.progress` reads.
"""
import json
import math
import os
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import genpareto
import config as C
import dgp
import models as Mo
import nuts_gpu as NG
import pfilter as PF

DEADLINE = [None]          # set by run.one_day: epoch seconds after which no new origin is started

SPECS = {"M2": ("M2", "hier", "gauss", "latent", C.G), "M0": ("M0", "hier", "gauss", "latent", C.G),
         "NBnone": ("M2", "none", "gauss", "latent", C.G), "NBcomplete": ("M2", "complete", "gauss", "latent", C.G)}


def origins_of(name):
    return C.ORIGINS[::C.B_M2_STRIDE] if name == "M2" else C.ORIGINS[::C.B_BASELINE_STRIDE]


def _alpha_items(a, group, I, poisson, S):
    """(S, n_alpha) -> (S, I) for hier (G), none (I) and complete (1) shapes; inf for Poisson."""
    if poisson:
        return jnp.full((S, I), jnp.inf)
    if a.shape[-1] == I:
        return a
    if a.shape[-1] == 1:
        return jnp.broadcast_to(a, (S, I))
    return a[:, group]


def make_post_fn(poisson, I, K, G):
    """Summaries of a fit at origin t = d['t_obs'] (traced, so one compiled function serves every
    origin): the particle pack, the plug-in SRI of the 24 months before t (sigma-rule history) and the
    posterior-mean log mu path (severity submodel)."""
    win = C.SIGMA_WINDOW

    def post_fn(post, d):
        t = d["t_obs"]
        group = d["group"]
        S = post["beta"].shape[0]
        m_t, dlt = post["m_t"], post["delta_t"]                              # (S, G, T)
        alpha = _alpha_items(post.get("alpha"), group, I, poisson, S)
        idx = jnp.clip(t - 1, 0, m_t.shape[-1] - 1)
        pack = dict(m=m_t[:, :, idx], d=dlt[:, :, idx], beta=post["beta"], logC=post["logC"], alpha=alpha,
                    sig=post.get("sigma_eta", jnp.zeros(S)), tau=post.get("tau_zeta", jnp.zeros(S)),
                    rho=post.get("rho_x", jnp.zeros((S, K))))
        # plug-in SRI for the decision months t-24..t-1 (0-based), i.e. count columns t-24..t-1
        Xf = d["Xf"]                                                           # (I, T+1, K), LOCF-filled
        cols = jnp.clip(t - win + jnp.arange(win), 0, m_t.shape[-1] - 1)      # (win,)
        m_win = jnp.take(m_t, cols, axis=-1)[:, group, :]                     # (S, I, win)
        x_win = jnp.take(Xf, cols, axis=1)                                      # (I, win, K)
        logmu = m_win + jnp.einsum("iwk,sk->siw", x_win, post["beta"]) + post["logC"][:, :, None]
        mu = jnp.exp(jnp.clip(logmu, *C.LOGMU_CLIP)).mean(0)                   # (I, win)
        if poisson:
            sri = jnp.exp(-mu)
        else:
            a = alpha.mean(0)[:, None]
            sri = (1.0 + mu / a) ** (-a)
        # posterior-mean log mu path for the severity submodel (I, T)
        logmu_hat = m_t.mean(0)[group] + jnp.einsum("itk,k->it", Xf[:, :m_t.shape[-1]], post["beta"].mean(0)) \
            + post["logC"].mean(0)[:, None]
        return dict(pack=pack, sri_hist=sri.T, logmu_hat=logmu_hat)
    return post_fn


def _gpd(S_obs):
    s = S_obs[np.isfinite(S_obs)]
    if s.size < 20:
        return None
    u = np.quantile(s, C.GPD_QUANTILE)
    exc = s[s > u] - u
    if exc.size < C.GPD_MIN_EXC:
        return None
    xi, _, sc = genpareto.fit(exc, floc=0.0)
    return dict(u=float(u), xi=float(xi), sigma=float(sc), zeta=float(exc.size / s.size))


def _sev_cap(sims, t):
    """Capacity of the gathered severity data for a batch (same for all its replications)."""
    I, T = sims[0]["N"].shape
    n = max(int((np.isfinite(s["S"]) & s["Nmask"] & (np.arange(T)[None, :] < t)).sum()) for s in sims)
    cap = int(math.ceil(C.SEV_CAP_FRAC * I * T))
    return cap if n <= cap else I * T


def _sev_data(sim, Xf, logmu_hat, t, cap):
    """Severity data: the item-periods with a loss observed before t (N >= 1), gathered into a vector of
    fixed capacity (v1.4: config.SEV_CAP_FRAC of the item-periods; the Beta log density was evaluated on
    every item-period before, 3-5 times more lgamma calls)."""
    I, T = sim["N"].shape
    S = sim["S"]
    ok = (np.isfinite(S) & sim["Nmask"] & (np.arange(T)[None, :] < t)).ravel()
    idx = np.flatnonzero(ok)
    pad = cap - idx.size
    sel = np.concatenate([idx, np.zeros(pad, np.int64)])
    return dict(S=np.nan_to_num(S, nan=0.5).ravel()[sel].astype(np.float32), smask=np.arange(cap) < idx.size,
                logmu=np.asarray(logmu_hat).ravel()[sel].astype(np.float32),
                x=Xf[:, :T].reshape(-1, Xf.shape[2])[sel].astype(np.float32))


def _sev_post(post, d):
    return dict(gamma=post["gamma"], phi=post["phi"])


class Deadline(RuntimeError):
    """Raised between origins when run.DEADLINE has passed; progress is already saved."""


def _done_path(tmp_dir, sim, name, xmode):
    return os.path.join(tmp_dir, "done", f"{sim['uid'].replace(':', '_')}__{name}__{xmode}.pkl")


def run(sims, name, tag, tmp_dir, seed_base, batch, xmode="latent", log=None, cap=None):
    """Run model `name` through its origins for the replications `sims` (a batch, possibly of several
    cells). Replications whose result is already cached under <tmp_dir>/done are not refitted, so a
    change of batch composition (out-of-memory retry, resume) never repeats finished work.
    Returns (list of per-replication result dicts, progress-file path)."""
    log = log or NG.LOG[0]
    os.makedirs(os.path.join(tmp_dir, "done"), exist_ok=True)
    cached = {r: _done_path(tmp_dir, s, name, xmode) for r, s in enumerate(sims)}
    have = {r: p for r, p in cached.items() if os.path.exists(p)}
    todo = [r for r in range(len(sims)) if r not in have]
    results = {r: pickle.load(open(p, "rb")) for r, p in have.items()}
    if have:
        log(f"{tag} {name}: {len(have)}/{len(sims)} replications already done (cache), fitting {len(todo)}")
    if todo:
        sub = [sims[r] for r in todo]
        res, path = _run(sub, name, tag, tmp_dir, seed_base, batch, xmode, log, cap)
        for r, rec in zip(todo, res):
            tmp = cached[r] + ".tmp"
            pickle.dump(rec, open(tmp, "wb")); os.replace(tmp, cached[r])
            results[r] = rec
    else:
        path = None
    return [results[r] for r in range(len(sims))], path


def _run(sims, name, tag, tmp_dir, seed_base, batch, xmode, log, cap):
    spec = SPECS[name][:3] + (xmode, C.G)
    poisson = spec[0] == "M0"
    R = len(sims)
    I, T = sims[0]["N"].shape
    K = sims[0]["X"].shape[2]
    group = sims[0]["group"]
    Xf = [dgp.locf(s["X"]).astype(np.float32) for s in sims]
    origins = origins_of(name)
    path = os.path.join(tmp_dir, f"{tag}__{name}__{xmode}.pkl")
    st = pickle.load(open(path, "rb")) if os.path.exists(path) else dict(done=0, reps=[dict(fits=[], seg=[]) for _ in range(R)],
                                                                          warm=None, seconds=0.0)
    if st["done"] and len(st["reps"]) != R:                  # a stale progress file of another composition
        st = dict(done=0, reps=[dict(fits=[], seg=[]) for _ in range(R)], warm=None, seconds=0.0)
    if cap is None:
        cap = max(Mo.mcap(I, T, K, s.get("miss", C.MISS_DEFAULT), s.get("cold", 0.0)) for s in sims)
    fitter = sfit = None
    for oi, t in enumerate(origins):
        if oi < st["done"]:
            continue
        if DEADLINE[0] is not None and time.time() > DEADLINE[0]:
            raise Deadline(f"{tag} {name}: deadline reached before origin {t}; progress saved ({oi}/{len(origins)})")
        t0 = time.time()
        ds = [Mo.prep(s, t, xmode, cap) for s in sims]
        for d, x in zip(ds, Xf):
            d["Xf"] = x
        if fitter is None:
            fitter = NG.BatchNUTS(Mo.ews_model, spec, ds[0], post_fn=make_post_fn(poisson, I, K, C.G),
                                  ess_sites=Mo.ess_sites(spec), name=f"alarm-{name}-{xmode}-I{I}-T{T}")
        seeds = [dgp.seed_of(seed_base, sims[r].get("uid", r), name, t) % (2 ** 31) for r in range(R)]
        warm = st["warm"]
        out, info, state = fitter.fit(ds, seeds, batch=batch, warm=warm)
        st["warm"] = jax.tree.map(np.asarray, state)
        packs = out["pack"]
        fit_s = time.time() - t0
        if oi == 0 and C.CHUNK_DIR is not None and R == 1:
            # v1.4 segmented runs (Mac VM, short commands): the bootstrap in checkpointed blocks of paths,
            # the same per-path keys as boot_q (pfilter.boot_q_keys), so the calibration is unchanged
            st["m_base"] = np.asarray(packs["m"].mean(1))
            xhold = jnp.asarray(np.stack([x[:, t] for x in Xf]))
            pk = jax.tree.map(lambda a: a[0], packs)
            pkeys = jax.random.split(jax.random.PRNGKey(seeds[0] + 1), C.BOOT_PATHS)
            parts = st.setdefault("boot_parts", [])
            while sum(p.shape[0] for p in parts) < C.BOOT_PATHS:
                if NG.time_left() < C.BOOT_SECONDS_MAX:
                    # saved with this origin's *starting* warm state: on resumption the origin's fit is then
                    # found in the segmented-fit cache (same settings) instead of being refitted warm-started
                    pickle.dump(dict(st, warm=warm), open(path + ".tmp", "wb")); os.replace(path + ".tmp", path)
                    raise NG.TimeUp(f"{tag} {name}: deadline in the bootstrap at origin {t} "
                                    f"({sum(p.shape[0] for p in parts)}/{C.BOOT_PATHS} paths)")
                a = sum(p.shape[0] for p in parts)
                b = min(C.BOOT_PATHS, a + C.BOOT_PATH_BLOCK)
                qp = PF.boot_q_keys(pk, pkeys[a:b], xhold[0], jnp.asarray(group), jnp.asarray(st["m_base"][0]),
                                    C.BOOT_MONTHS, poisson)
                parts.append(np.asarray(qp))
                pickle.dump(dict(st, warm=warm), open(path + ".tmp", "wb")); os.replace(path + ".tmp", path)
            st["calib"] = [PF.calibrate(np.concatenate(parts, 0))]
            del st["boot_parts"]
            st["sri_hist"] = [np.asarray(v) for v in out["sri_hist"]]
        elif oi == 0:
            if NG.time_left() < C.BOOT_SECONDS_MAX:                          # segmented runs: not enough time left for the bootstrap
                raise NG.TimeUp(f"{tag} {name}: deadline before the bootstrap at origin {t}")
            st["m_base"] = np.asarray(packs["m"].mean(1))                     # (R, G)
            st["calib"] = []
            xhold = jnp.asarray(np.stack([x[:, t] for x in Xf]))
            for c0 in range(0, R, C.BOOT_CHUNK):
                c1 = min(R, c0 + C.BOOT_CHUNK)
                pk = jax.tree.map(lambda a: a[c0:c1], packs)
                keys = jnp.stack([jax.random.PRNGKey(seeds[r] + 1) for r in range(c0, c1)])
                qp = PF.boot_q_batch(pk, keys, xhold[c0:c1], jnp.asarray(group), jnp.asarray(st["m_base"][c0:c1]),
                                     C.BOOT_MONTHS, C.BOOT_PATHS, poisson)
                qp = np.asarray(qp)
                st["calib"] += [PF.calibrate(qp[j]) for j in range(c1 - c0)]
            st["sri_hist"] = [np.asarray(v) for v in out["sri_hist"]]
        boot_s = time.time() - t0 - fit_s
        t_end = (origins[oi + 1] - 1) if oi + 1 < len(origins) else C.DECISION_END
        n_seg = jnp.asarray(np.stack([s["N"][:, t:t_end].T for s in sims]).astype(np.float32))
        mk_seg = jnp.asarray(np.stack([s["Nmask"][:, t:t_end].T for s in sims]))
        x_seg = jnp.asarray(np.stack([np.moveaxis(x[:, t:t_end + 1], 1, 0) for x in Xf]).astype(np.float32))
        keys = jnp.stack([jax.random.PRNGKey(sd + 2) for sd in seeds])
        t_seg = time.time()
        seg_out = PF.segment_batch(poisson)(packs, keys, n_seg, mk_seg, x_seg, jnp.asarray(group), jnp.asarray(st["m_base"]))
        seg_out = {k: np.asarray(v) for k, v in seg_out.items()}
        seg_s = time.time() - t_seg
        t_sev = time.time()
        sev = None
        if name == "M2":
            lh = np.asarray(out["logmu_hat"])
            scap = _sev_cap(sims, t)
            sd = [_sev_data(s, Xf[r], lh[r], t, scap) for r, s in enumerate(sims)]
            if sfit is None or sfit.cap != scap:
                sfit = NG.BatchNUTS(Mo.severity_model, None, sd[0], post_fn=_sev_post, rhat_extra=(),
                                    ess_sites=("gamma", "phi"), name=f"severity-I{I}-T{T}-n{scap}")
                sfit.cap = scap
            sout, sinfo, _ = sfit.fit(sd, [x + 3 for x in seeds], batch=batch)
            sev = (np.asarray(sout["gamma"]), sinfo, np.asarray(sout["phi"]))
        sev_s = time.time() - t_sev
        for r in range(R):
            rec = dict(origin=t, **{k: v for k, v in info[r].items() if k != "attempts"}, attempts=info[r]["attempts"])
            st["reps"][r]["fits"].append(rec)
            seg = {k: v[r] for k, v in seg_out.items()}
            seg["months"] = np.arange(t, t_end + 1)
            if sev is not None:
                g = sev[0][r]                                                  # (draws, 2 + K)
                eta = g[:, None, None, 0] + g[:, None, None, 1] * seg["logmu_hat"][None] + \
                    np.einsum("mik,sk->smi", np.asarray(x_seg[r]), g[:, 2:])
                seg["E_S"] = (1 / (1 + np.exp(-eta))).mean(0)                  # (M, I) body mean
                seg["gpd"] = _gpd(sims[r]["S"][:, :t])
                # v1.24 (B5): posterior predictive 5/95% quantiles of the loss fraction, one Beta draw per
                # posterior draw of (gamma, phi) -> (M, I); and the posterior summaries of gamma, phi
                ph = np.clip(sev[2][r], 1e-6, None)                              # (draws,)
                mm = np.clip(1 / (1 + np.exp(-eta)), 1e-6, 1 - 1e-6)             # (draws, M, I)
                srng = np.random.default_rng(seeds[r] + 7 + t)
                sdraw = srng.beta(ph[:, None, None] * mm, ph[:, None, None] * (1 - mm))
                seg["S_q05"], seg["S_q95"] = np.quantile(sdraw, 0.05, axis=0), np.quantile(sdraw, 0.95, axis=0)
                gq = np.quantile(g, [0.05, 0.5, 0.95], axis=0)                  # (3, 2 + K)
                pq = np.quantile(sev[2][r], [0.05, 0.5, 0.95])
                st["reps"][r].setdefault("sev_fits", []).append(dict(origin=t, failed=sev[1][r]["failed"],
                                                                     rhat_max=sev[1][r]["rhat_max"],
                                                                     gamma_q05=gq[0].tolist(), gamma_q50=gq[1].tolist(), gamma_q95=gq[2].tolist(),
                                                                     phi_q05=float(pq[0]), phi_q50=float(pq[1]), phi_q95=float(pq[2]),
                                                                     n_loss=int(sd[r]["smask"].sum())))
            st["reps"][r]["seg"].append(seg)
        st["done"] = oi + 1
        st["seconds"] += time.time() - t0
        tmp = path + ".tmp"
        pickle.dump(st, open(tmp, "wb"))
        os.replace(tmp, path)
        nf = sum(i["failed"] for i in info)
        nr = sum(i["n_refit"] for i in info)
        msg = (f"{tag} {name} origin {t} ({oi + 1}/{len(origins)}): fit {fit_s:.0f}s"
               + (f", bootstrap {boot_s:.0f}s" if oi == 0 else "") + f", filter {seg_s:.0f}s"
               + (f", severity {sev_s:.0f}s" if sev is not None else "") + f", total {time.time() - t0:.0f}s"
               f", failed {nf}/{R}, refit {nr}, steps {np.mean([i['attempts'][0]['mean_steps'] for i in info]):.0f}"
               f", rhat_over {np.mean([i['n_rhat_over'] for i in info]):.1f}/{fitter.n_rhat}")
        log(msg)
        json.dump(dict(done=st["done"], n_origins=len(origins), R=R, seconds=st["seconds"], last=msg),
                  open(path + ".json", "w"))
        NG.clear_done()
        NG.AFTER_ORIGIN[0]()
    # assemble per replication
    res = []
    for r in range(R):
        segs = st["reps"][r]["seg"]
        cat = {k: np.concatenate([s[k] for s in segs], 0) for k in ("p", "p_ar", "q", "pmf", "mu_hat", "alpha_hat", "logmu_hat", "months")}
        if name == "M2":
            E_S = np.concatenate([s["E_S"] for s in segs], 0)
            E_adj = np.concatenate([np.full_like(s["E_S"], np.nan) if s["gpd"] is None else _tail_adjust(s["E_S"], s["gpd"])
                                    for s in segs], 0)
            cat.update(E_S=E_S, E_S_adj=np.where(np.isnan(E_adj), E_S, E_adj))
            if "S_q05" in segs[0]:                                                # v1.24 (B5)
                cat.update(S_q05=np.concatenate([s["S_q05"] for s in segs], 0), S_q95=np.concatenate([s["S_q95"] for s in segs], 0))
        cat.update(fits=st["reps"][r]["fits"], calib=st["calib"][r], m_base=st["m_base"][r],
                   sri_hist=st["sri_hist"][r], sev_fits=st["reps"][r].get("sev_fits"), origins=list(origins))
        res.append(cat)
    return res, path


def _tail_adjust(E, g):
    """Tail-adjusted E[S]: body mean below u, GPD mean above u (spliced at the exceedance rate)."""
    if g["xi"] >= 1:
        tail = 1.0
    else:
        tail = min(g["u"] + g["sigma"] / (1 - g["xi"]), 1.0)
    return (1 - g["zeta"]) * np.minimum(E, g["u"]) + g["zeta"] * tail
