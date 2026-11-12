"""nuts_gpu.py — NUTS for many replications at once (v1.3).

Replications with identical array shapes — the replications of one cell, or of every cell with the
same (I, T) — are stacked, and the whole sampler (initialisation, warm-up adaptation, sampling,
convergence diagnostics and the per-replication summaries) is one jit-compiled function, vmapped
over chains and over replications. It is the same NUTS as numpyro.infer.MCMC (numpyro.infer.hmc.hmc);
each chain keeps its own step size and diagonal mass matrix.

v1.3 changes
  * chains start at the prior median (config.INIT_UNCONSTRAINED, m1 at the data's log mean) plus
    N(0, 0.2^2) jitter — uniform(-2, 2) starts landed on the clipped plateau of exp(logmu) where
    the gradient vanishes, which cost most of the 1,000 warm-up iterations and produced the stuck
    chains seen on the Mac;
  * warm start: a fit can begin at another fit's final draws, step sizes and mass matrices and
    re-adapt for a short warm-up (rolling origins, config.WARM_START_WARMUP);
  * post_fn: the caller's summaries (quantiles, particle packs, posterior means) are computed on
    the device inside the compiled function, so that thousands of replications fit in memory;
  * padded batch sizes: replications are padded to a few compiled bucket sizes (one compile per
    bucket and setting; the compilation cache on Drive keeps them across sessions).
Convergence (§4.1) is checked on the device (diagnostics.py); a replication that fails is refitted
once on the same data with target acceptance 0.99 and tree depth 10 (config.REFIT) and otherwise
recorded as failed.
"""
import hashlib
import math
import os
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from numpyro.handlers import seed as nseed, trace as ntrace
from numpyro.infer.hmc import hmc
from numpyro.infer.util import potential_energy, constrain_fn
import config as C
import diagnostics as D

_CACHE = {}
COMPILE_SECONDS = [0.0]        # total compile time in this process (kept out of the per-fit seconds)
LOG = [print]                  # replaceable logger
AFTER_ORIGIN = [lambda: None]  # v1.4: called after every checkpointed origin (run_mac: memory check / fresh process)
DONE_FILES = []                # v1.4 segmented fits: finished-fit files to delete once the caller has stored the result


class TimeUp(RuntimeError):
    """v1.4 segmented fits (config.CHUNK_DIR): raised between segments when config.DEADLINE_EPOCH is near;
    the fit is checkpointed and resumes from the same point when it is called again."""


def clear_done():
    """Delete the finished-fit files of this process (call after the results are stored)."""
    while DONE_FILES:
        p = DONE_FILES.pop()
        try:
            os.remove(p)
        except OSError:
            pass


def time_left():
    return float("inf") if C.DEADLINE_EPOCH is None else C.DEADLINE_EPOCH - time.time()


def _np(t):
    return jax.tree.map(lambda a: None if a is None else np.asarray(a), t)


def _jnp(t):
    return jax.tree.map(lambda a: None if a is None else jnp.asarray(a), t)


def _latent_shapes(model, spec, d):
    tr = ntrace(nseed(model, 0)).get_trace({k: jnp.asarray(v) for k, v in d.items()}, spec)
    lat = {k: v["value"].shape for k, v in tr.items() if v["type"] == "sample" and not v["is_observed"]}
    det = {k: v["value"].shape for k, v in tr.items() if v["type"] == "deterministic"}
    return lat, det


def stack(ds):
    return {k: jnp.asarray(np.stack([d[k] for d in ds])) for k in ds[0]}


def buckets(b):
    """Compiled batch sizes: b, b/2, b/4, ... down to 1 (padding lanes are cheap, compiles are not)."""
    out = {b}
    while b > 1:
        b = math.ceil(b / 2)
        out.add(b)
    return sorted(out)


class BatchNUTS:
    """model(d, spec); post_fn(post, d) -> dict of per-replication summaries (device arrays).
    `post` holds the constrained draws of the stored sites (every latent except the excluded auxiliary
    vectors, every deterministic site, plus `store_extra`), each (chains*draws, ...)."""

    def __init__(self, model, spec, template, post_fn, rhat_exclude=C.RHAT_EXCLUDE,
                 rhat_extra=C.RHAT_EXTRA_DET, ess_sites=C.ESS_SITES, chains=C.CHAINS,
                 samples=C.SAMPLES, warmup=C.WARMUP, target_accept=C.TARGET_ACCEPT,
                 depth=C.MAX_TREE_DEPTH, refit=C.REFIT, dense_blocks=None, name=None, store_extra=None):
        self.model, self.spec, self.chains, self.samples = model, spec, chains, samples
        self.name = name or f"{model.__name__}{spec}"
        self.design = (warmup, target_accept, depth, samples)
        self.refit = tuple(refit) if refit is not None else None
        self.post_fn = post_fn
        self.lat, self.det = _latent_shapes(model, spec, template)
        self.rhat_sites = tuple([k for k in self.lat if k not in rhat_exclude] +
                                [k for k in rhat_extra if k in self.det])
        self.ess_sites = tuple(k for k in ess_sites if k in self.lat or k in self.det)
        self.shape_key = tuple(sorted((k, np.shape(v)) for k, v in template.items()))
        self.dense = list(dense_blocks) if dense_blocks else (list(C.DENSE_BLOCKS) if C.DENSE_BLOCKS else False)
        if self.dense:
            self.dense = [tuple(s for s in blk if s in self.lat) for blk in self.dense]
            self.dense = [b for b in self.dense if b] or False
        self.n_rhat = int(sum(np.prod(self.lat[k] if k in self.lat else self.det[k]) for k in self.rhat_sites))
        self.n_lat = int(sum(np.prod(s) for s in self.lat.values()))
        # sites stored per draw: every latent except the excluded auxiliary vectors, every deterministic
        # site, and whatever the diagnostics need
        self.store_sites = tuple(dict.fromkeys([k for k in self.lat if k not in rhat_exclude] + list(self.det)
                                               + list(self.rhat_sites) + list(self.ess_sites) + list(store_extra or ())))
        self.store_sites = tuple(k for k in self.store_sites if k in self.lat or k in self.det)

    # ---------------------------------------------------------------- initial point
    def _init(self, d, key):
        """Prior-median start (config.INIT_UNCONSTRAINED; m1 at the data's log mean) + jitter; v1.4: the trend
        level / slope (m_bar, d_bar) start at the data's least-squares line with small jitter."""
        ks = jax.random.split(key, len(self.lat))
        z0 = {}
        for (k, s), kk in zip(self.lat.items(), ks):
            base = C.INIT_UNCONSTRAINED.get(k, 0.0)
            if k == "m1":
                base = d["mu0_log"]
            elif k in ("m_bar", "d_bar"):
                base = d["init_" + k.replace("_", "")]                 # data-based line (models.fast_stats)
            z0[k] = base + C.INIT_JITTER_SITE.get(k, C.INIT_JITTER) * jax.random.normal(kk, s)
        return z0

    BIG_SITES = ("eta_raw", "zeta_raw", "x_mis", "u_raw", "lam_hs", "z_hs", "nu_hs")

    def _metric0(self, pe, flat0, z0):
        """v1.4: initial diagonal inverse mass matrix of the warm-up — 1 / H_jj (Hessian diagonal of the
        potential at the start point, exact by Hessian-vector products) for the coordinates of the
        low-dimensional sites, 1 for the standard-normal vectors. numpyro's default start is 1 for all,
        which made the first 100 warm-up iterations run at the maximal tree depth."""
        idx, o = [], 0
        for k in sorted(z0.keys()):
            n = int(np.prod(z0[k].shape)) if z0[k].shape else 1
            if k not in self.BIG_SITES:
                idx += list(range(o, o + n))
            o += n
        idx = jnp.asarray(idx, jnp.int32)
        g = jax.grad(pe)
        E = jnp.zeros((idx.size, flat0.size)).at[jnp.arange(idx.size), idx].set(1.0)
        hd = jax.lax.map(lambda v: jax.jvp(g, (flat0,), (v,))[1], E, batch_size=8)[jnp.arange(idx.size), idx]
        val = jnp.where(jnp.isfinite(hd) & (hd > 1.0), 1.0 / hd, 1.0)
        return jnp.ones_like(flat0).at[idx].set(val)

    # ---------------------------------------------------------------- one chain
    def _chain(self, d, key, warmup, ta, depth, samples, warm):
        """One NUTS chain on the flat vector of unconstrained latent values (v1.4: numpyro's NUTS sees one
        array, the model sees the dict)."""
        model, spec = self.model, self.spec
        k_init, k_run = jax.random.split(key)
        z0 = self._init(d, k_init)
        flat0, unravel = ravel_pytree(z0)
        pe = lambda f: potential_energy(model, (d, spec), {}, unravel(f))
        init_kernel, sample_kernel = hmc(pe, algo="NUTS")
        kw = dict(num_warmup=warmup, target_accept_prob=ta, max_tree_depth=depth, rng_key=k_run, dense_mass=False)
        if warm is None:
            if C.INIT_METRIC and warmup > 0:
                kw["inverse_mass_matrix"] = self._metric0(pe, flat0, z0)
            st = init_kernel(flat0, **kw)
        else:
            zf, eps, imm = warm
            # keep the inherited (fully adapted) mass matrix: with a short warm-up numpyro would
            # replace it by a Welford estimate from a few dozen draws; only the step size re-adapts
            kw.update(step_size=eps, inverse_mass_matrix=imm, adapt_mass_matrix=False)
            st = init_kernel(zf, **kw)

        def warm_step(st, _):
            st = sample_kernel(st)
            return st, st.num_steps

        def draw(st, _):
            st = sample_kernel(st)
            # constrained values + deterministic sites of this draw; only the stored sites are kept,
            # so the big auxiliary vectors (innovations, missing-indicator fill-ins) never pile up
            cons = constrain_fn(model, (d, spec), {}, unravel(st.z), return_deterministic=True)
            return st, ({k: cons[k] for k in self.store_sites}, st.diverging, st.num_steps)
        st, wsteps = jax.lax.scan(warm_step, st, None, length=warmup)
        st, (post, div, nsteps) = jax.lax.scan(draw, st, None, length=samples)
        state = (st.z, st.adapt_state.step_size, st.adapt_state.inverse_mass_matrix)
        return post, div, (nsteps, wsteps), state

    # ---------------------------------------------------------------- one replication, all chains
    def _sample(self, d, key, warmup, ta, depth, samples, warm):
        keys = jax.random.split(key, self.chains)
        vm = jax.vmap if C.CHAIN_MAP == "vmap" else (lambda f: (lambda *a: jax.lax.map(lambda x: f(*x), a)))
        if warm is None:
            return vm(lambda k: self._chain(d, k, warmup, ta, depth, samples, None))(keys)
        return vm(lambda k, w: self._chain(d, k, warmup, ta, depth, samples, w))(keys, warm)

    def _summarise(self, post, div, nsteps, state, d):
        """Diagnostics (§4.1) and the caller's summaries for one replication; post (chains, draws, ...)."""
        nsteps, wsteps = nsteps
        byc = {k: post[k].reshape((self.chains, post[k].shape[1], -1)) for k in set(self.rhat_sites) | set(self.ess_sites)}

        def observed(k):
            """v1.4: the trend paths of a rolling-origin fit are checked over the observed months t < t_obs
            only. The months after t_obs exist only because every origin is fitted on arrays padded to the
            full length (one compiled sampler); there they are prior extrapolations of the random walk with
            heavy tails, not estimates of the fit, and their tail ESS failed the check at early origins."""
            n = byc[k].shape[-1]
            if k in ("m_t", "delta_t") and "t_obs" in d and n % C.G == 0:
                T = n // C.G
                return jnp.tile(jnp.arange(T) < d["t_obs"], C.G)
            return jnp.ones(n, bool)
        r = jnp.concatenate([jnp.where(observed(k), D.rhat_only(byc[k]), 1.0) for k in self.rhat_sites])
        r = jnp.where(jnp.isfinite(r), r, 1e3)
        ess_x = jnp.concatenate([byc[k] for k in self.ess_sites], -1)
        obs_e = jnp.concatenate([observed(k) for k in self.ess_sites])
        eb = jnp.where(obs_e, jax.vmap(D.ess_bulk)(jnp.moveaxis(ess_x, -1, 0)), jnp.inf)
        et = jnp.where(obs_e, jax.vmap(D.ess_tail)(jnp.moveaxis(ess_x, -1, 0)), jnp.inf)
        sizes = jnp.array([byc[k].shape[-1] for k in self.rhat_sites])
        worst = jnp.searchsorted(jnp.cumsum(sizes), jnp.argmax(r), side="right")
        diag = dict(rhat_max=r.max(), rhat_site=worst, n_rhat_over=(r >= C.RHAT_MAX).sum(),
                    ess_bulk_min=jnp.nanmin(eb), ess_tail_min=jnp.nanmin(et),
                    n_div=div.sum(), mean_steps=nsteps.mean(), warm_steps=wsteps.mean(), step_size=state[1].mean())
        n = next(iter(post.values())).shape[1]
        if n > self.samples:                                       # a longer refit: thin to the design's draws
            step = n // self.samples
            post = {k: v[:, ::step][:, :self.samples] for k, v in post.items()}
        flat = {k: v.reshape((-1,) + v.shape[2:]) for k, v in post.items()}          # (chains*draws, ...)
        return self.post_fn(flat, d), diag

    def _batch(self, db, keys, warmup, ta, depth, samples, warm):
        """All replications of a batch: sampling vmapped over replications; diagnostics and
        summaries in chunks of C.DIAG_CHUNK replications (they need large temporaries)."""
        vm = jax.vmap if C.CHAIN_MAP == "vmap" else (lambda f: (lambda *a: jax.lax.map(lambda x: f(*x), a)))
        if warm is None:
            post, div, nsteps, state = vm(lambda d, k: self._sample(d, k, warmup, ta, depth, samples, None))(db, keys)
        else:
            post, div, nsteps, state = vm(lambda d, k, w: self._sample(d, k, warmup, ta, depth, samples, w))(db, keys, warm)
        R = keys.shape[0]
        out, diag = jax.lax.map(lambda x: self._summarise(*x), (post, div, nsteps, state, db),
                                batch_size=max(1, min(C.DIAG_CHUNK, R)))
        return out, diag, state

    def _compiled(self, R, warmup, ta, depth, samples, warm, args):
        key = (self.name, self.shape_key, R, warmup, ta, depth, self.chains, samples, warm is not None,
               self.rhat_sites, self.ess_sites, str(self.dense))
        if key not in _CACHE:
            if warm is None:
                f = lambda db, keys: self._batch(db, keys, warmup, ta, depth, samples, None)
            else:
                f = lambda db, keys, w: self._batch(db, keys, warmup, ta, depth, samples, w)
            t0 = time.time()
            LOG[0](f"[compile] {self.name} R={R} warmup={warmup} samples={samples} accept={ta} depth={depth} warm={warm is not None} ...")
            _CACHE[key] = jax.jit(f).lower(*args).compile()
            dt = time.time() - t0
            COMPILE_SECONDS[0] += dt
            LOG[0](f"[compile] done in {dt:.0f}s")
        return _CACHE[key]

    # ---------------------------------------------------------------- v1.4: segmented fit (Mac VM, 3-minute commands)
    def _kernel(self, d, settings, warm_c):
        """numpyro's NUTS kernel for this model and settings; init_kernel is called only to set its
        closures (the returned state is discarded by the caller, XLA drops the computation)."""
        warmup, ta, depth, samples = settings
        model, spec = self.model, self.spec
        z0 = self._init(d, jax.random.PRNGKey(0))
        flat0, unravel = ravel_pytree(z0)
        pe = lambda f: potential_energy(model, (d, spec), {}, unravel(f))
        init_kernel, sample_kernel = hmc(pe, algo="NUTS")
        kw = dict(num_warmup=warmup, target_accept_prob=ta, max_tree_depth=depth, rng_key=jax.random.PRNGKey(0), dense_mass=False)
        if warm_c is not None:
            kw.update(step_size=warm_c[1], inverse_mass_matrix=warm_c[2], adapt_mass_matrix=False)
        return init_kernel, sample_kernel, kw, flat0, unravel, pe, z0

    def _seg_fns(self, settings, warm):
        key = ("seg", self.name, self.shape_key, settings, warm, self.store_sites)
        if key in _CACHE:
            return _CACHE[key]
        model, spec = self.model, self.spec
        warmup = settings[0]

        def init_fn(d, k, warm_c):
            init_kernel, _, kw, flat0, unravel, pe, _z = self._kernel(d, settings, warm_c)
            k_init, k_run = jax.random.split(k)
            z0 = self._init(d, k_init)
            flat0, _ = ravel_pytree(z0)
            kw["rng_key"] = k_run
            if warm_c is None:
                if C.INIT_METRIC and warmup > 0:
                    kw["inverse_mass_matrix"] = self._metric0(pe, flat0, z0)
                return init_kernel(flat0, **kw)
            return init_kernel(warm_c[0], **kw)

        def warm_fn(d, st, warm_c, n):
            init_kernel, sample_kernel, kw, flat0, _, _, _ = self._kernel(d, settings, warm_c)
            init_kernel(flat0, **kw)                                   # sets the kernel's closures only

            def step(st, _):
                st = sample_kernel(st)
                return st, st.num_steps
            return jax.lax.scan(step, st, None, length=n)

        def draw_fn(d, st, warm_c, n):
            init_kernel, sample_kernel, kw, flat0, unravel, _, _ = self._kernel(d, settings, warm_c)
            init_kernel(flat0, **kw)

            def step(st, _):
                st = sample_kernel(st)
                cons = constrain_fn(model, (d, spec), {}, unravel(st.z), return_deterministic=True)
                return st, ({k: cons[k] for k in self.store_sites}, st.diverging, st.num_steps)
            return jax.lax.scan(step, st, None, length=n)

        def summ_fn(post, div, nsteps, wsteps, z, eps, imm, d):
            return self._summarise(post, div, (nsteps, wsteps), (z, eps, imm), d)
        fns = dict(init=jax.jit(init_fn), warm=jax.jit(warm_fn, static_argnums=3), draw=jax.jit(draw_fn, static_argnums=3),
                   summ=jax.jit(summ_fn))
        _CACHE[key] = fns
        return fns

    def _ckpt_path(self, d, seed, settings, warm):
        h = hashlib.sha1()
        h.update(repr((self.name, self.shape_key, tuple(settings), int(seed), warm is not None)).encode())
        for k in ("N", "Nmask", "t_obs", "X"):
            if k in d:
                h.update(np.ascontiguousarray(np.asarray(d[k])).tobytes())
        return os.path.join(C.CHUNK_DIR, h.hexdigest()[:20] + ".pkl")

    def _run_segmented(self, d, seed, settings, warm_c):
        """One replication, chains one after another, in segments of config.CHUNK_ITERS iterations with a
        checkpoint whenever the deadline is near (TimeUp). Same Markov chains as the unsegmented sampler."""
        warmup, ta, depth, samples = settings
        os.makedirs(C.CHUNK_DIR, exist_ok=True)
        path = self._ckpt_path(d, seed, settings, warm_c)
        done_path = path[:-4] + ".done.pkl"
        if os.path.exists(done_path):
            DONE_FILES.append(done_path)
            return pickle.load(open(done_path, "rb"))
        ck = pickle.load(open(path, "rb")) if os.path.exists(path) else dict(c=0, phase="init", it=0, st=None, sec=0.0, chains=[])
        dj = {k: jnp.asarray(v) for k, v in d.items()}
        wj = None if warm_c is None else _jnp(warm_c)
        keys = jax.random.split(jax.random.PRNGKey(int(seed)), self.chains)
        fns = self._seg_fns(tuple(settings), warm_c is not None)
        seg = C.CHUNK_ITERS
        est = [10.0]                                                   # running estimate of one segment's seconds
        seen = fns.setdefault("_seen", set())                          # functions already run (compiled) in this process

        def save_and_stop(msg):
            pickle.dump(ck, open(path + ".tmp", "wb")); os.replace(path + ".tmp", path)
            raise TimeUp(msg)

        def timed(f, *a, name="", n=0):
            t0 = time.time()
            out = f(*a)
            jax.block_until_ready(out)
            dt = time.time() - t0
            if (name, n) in seen:
                est[0] = max(0.8 * est[0], dt)                           # compile-free segments only
            seen.add((name, n))
            return out, dt
        for c in range(ck["c"], self.chains):
            wc = None if wj is None else jax.tree.map(lambda a: a[c], wj)
            if ck["phase"] == "init":
                if time_left() < 20:
                    save_and_stop(f"{self.name}: deadline before chain {c}")
                st, dt = timed(fns["init"], dj, keys[c], wc, name="init")
                ck.update(st=_np(st), phase="warm", it=0, wsteps=[], post=[], div=[], nsteps=[])
            while ck["phase"] == "warm":
                n = min(seg, warmup - ck["it"])
                if n <= 0:
                    ck.update(phase="draw", it=0)
                    break
                if time_left() < est[0] + 10:
                    save_and_stop(f"{self.name}: deadline in chain {c} warm-up at {ck['it']}/{warmup}")
                (st, ws), dt = timed(fns["warm"], dj, _jnp(ck["st"]), wc, n, name="warm", n=n)
                ck["sec"] += dt
                ck["st"] = _np(st); ck["wsteps"].append(np.asarray(ws)); ck["it"] += n
            while ck["phase"] == "draw":
                n = min(seg, samples - ck["it"])
                if n <= 0:
                    st = ck["st"]
                    ck["chains"].append(dict(post={k: np.concatenate([p[k] for p in ck["post"]], 0) for k in ck["post"][0]},
                                             div=np.concatenate(ck["div"]), nsteps=np.concatenate(ck["nsteps"]),
                                             wsteps=np.concatenate(ck["wsteps"]) if ck["wsteps"] else np.zeros(0, np.int32),
                                             z=st.z, eps=st.adapt_state.step_size, imm=st.adapt_state.inverse_mass_matrix))
                    ck.update(c=c + 1, phase="init", it=0, st=None, post=[], div=[], nsteps=[], wsteps=[])
                    break
                if time_left() < est[0] + 10:
                    save_and_stop(f"{self.name}: deadline in chain {c} sampling at {ck['it']}/{samples}")
                (st, (post, div, ns)), dt = timed(fns["draw"], dj, _jnp(ck["st"]), wc, n, name="draw", n=n)
                ck["sec"] += dt
                ck["st"] = _np(st); ck["post"].append(_np(post)); ck["div"].append(np.asarray(div)); ck["nsteps"].append(np.asarray(ns))
                ck["it"] += n
        chs = ck["chains"]
        post = {k: jnp.asarray(np.stack([c_["post"][k] for c_ in chs])) for k in chs[0]["post"]}
        div = jnp.asarray(np.stack([c_["div"] for c_ in chs]))
        nsteps = jnp.asarray(np.stack([c_["nsteps"] for c_ in chs]))
        wsteps = jnp.asarray(np.stack([c_["wsteps"] for c_ in chs])) if warmup > 0 else jnp.zeros((self.chains, 1), jnp.int32)
        z = jnp.asarray(np.stack([c_["z"] for c_ in chs])); eps = jnp.asarray(np.stack([c_["eps"] for c_ in chs]))
        imm = jnp.asarray(np.stack([c_["imm"] for c_ in chs]))
        out, diag = fns["summ"](post, div, nsteps, wsteps, z, eps, imm, dj)
        res = (jax.tree.map(lambda a: np.asarray(a)[None], out), {k: np.asarray(v)[None] for k, v in diag.items()},
               (np.asarray(z)[None], np.asarray(eps)[None], np.asarray(imm)[None]), ck["sec"])
        pickle.dump(res, open(done_path + ".tmp", "wb")); os.replace(done_path + ".tmp", done_path)
        DONE_FILES.append(done_path)
        try:
            os.remove(path)
        except OSError:
            pass
        return res

    def _run(self, ds, seeds, settings, warm=None):
        if C.CHUNK_DIR and len(ds) == 1:
            res = self._run_segmented(ds[0], seeds[0], tuple(settings), None if warm is None else jax.tree.map(lambda a: np.asarray(a)[0], warm))
            out, diag, state, sec = res
            return jax.tree.map(jnp.asarray, out), diag, jax.tree.map(jnp.asarray, state), sec
        R0 = len(ds)
        R = next(b for b in buckets(self.batch) if b >= R0)
        ds = ds + [ds[-1]] * (R - R0)                       # pad to a compiled size; padded lanes dropped
        seeds = list(seeds) + [seeds[-1]] * (R - R0)
        keys = jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds])
        args = [stack(ds), keys]
        if warm is not None:
            warm = jax.tree.map(lambda a: jnp.concatenate([a, jnp.repeat(a[-1:], R - R0, 0)], 0) if R > R0 else a, warm)
            args.append(warm)
        fn = self._compiled(R, *settings, warm, args)
        t0 = time.time()
        out, diag, state = fn(*args)
        jax.block_until_ready(diag["rhat_max"])
        sec = time.time() - t0
        diag = {k: np.asarray(v)[:R0] for k, v in diag.items()}
        out = jax.tree.map(lambda v: v[:R0], out)
        state = jax.tree.map(lambda v: v[:R0], state)
        return out, diag, state, sec

    def _ok(self, diag):
        return ((diag["rhat_max"] < C.RHAT_MAX) & (diag["ess_bulk_min"] > C.ESS_MIN) &
                (diag["ess_tail_min"] > C.ESS_MIN) & (diag["n_div"] <= C.DIV_MAX))

    def fit(self, ds, seeds, batch=None, warm=None, warmup=None):
        """Fit replications `ds` (list of data dicts) with seeds. warm = state returned by a previous
        fit of the same replications (rolling origins). Returns (out, info, state):
        out = post_fn summaries with a leading replication axis; info = per-replication dicts;
        state = (z, step_size, inverse_mass_matrix) per replication and chain, for the next warm start."""
        self.batch = batch or len(ds)
        design = (warmup if warmup is not None else (C.WARM_START_WARMUP if warm is not None else self.design[0]),
                  self.design[1], self.design[2], self.design[3])
        out, diag, state, sec = self._run(ds, seeds, design, warm)
        ok = self._ok(diag)
        info = [dict(attempts=[self._attempt(diag, i, design, sec / len(ds), warm is not None)]) for i in range(len(ds))]
        bad_all = np.flatnonzero(~ok)
        if bad_all.size and self.refit is not None:
            # v1.4: the refit keeps the design's target acceptance unless the first fit diverged
            div = np.asarray(diag["n_div"]) > 0
            for sel in (bad_all[~div[bad_all]], bad_all[div[bad_all]]):
                if not sel.size:
                    continue
                settings = self.refit if not div[sel[0]] else (self.refit[0], C.REFIT_ACCEPT_DIV) + tuple(self.refit[2:])
                out2, d2, st2, sec2 = self._run([ds[i] for i in sel], [seeds[i] + 7919 for i in sel], settings)
                ok2 = self._ok(d2)
                out = jax.tree.map(lambda a, b: a.at[jnp.asarray(sel)].set(b), out, out2)   # the refit replaces the design fit
                state = jax.tree.map(lambda a, b: a.at[jnp.asarray(sel)].set(b), state, st2)
                for j, i in enumerate(sel):
                    info[i]["attempts"].append(self._attempt(d2, j, settings, sec2 / sel.size, False))
                    ok[i] = ok2[j]
        for i, rec in enumerate(info):
            last = rec["attempts"][-1]
            rec.update({k: last[k] for k in ("rhat_max", "rhat_site", "n_rhat_over", "ess_bulk_min", "ess_tail_min", "n_div")},
                       failed=int(not ok[i]), n_refit=len(rec["attempts"]) - 1,
                       seconds=round(sum(a["seconds"] for a in rec["attempts"]), 2))
        return out, info, state

    def _attempt(self, diag, i, settings, sec, warm):
        return dict(warmup=settings[0], target_accept=settings[1], depth=settings[2], samples=settings[3], warm_start=int(warm),
                    rhat_max=round(float(diag["rhat_max"][i]), 4),
                    rhat_site=self.rhat_sites[min(int(diag["rhat_site"][i]), len(self.rhat_sites) - 1)],
                    n_rhat_over=int(diag["n_rhat_over"][i]), n_rhat=self.n_rhat,
                    ess_bulk_min=round(float(diag["ess_bulk_min"][i]), 1),
                    ess_tail_min=round(float(diag["ess_tail_min"][i]), 1),
                    n_div=int(diag["n_div"][i]), mean_steps=round(float(diag["mean_steps"][i]), 1),
                    warm_steps=round(float(diag["warm_steps"][i]), 1),
                    step_size=round(float(diag["step_size"][i]), 5),
                    ok=int(self._ok({k: v[i:i + 1] for k, v in diag.items()})[0]),
                    seconds=round(sec, 3))


# ---------------------------------------------------------------- common summaries
def quantiles(x, axis=0):
    q = jnp.percentile(x, jnp.array([5.0, 50.0, 95.0]), axis=axis)
    return dict(q05=q[0], q50=q[1], q95=q[2])
