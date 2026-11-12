"""study_a.py — Study A, parameter recovery (§5.1-5.2).

Grid: alpha' {0.5, 2, 10, inf} x mu {0.05, 0.2, 0.5} x regime {stable, gradual, abrupt}
      x (I, T) {(10, 60), (30, 120), (30, 240)} = 108 cells; 100 replications (reference cell 200)
      as the final target, filled in passes (config.PASS1_REPS first).
Every replication is fitted with the hierarchical, no-pooling and complete-pooling variants on the
same simulated data; in the abrupt regime also with horseshoe innovations (hierarchical).
v1.3: the replications of all 36 cells of one (I, T) are fitted together in one GPU call per
variant; the posterior summaries (5/50/95% quantiles, changepoint statistics) are computed on the
device. Results: <out>/A/<cell>__<variant>.jsonl, one line per replication.
"""
import time
import numpy as np
import jax.numpy as jnp
import config as C
import dgp
import models as Mo
import nuts_gpu as NG
import metrics as MT
import store

VARIANT_SPEC = {"hier": ("M2", "hier", "gauss", "latent", C.G), "none": ("M2", "none", "gauss", "latent", C.G),
                "complete": ("M2", "complete", "gauss", "latent", C.G),
                "horseshoe": ("M2", "hier", "horseshoe", "latent", C.G)}
Q_SITES = ("beta", "w", "kappa_w", "alpha", "logC", "tau_C", "sigma_eta", "tau_zeta", "m_alpha", "tau_alpha",
           "m_t", "delta_t")


def _allowed(levels, flt):
    return flt is None or all(levels[k] in v for k, v in flt.items())


def cells(size=None):
    out = []
    for I, T in C.SIZES:
        if size is not None and (I, T) != tuple(size):
            continue
        for a in C.ALPHAS:
            for mu in C.MUS:
                for reg in C.REGIMES:
                    ref = (a, mu, reg, I, T) == (C.REF["alpha"], C.REF["mu"], C.REF["regime"], C.REF["I"], C.REF["T"])
                    lev = dict(alpha=a, mu=mu, regime=reg, size=(I, T))
                    pool = _allowed(lev, C.A_POOLING_CELLS)
                    hs = reg in C.A_HORSESHOE_REGIMES and _allowed(lev, C.A_HORSESHOE_CELLS)
                    var = ["hier"] + [v for v in C.A_VARIANTS if v != "hier" and pool] + (["horseshoe"] if hs else [])
                    # slice: a cell of the pooling / horseshoe comparisons of a reduced profile (config.A_SLICE_REPS)
                    sl = (C.A_POOLING_CELLS is not None and pool) or (C.A_HORSESHOE_CELLS is not None and hs)
                    out.append(dict(id=f"A-a{a}-mu{mu}-{reg}-I{I}-T{T}", alpha=a, mu=mu, regime=reg, I=I, T=T,
                                    reps=C.A_REPS_REF if ref else C.A_REPS, reference=ref, variants=var,
                                    slice=sl, long=(I, T) == tuple(C.A_LONG_SIZE)))
    return out


def pass_reps(c, reps):
    """Replications of cell `c` in a pass: `reps` (int, or a dict with keys 'A' and 'A_ref' as
    config.PASS1_REPS), raised to config.A_SLICE_REPS in the cells of the pooling / horseshoe slices
    and capped at config.A_LONG_REPS in the (30, 240) cells (reduced profiles; None = no change),
    never above the cell's manuscript target."""
    n = reps if isinstance(reps, int) else reps["A_ref" if c["reference"] else "A"]
    if c["slice"] and C.A_SLICE_REPS is not None and not c["reference"]:
        n = max(n, C.A_SLICE_REPS)
    if c["long"] and C.A_LONG_REPS is not None:
        n = min(n, C.A_LONG_REPS)
    return min(n, c["reps"])


def simulate(cell, r):
    s = dgp.simulate(cell["I"], cell["T"], float(cell["mu"]), float(cell["alpha"]), cell["regime"],
                     seed=dgp.seed_of("A", cell["id"], r))
    s["miss"] = C.MISS_DEFAULT
    s["uid"] = f"{cell['id']}:{r}"
    return s


def post_fn(post, d):
    """Quantiles of the reported sites and the changepoint statistics (metrics.py)."""
    out = {k: NG.quantiles(post[k]) for k in Q_SITES if k in post}
    m, dl = post["m_t"], post["delta_t"]
    innov = jnp.abs(jnp.diff(m, axis=-1) - dl[..., :-1])                 # (S, G, T-1)
    tau = innov.argmax(-1) + 1                                            # (S, G)
    out["cp"] = dict(NG.quantiles(tau.astype(jnp.float32)), p_t0=(tau == (m.shape[-1] // 2)).mean(0))  # P(tau == t0), t0 = T // 2
    return out


_FITTERS = {}


def fitter_for(variant, template, I, T):
    key = (variant, I, T)
    if key not in _FITTERS:
        _FITTERS[key] = NG.BatchNUTS(Mo.ews_model, VARIANT_SPEC[variant], template, post_fn=post_fn,
                                     name=f"A-{variant}-I{I}-T{T}")
    return _FITTERS[key]


def run_items(items, variant, out, batch, study="A", log=None):
    """Fit the (cell, replication) pairs `items` — same (I, T) — with `variant`, in GPU batches."""
    log = log or NG.LOG[0]
    if not items:
        return 0
    I, T = items[0][0]["I"], items[0][0]["T"]
    cap = Mo.mcap(I, T, C.K, C.MISS_DEFAULT)
    for b0 in range(0, len(items), batch):
        chunk = items[b0:b0 + batch]
        t0 = time.time()
        sims = [simulate(c, r) for c, r in chunk]
        ds = [Mo.prep(s, T, "latent", cap) for s in sims]
        fitter = fitter_for(variant, ds[0], I, T)
        seeds = [dgp.seed_of("A-fit", c["id"], variant, r) % 2 ** 31 for c, r in chunk]
        res, info, _ = fitter.fit(ds, seeds, batch=len(chunk))
        host = {k: (np.asarray(v) if not isinstance(v, dict) else {kk: np.asarray(vv) for kk, vv in v.items()})
                for k, v in res.items()}
        for j, (c, r) in enumerate(chunk):
            rec = dict(rep=r, cell=c["id"], variant=variant, design=C.DESIGN, **info[j])
            q = {k: {kk: vv[j] for kk, vv in v.items()} for k, v in host.items() if k != "cp"}
            tr = sims[j]["truth"]
            rec["truth"] = dict(alpha=tr["alpha"], kappa_w=tr["kappa_w"])
            rec["recovery"] = MT.recovery_q(q, tr, variant)
            if c["regime"] == "abrupt":
                cp = {k: v[j] for k, v in host["cp"].items()}
                rec["changepoint"] = MT.changepoint_q(cp, tr["t0"])
            store.append(out, study, c["id"], variant, rec)
        NG.clear_done()
        nf = sum(i["failed"] for i in info)
        nr = sum(i["n_refit"] for i in info)
        ncell = len({c["id"] for c, _ in chunk})
        log(f"A {variant} I{I}-T{T}: {len(chunk)} reps of {ncell} cells in {time.time() - t0:.0f}s "
            f"(fit {np.sum([i['seconds'] for i in info]):.0f}s), failed {nf}, refit {nr}, "
            f"steps {np.mean([i['attempts'][0]['mean_steps'] for i in info]):.0f}, "
            f"rhat_over {np.mean([i['n_rhat_over'] for i in info]):.1f}/{fitter.n_rhat}")
    return len(items)


def todo_items(size, variant, out, reps, study="A"):
    """(cell, r) pairs of the cells of `size` still missing for `variant`, up to pass_reps(cell, reps)
    per cell (reps: int, or dict with keys 'A' and 'A_ref' as config.PASS1_REPS)."""
    items = []
    for c in cells(size):
        if variant not in c["variants"]:
            continue
        n = pass_reps(c, reps)
        done = store.load(out, study, c["id"], variant)
        items += [(c, r) for r in range(n) if r not in done]
    return items


def run_size(size, variant, out, batch, reps, study="A", log=None):
    return run_items(todo_items(size, variant, out, reps, study), variant, out, batch, study, log)
