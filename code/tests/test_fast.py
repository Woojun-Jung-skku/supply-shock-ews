"""v1.4: the fast, reparameterised log density (models.ews_model) equals the v1.3 reference
(models.ews_model_ref) up to a constant, at random points mapped between the two parameterisations
((m1, d1) <-> level / slope of the least-squares line of the trend path over the observed months).
usage: python tests/test_fast.py"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax, jax.numpy as jnp
from numpyro.infer.util import potential_energy
from numpyro.handlers import seed as nseed, trace as ntrace
import config as C
import models as Mo, study_a, study_b, alarm


def latent(model, spec, d):
    tr = ntrace(nseed(model, 0)).get_trace({k: jnp.asarray(v) for k, v in d.items()}, spec)
    lat = {k: v["value"].shape for k, v in tr.items() if v["type"] == "sample" and not v["is_observed"]}
    return lat


def check(spec, d, label, n=6, tol=0.05):
    dj = {k: jnp.asarray(v) for k, v in d.items()}
    lat_r = latent(Mo.ews_model_ref, spec, d)
    lat_f = latent(Mo.ews_model, spec, d)
    assert set(lat_f) == (set(lat_r) - {"m1", "d1"}) | {"m_bar", "d_bar"}, (lat_r.keys(), lat_f.keys())
    pe_r = jax.jit(lambda z: potential_energy(Mo.ews_model_ref, (dj, spec), {}, z))
    pe_f = jax.jit(lambda z: potential_energy(Mo.ews_model, (dj, spec), {}, z))
    trace_f = jax.jit(lambda z: ntrace(nseed(Mo.ews_model, 0)).get_trace(dj, spec))
    key = jax.random.PRNGKey(1)
    diffs = []
    for i in range(n):
        key, *ks = jax.random.split(key, len(lat_r) + 1)
        z = {k: 0.3 * jax.random.normal(kk, s) + (d["mu0_log"] if k == "m1" else C.INIT_UNCONSTRAINED.get(k, 0.0))
             for (k, s), kk in zip(lat_r.items(), ks)}
        z["d1"] = 0.02 * jax.random.normal(ks[0], lat_r["d1"])
        # path of the reference model at z -> least-squares level / slope over the observed months
        from numpyro.infer.util import constrain_fn
        cons = constrain_fn(Mo.ews_model_ref, (dj, spec), {}, z, return_deterministic=True)
        m = np.asarray(cons["m_t"])
        w, tb, stt = d["sh_w"], d["sh_tbar"], d["sh_stt"]
        tt = np.arange(m.shape[1])
        mbar = (m * w).sum(1)
        dbar = (m * w * (tt[None] - tb[:, None])).sum(1) / stt
        zf = {k: v for k, v in z.items() if k not in ("m1", "d1")}
        zf["m_bar"], zf["d_bar"] = jnp.asarray(mbar), jnp.asarray(dbar)
        cf = constrain_fn(Mo.ews_model, (dj, spec), {}, zf, return_deterministic=True)
        assert np.allclose(np.asarray(cf["m_t"]), m, atol=2e-3), np.abs(np.asarray(cf["m_t"]) - m).max()
        assert np.allclose(np.asarray(cf["m1"]), np.asarray(cons["m_t"])[:, 0], atol=2e-3)
        diffs.append(float(pe_r(z) - pe_f(zf)))
    spread = np.ptp(diffs)
    ok = spread < tol * max(1.0, abs(np.mean(diffs)) * 1e-4 / tol)
    print(f"{label:45s} potential(ref) - potential(fast) = {np.mean(diffs):12.3f}, spread {spread:.4f} {'OK' if ok else 'FAIL'}")
    return ok


ok = True
cap = Mo.mcap(30, 120, 3, C.MISS_DEFAULT)
cell = next(c for c in study_a.cells((30, 120)) if c["reference"])
d = Mo.prep(study_a.simulate(cell, 0), 120, "latent", cap)
for v, spec in study_a.VARIANT_SPEC.items():
    ok &= check(spec, d, f"A reference cell, {v}")
ok &= check(("M2", "hier", "horseshoe2", "latent", C.G), d, "A reference cell, horseshoe2 (follow-up A6)")
c2 = next(c for c in study_a.cells((10, 60)) if (c["alpha"], c["mu"], c["regime"]) == ("0.5", "0.5", "abrupt"))
ok &= check(study_a.VARIANT_SPEC["hier"], Mo.prep(study_a.simulate(c2, 1), 60, "latent", Mo.mcap(10, 60, 3, C.MISS_DEFAULT)),
            "A a0.5 mu0.5 abrupt (10,60), hier")
sb = study_b.simulate(study_b.cells()[5], 0)
for name, spec in alarm.SPECS.items():
    ok &= check(spec, Mo.prep(sb, 48, "latent", cap), f"B origin 48, {name}")
sr = study_b.simulate(study_b.cells()[0], 2, "R", cold=0.2, miss=0.15)
capr = Mo.mcap(30, 120, 3, 0.15)
ok &= check(alarm.SPECS["M2"], Mo.prep(sr, 90, "latent", capr), "R cold 20% + miss 15%, origin 90")
ok &= check(("M2", "hier", "gauss", "locf", 3), Mo.prep(sr, 120, "locf", capr), "R locf arm, full series")
print("ALL OK" if ok else "SOME FAILED")
