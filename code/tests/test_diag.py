"""Compare diagnostics.py with ArviZ on autocorrelated chains."""
import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax.numpy as jnp, arviz as az
import diagnostics as D
rng = np.random.default_rng(0)
worst = 0
for trial in range(40):
    phi = rng.uniform(0, 0.97); C, S = 4, 1000
    x = np.zeros((C, S)); e = rng.standard_normal((C, S))
    for t in range(1, S): x[:, t] = phi * x[:, t-1] + e[:, t]
    x += rng.normal(0, rng.uniform(0, 0.3), (C, 1))          # between-chain offsets
    if trial % 3 == 0: x = np.exp(x / 2)                    # skewed
    r, eb, et = [float(v[0]) for v in D.batch_stats(jnp.asarray(x[..., None], jnp.float32))]
    ra = float(az.rhat(x, method="rank")); eba = float(az.ess(x, method="bulk")); eta = float(az.ess(x, method="tail"))
    d = max(abs(r - ra), abs(eb - eba) / eba, abs(et - eta) / eta)
    worst = max(worst, d)
    if d > 0.01: print(trial, round(phi, 3), r, ra, eb, eba, et, eta)
print("worst relative difference", worst)
