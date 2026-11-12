"""v1.4: the blocked bootstrap (pfilter.boot_q_keys over consecutive key blocks) equals boot_q."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax, jax.numpy as jnp
import pfilter as PF

S, G, I, K = 40, 3, 12, 3
rng = np.random.default_rng(0)
group = jnp.asarray(np.repeat(np.arange(G), I // G))
pack = dict(m=jnp.asarray(rng.normal(-1.5, 0.2, (S, G)), jnp.float32), d=jnp.asarray(rng.normal(0, 0.01, (S, G)), jnp.float32),
            beta=jnp.asarray(rng.normal(0.3, 0.05, (S, K)), jnp.float32), logC=jnp.asarray(rng.normal(0, 0.3, (S, I)), jnp.float32),
            alpha=jnp.asarray(np.exp(rng.normal(np.log(2), 0.2, (S, I))), jnp.float32),
            sig=jnp.asarray(np.abs(rng.normal(0.03, 0.01, S)), jnp.float32), tau=jnp.asarray(np.abs(rng.normal(0.003, 0.001, S)), jnp.float32),
            rho=jnp.asarray(rng.uniform(0.3, 0.7, (S, K)), jnp.float32))
x_hold = jnp.asarray(rng.normal(0, 1, (I, K)), jnp.float32)
m_base = pack["m"].mean(0)
months, paths = 12, 50
key = jax.random.PRNGKey(123)
for poisson in (False, True):
    full = np.asarray(PF.boot_q_batch(jax.tree.map(lambda a: a[None], pack), key[None], x_hold[None], group, m_base[None],
                                      months, paths, poisson))[0]
    keys = jax.random.split(key, paths)
    blocks = [np.asarray(PF.boot_q_keys(pack, keys[a:a + 25], x_hold, group, m_base, months, poisson)) for a in range(0, paths, 25)]
    blk = np.concatenate(blocks, 0)
    print("poisson", poisson, full.shape, blk.shape, "max abs diff", float(np.abs(full - blk).max()),
          "calib", PF.calibrate(full) == PF.calibrate(blk))
