"""ARL0 of Poisson-calibrated and dispersion-corrected two-sigma count limits under NB counts.

Reproduces the run-length numbers of Section 4.3 of the manuscript, paragraph "Why not a sigma-rule?"
(Poisson-calibrated limit: ARL0 about 7-57 over the five base rates) and the sentence on dispersion-corrected limits
(NB variance at the true alpha': ARL0 about 10-57 over the same grid). It also prints the inflation of the
false-alarm rate (4.8 at mu=1), which figures/out_final/extra_sigma_rule shows.
"""
import numpy as np
from scipy.stats import nbinom, poisson

def nb_sf(k, mu, a):           # P(N >= k), NB with mean mu and dispersion a (Var = mu + mu^2/a)
    return nbinom.sf(k - 1, a, a / (a + mu))

def arl0(mu, a, limit_var):    # alarm when N > mu + 2*sqrt(limit_var)
    k = int(np.floor(mu + 2 * np.sqrt(limit_var))) + 1
    p = poisson.sf(k - 1, mu) if np.isinf(a) else nb_sf(k, mu, a)
    return 1 / p, k

mus = [0.05, 0.2, 0.5, 1, 2]
alphas = np.concatenate([np.logspace(-1, 2, 3001), [np.inf]])   # 0.1 ... 100 and the Poisson limit
print(f"{'mu':>5} | Poisson-calibrated limit: ARL0 min-max (max inflation) | NB-variance limit at true a': ARL0 min-max")
gmin_p = gmin_c = 1e9; gmax_p = gmax_c = 0
for mu in mus:
    k_p = int(np.floor(mu + 2 * np.sqrt(mu))) + 1
    p_pois = poisson.sf(k_p - 1, mu)
    A_p = np.array([1 / (p_pois if np.isinf(a) else nb_sf(k_p, mu, a)) for a in alphas])
    infl = (1 / A_p) / p_pois
    A_c = np.array([arl0(mu, a, mu + (0 if np.isinf(a) else mu * mu / a))[0] for a in alphas])
    print(f"{mu:>5} | N>={k_p}: {A_p.min():6.2f} - {A_p.max():6.2f}  (x{infl.max():.2f} at a'={alphas[infl.argmax()]:.2f}) | {A_c.min():6.2f} - {A_c.max():6.2f}")
    gmin_p, gmax_p = min(gmin_p, A_p.min()), max(gmax_p, A_p.max())
    gmin_c, gmax_c = min(gmin_c, A_c.min()), max(gmax_c, A_c.max())
print(f"over the five base rates: Poisson-calibrated {gmin_p:.2f}-{gmax_p:.2f}; dispersion-corrected {gmin_c:.2f}-{gmax_c:.2f}")
