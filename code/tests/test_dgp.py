"""Hawkes window counts vs eqs. (14)-(15) (Table A1) and basic DGP checks."""
import sys, os, math; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, dgp
rng = np.random.default_rng(1)
for n, kd in ((0.3, 1.0), (0.6, 1.0), (0.6, 4.0), (0.3, 0.3)):
    mu = np.full((200, 300), 0.5)
    N = dgp.hawkes_counts(mu, n, kd, rng)
    D = N.var() / N.mean(); g = 1 - (1 - math.exp(-kd)) / kd
    print(n, kd, "mean", round(N.mean(), 3), "D emp", round(D, 3), "D (14)", round(1 + n * (2 - n) / (1 - n) ** 2 * g, 3))
s = dgp.simulate(30, 120, 0.2, 2.0, "gradual", seed=3)
print("gradual m at t0-1,t0,t0+1:", s["truth"]["m"][0, 59:62], "mean count", s["N"].mean(), "missing x", np.isnan(s["X"]).mean())
s = dgp.simulate(30, 120, 0.2, 2.0, "stable", seed=3, cold=0.2, severity="pareto")
print("cold items", s["truth"]["cold_items"].sum(), "masked", (~s["Nmask"]).sum(), "S range", np.nanmin(s["S"]), np.nanmax(s["S"]))
