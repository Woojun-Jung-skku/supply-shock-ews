"""validate_regen.py — check that the Study B replication data can be regenerated exactly from the
archived seeds (dgp.seed_of('B', cell, rep)) before any cost calculation uses them.

Checks, for all 65 replications of B_replications.csv:
  1. base_rate (share of valid decision item-months with >= 1 shock within 3 months) == M2_base_rate
  2. EWMA / CUSUM chart metrics recomputed with charts.chart_alarms + metrics.event_metrics ==
     ewma_dr1, ewma_fa_stable, cusum_dr1, cusum_fa_stable, ewma_arl0, cusum_arl0 of the CSV
usage: python -I validate_regen.py CODE_DIR ANALYSIS_DIR
"""
import sys, os
CODE, AN = sys.argv[1], sys.argv[2]
sys.path.insert(0, CODE)
import numpy as np
import pandas as pd
import config as C
C.apply_profile("t4_min")
import dgp, charts as CH, metrics as MT


def labels(N):
    """copy of ml.labels (ml.py imports jax): y[s, i] = 1{N_{i,s+1} + N_{i,s+2} + N_{i,s+3} >= 1}."""
    I, T = N.shape
    y = np.zeros((T + 1, I), np.float32)
    for s in range(T - 2):
        y[s] = N[:, s:s + 3].sum(1) > 0
    return y

B = pd.read_csv(os.path.join(AN, "B_replications.csv"))
months = np.arange(C.ORIGINS[0], C.DECISION_END + 1)        # 36..117
I_, T_ = C.REF["I"], C.REF["T"]

def regen(cell_id, rep):
    parts = cell_id.split("-"); a = parts[1][1:]; mu = parts[2][2:]; reg = parts[3]
    I_cell = int(parts[4][1:]) if len(parts) > 4 and parts[4].startswith("I") else I_              # cells with their own I (…-I100)
    return dgp.simulate(I_cell, T_, float(mu), float(a), reg, seed=dgp.seed_of("B", cell_id, int(rep))), reg

rows = []
for _, r in B.iterrows():
    s, reg = regen(r.cell, r.rep)
    N, mk, t0 = s["N"], s["Nmask"], s["truth"]["t0"]
    y = labels(N)[months]
    val = np.stack([mk[:, m:m + 3].all(1) for m in months])
    base = float(y[val].mean())
    al = CH.chart_alarms(N, dgp.seed_of("chart", r.cell, int(r.rep)), months[0], months[-1])
    ev_e = MT.event_metrics(al["ewma"], months, N, mk, t0, reg); rl_e = MT.run_length(al["ewma"], months, t0, reg)
    ev_c = MT.event_metrics(al["cusum"], months, N, mk, t0, reg); rl_c = MT.run_length(al["cusum"], months, t0, reg)
    rows.append(dict(cell=r.cell, rep=r.rep, base=base, base_csv=r.M2_base_rate,
                     ewma_dr1=ev_e["dr_h1"], ewma_dr1_csv=r.ewma_dr1, ewma_fa=ev_e["fa_per_itemyear_stable"], ewma_fa_csv=r.ewma_fa_stable,
                     ewma_arl0=rl_e["arl0"], ewma_arl0_csv=r.ewma_arl0,
                     cusum_dr1=ev_c["dr_h1"], cusum_dr1_csv=r.cusum_dr1, cusum_fa=ev_c["fa_per_itemyear_stable"], cusum_fa_csv=r.cusum_fa_stable,
                     cusum_arl0=rl_c["arl0"], cusum_arl0_csv=r.cusum_arl0))
d = pd.DataFrame(rows)
for k in ("base", "ewma_dr1", "ewma_fa", "ewma_arl0", "cusum_dr1", "cusum_fa", "cusum_arl0"):
    diff = (d[k] - d[k + "_csv"]).abs()
    print(f"{k:12s} max |diff| = {diff.max():.6f}   n exact(<1e-4) = {(diff < 1e-4).sum()}/{len(d)}")
print(d.head(3).to_string())
