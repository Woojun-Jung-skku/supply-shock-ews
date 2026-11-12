# -*- coding: utf-8 -*-
"""cost_tail.py — a risk-averse reading of the Model A cost curves: the 90th percentile and the conditional mean beyond it
(CVaR_0.90) of the per-replication cost EC_j(r) = AM_j + r (1 - DR_j) E_j across the replications of a subset, for every
policy at the level that minimises the chosen measure at each r (the lower envelope under that measure).

usage: python -I cost_tail.py COST_INPUTS_BY_REPLICATION_CSV OUT_DIR
writes cost_tail_A.csv (mean / q90 / CVaR90 by policy and r, with the level chosen), cost_tail_bands.csv (the band of r in
which M2 beats both trivial policies under each measure, the largest saving against the better trivial policy, and the
crossing with the always-act line) and cost_tail_digest.txt.
"""
import os, sys
import numpy as np
import pandas as pd

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
X = pd.read_csv(SRC)
R_REPORT = [2, 5, 10, 20, 50]
R_GRID = np.exp(np.linspace(np.log(1.0), np.log(100.0), 600))
Q = 0.90
LEVELS = {"M2": ["M2_tri1", "M2_tri2", "M2_tri3"], "NBcomplete": ["NBcomplete_tri1", "NBcomplete_tri2", "NBcomplete_tri3"],
          "NBnone": ["NBnone_tri1", "NBnone_tri2", "NBnone_tri3"], "M0": ["M0_tri1", "M0_tri2", "M0_tri3"],
          "xgb": ["xgb_tri1", "xgb_tri2", "xgb_tri3"], "lstm": ["lstm_tri1", "lstm_tri2", "lstm_tri3"], "tft": ["tft_tri1", "tft_tri2", "tft_tri3"],
          "sigma": ["sigma_1", "sigma_2", "sigma_3"], "ewma": ["ewma"], "cusum": ["cusum"], "nbcusum": ["nbcusum"], "farr": ["farr"]}
LABEL = {"M2": "M2 (proposed)", "NBcomplete": "NB complete pooling", "NBnone": "NB no pooling", "M0": "Poisson M0", "xgb": "XGBoost", "lstm": "LSTM",
         "tft": "TFT", "sigma": "Sigma-rule (plug-in SRI)", "ewma": "EWMA", "cusum": "CUSUM", "nbcusum": "CUSUM, negative-binomial design",
         "farr": "Farrington-type threshold", "never": "Never act", "always": "Always act"}
SUBSETS = {"All cells": X, "mu 0.05": X[X.mu == 0.05], "mu 0.2": X[X.mu == 0.2]}


def ec(d, key, r):
    return d[f"{key}_am"].values + r * (1.0 - d[f"{key}_dr1"].values) * d["E_h1"].values


def q90(c):
    return float(np.quantile(c, Q))


def cvar(c):
    t = np.quantile(c, Q)
    return float(c[c >= t].mean())


MEASURES = {"mean": lambda c: float(c.mean()), "q90": q90, "cvar90": cvar}
rows, bands, dig = [], [], []
P = dig.append
for sub, d in SUBSETS.items():
    E = d["E_h1"].values
    P(f"\n######## {sub} (n = {len(d)}) ########")
    # table at the reported r
    for m, keys in LEVELS.items():
        rec = dict(subset=sub, n=len(d), method=m, label=LABEL[m])
        for r in R_REPORT:
            for name, f in MEASURES.items():
                vals = [(f(ec(d, k, r)), i + 1) for i, k in enumerate(keys)]
                v, lv = min(vals)
                rec[f"{name}_r{r}"] = v
                rec[f"{name}_level_r{r}"] = lv if len(keys) > 1 else np.nan
        rows.append(rec)
    for name, f in MEASURES.items():
        rec = dict(subset=sub, n=len(d), method="never", label="Never act")
        for r in R_REPORT:
            rec[f"{name}_r{r}"] = f(r * E)
        rows.append(rec) if name == "mean" else rows[-1].update({k: v for k, v in rec.items() if k.startswith(name)})
    rec = dict(subset=sub, n=len(d), method="always", label="Always act")
    for r in R_REPORT:
        for name in MEASURES:
            rec[f"{name}_r{r}"] = 12.0
    rows.append(rec)
    # bands on the grid
    for name, f in MEASURES.items():
        m2 = np.array([min(f(ec(d, k, r)) for k in LEVELS["M2"]) for r in R_GRID])
        nev = np.array([f(r * E) for r in R_GRID])
        triv = np.minimum(nev, 12.0)
        beats = (m2 < nev) & (m2 < 12.0)
        rr = R_GRID[beats]
        sav = 1.0 - m2 / triv
        i = int(np.argmax(sav))
        cross_always = R_GRID[np.argmax(m2 > 12.0)] if (m2 > 12.0).any() else np.nan
        above10 = R_GRID[sav > 0.10]
        bands.append(dict(subset=sub, measure=name, band_lo=rr.min() if beats.any() else np.nan, band_hi=rr.max() if beats.any() else np.nan,
                          max_saving=sav[i], r_at_max=R_GRID[i], cross_always=cross_always,
                          saving_above_10pct_lo=above10.min() if above10.size else np.nan, saving_above_10pct_hi=above10.max() if above10.size else np.nan))
        P(f"{name:7s}: M2 beats both trivial policies for r in [{bands[-1]['band_lo']:.1f}, {bands[-1]['band_hi']:.1f}]; max saving {100 * sav[i]:.1f}% at r = {R_GRID[i]:.1f}; "
          f"M2 above the always-act line from r = {cross_always:.1f}; saving > 10% for r in [{bands[-1]['saving_above_10pct_lo']:.1f}, {bands[-1]['saving_above_10pct_hi']:.1f}]")
    # ordering of the alarm policies under the three measures at the reported r
    for r in R_REPORT:
        orders = {}
        for name, f in MEASURES.items():
            vals = {m: min(f(ec(d, k, r)) for k in keys) for m, keys in LEVELS.items()}
            orders[name] = sorted(vals, key=vals.get)
        P(f"r = {r}: mean {orders['mean']}\n        q90  {orders['q90']}\n        cvar {orders['cvar90']}")

T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "cost_tail_A.csv"), index=False)
pd.DataFrame(bands).to_csv(os.path.join(OUT, "cost_tail_bands.csv"), index=False)
open(os.path.join(OUT, "cost_tail_digest.txt"), "w").write("\n".join(dig))
print("\n".join(dig))
print("written:", os.path.join(OUT, "cost_tail_A.csv"), os.path.join(OUT, "cost_tail_bands.csv"))
