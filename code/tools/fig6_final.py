"""Fig. 6 (Study A): recovery of alpha' by base rate, problem size and pooling variant, on the cells of the
pooling comparison (gradual regime, alpha' = 2; (I, T) = (10, 60) and (30, 120)) so that the three
variants are compared on the same data. Means over replications (10 per cell, 20 in the reference cell).
usage: python tools/fig6_final.py ANALYSIS_DIR OUT_PREFIX"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
pf.apply()
import matplotlib.pyplot as plt
d = pd.read_csv(os.path.join(sys.argv[1], "A_replications.csv"))
out = sys.argv[2]
d = d[np.isfinite(d.get("alpha_cover", np.nan))] if "alpha_cover" in d else d
d = d[(d.regime == "gradual") & (pd.to_numeric(d.alpha, errors="coerce") == 2)]          # the pooling slice
fig, axes = plt.subplots(1, 2, figsize=(pf.DOUBLE * 0.72, 58 * pf.MM))
series = []
for v, vname in (("hier", "Hierarchical"), ("none", "No pooling"), ("complete", "Complete pooling")):
    for (I, T), sname in (((10, 60), "I = 10, T = 60"), ((30, 120), "I = 30, T = 120")):
        sub = d[(d.variant == v) & (d.I == I) & (d["T"] == T)]
        if len(sub) >= 3:
            series.append((v, vname, (I, T), sname, sub))
# model = marker (and grey for no pooling), problem size = line style and fill (I = 30: solid, filled)
MODEL = {"hier": ("o", pf.INK), "none": ("D", pf.GREY), "complete": ("^", pf.INK)}
for k, (v, vname, size, sname, sub) in enumerate(series):
    mk, col = MODEL[v]
    big = size == (30, 120)
    ls, mf = ("-", col) if big else ("--", "white")
    print(vname, sname, "n per base rate", sub.groupby("mu").size().to_dict())
    g = sub.groupby("mu")
    mus = np.array(sorted(g.groups.keys()))
    cov = g["alpha_cover"].mean().reindex(mus).values
    wid = g["alpha_width_log"].mean().reindex(mus).values
    lab = f"{vname}, {sname}"
    axes[0].plot(mus, cov, ls=ls, color=col, marker=mk, mfc=mf, mec=col, label=lab)
    axes[1].plot(mus, wid, ls=ls, color=col, marker=mk, mfc=mf, mec=col, label=lab)
for ax in axes:
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.2, 0.5]); ax.set_xticklabels(["0.05", "0.2", "0.5"])
    ax.set_xlabel("Base rate $\\bar\\mu$ per item-month")
    ax.minorticks_off()
axes[0].axhline(0.9, color=pf.LIGHT, lw=0.6, ls="--")
axes[0].set_ylabel("Coverage of 90% interval for $\\alpha'$")
axes[0].set_ylim(0, 1.02)
axes[1].set_ylabel("Width of 90% interval, log $\\alpha'$")
axes[1].set_yscale("log")
axes[1].set_yticks([0.1, 0.2, 0.5, 1, 2, 5]); axes[1].set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5"])
axes[1].minorticks_off()
pf.panel(axes[0], "a"); pf.panel(axes[1], "b")
axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout(w_pad=2.5)
pf.save(fig, out)
print("saved", os.path.basename(out), len(series), "series")
