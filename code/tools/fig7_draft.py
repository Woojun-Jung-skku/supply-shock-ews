"""Draft of Fig. 7 (Study B): run lengths of the change channels and the probability quality of the
methods. usage: python tools/fig7_draft.py ANALYSIS_DIR OUT_PREFIX"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
pf.apply()
import matplotlib.pyplot as plt
d = pd.read_csv(os.path.join(sys.argv[1], "B_replications.csv"))
out = sys.argv[2]
fig, axes = plt.subplots(1, 3, figsize=(pf.DOUBLE, 58 * pf.MM))
# (a) ARL0 in the stable windows, (b) ARL1 after t0 — per method, pooled over cells (median and IQR over replications)
methods = [("M2_q_arl", "q channel (M2)"), ("M0_q_arl", "q channel (Poisson M0)"), ("sigma_arl", "$\\sigma$-rule (SRI)"),
           ("ewma_arl", "EWMA"), ("cusum_arl", "CUSUM")]
for ax, which, lab, letter in ((axes[0], "0", "ARL$_0$ (months, log scale)", "a"), (axes[1], "1", "ARL$_1$ (months)", "b")):
    for i, (col, name) in enumerate(methods):
        v = d[col + which].dropna().values
        if not v.size:
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([q1, q3], [i, i], color=pf.GREY, lw=1.6, solid_capstyle="butt")
        ax.plot([med], [i], "o", color=pf.INK, mfc=pf.INK, mec="white", mew=0.4, ms=4)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m[1] for m in methods] if letter == "a" else [])
    ax.invert_yaxis()
    ax.set_xlabel(lab)
    if which == "0":
        ax.set_xscale("log")
        ax.axvline(24, color=pf.LIGHT, lw=0.6, ls="--")
    pf.panel(ax, letter)
# (c) AUC-PR by method (probability methods), median and IQR over replications
pm = [("M2", "M2"), ("M0", "Poisson M0"), ("NBnone", "NB no pooling"), ("NBcomplete", "NB complete"), ("xgb", "XGBoost"),
      ("lstm", "LSTM"), ("tft", "TFT")]
ax = axes[2]
for i, (k, name) in enumerate(pm):
    v = d[f"{k}_aucpr"].dropna().values
    if not v.size:
        continue
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([q1, q3], [i, i], color=pf.GREY, lw=1.6, solid_capstyle="butt")
    ax.plot([med], [i], "o", color=pf.INK, mec="white", mew=0.4, ms=4)
ax.set_yticks(range(len(pm)))
ax.set_yticklabels([p[1] for p in pm])
ax.invert_yaxis()
ax.set_xlabel("AUC-PR")
pf.panel(ax, "c")
fig.tight_layout(w_pad=1.2)
pf.save(fig, out)
print("saved", out)
