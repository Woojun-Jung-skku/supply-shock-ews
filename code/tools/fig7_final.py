"""Fig. 7 (Study B), manuscript style: (a) ARL0 of the change channels in the stable windows (pooled
over replications: unit-months / alarm months, 90% bootstrap interval), (b) ARL1 after the change (median
and interquartile range over replications), (c) operating points of the alarm rules at triage level 2
(detection within one month against false alarms per item-year in the stable windows; means over
replications), (d) AUC-PR of each baseline minus that of M2, cell by cell (means over replications).
usage: python tools/fig7_final.py ANALYSIS_DIR OUT_PREFIX"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperfig as pf
import arl as ARL
pf.apply()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

an, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(os.path.join(an, "B_replications.csv"))
cells = pd.read_csv(os.path.join(an, "B_cells.csv"))

fig = plt.figure(figsize=(157 * pf.MM, 118 * pf.MM))    # tick labels outside the axes bring the saved figure to 190 mm
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.35], wspace=0.08, hspace=0.42)
ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])


def rowplot(ax, values, labels):
    for i, v in enumerate(values):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([q1, q3], [i, i], color=pf.GREY, lw=1.6, solid_capstyle="butt")
        ax.plot([med], [i], "o", color=pf.INK, mfc=pf.INK, mec="white", mew=0.4, ms=4)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)


# (a), (b) change channels
chan = [("M2_q_arl", "q channel, M2"), ("NBcomplete_q_arl", "q channel, NB complete pooling"),
        ("NBnone_q_arl", "q channel, NB no pooling"), ("M0_q_arl", "q channel, Poisson M0"),
        ("sigma_arl", "$\\sigma$-rule, level 1 (plug-in SRI)"), ("ewma_arl", "EWMA"), ("cusum_arl", "CUSUM")]
for i, (c, _) in enumerate(chan):
    pr = ARL.pooled(d, c + "0", c.replace("_arl", "_stable_alarm_units") if c.endswith("q_arl") else c.split("_")[0] + "_stable_alarm_units",
                    ARL.UNITS["q" if "_q_" in c else "item"])
    ax_a.plot([pr["lo"], pr["hi"]], [i, i], color=pf.GREY, lw=1.6, solid_capstyle="butt")
    ax_a.plot([pr["est"]], [i], "o", color=pf.INK, mfc=pf.INK, mec="white", mew=0.4, ms=4)
    print(f"ARL0 {c}: {pr['est']:.1f} [{pr['lo']:.1f}, {pr['hi']:.1f}] alarms {pr['alarms']} / {pr['unit_months']} "
          f"unit-months, no-alarm replications {pr['no_alarm_share']:.2f}")
ax_a.set_yticks(range(len(chan)))
ax_a.set_yticklabels([n for _, n in chan])
ax_a.set_ylim(len(chan) - 0.5, -0.5)
ax_a.set_xscale("log")
ax_a.set_xlim(1, 1000)
ax_a.set_xticks([1, 3, 10, 30, 100, 300, 1000])
ax_a.set_xticklabels(["1", "3", "10", "30", "100", "300", "1000"])
ax_a.axvline(24, color=pf.LIGHT, lw=0.6, ls="--", zorder=0)
ax_a.set_xlabel("ARL$_0$ in stable windows (months, log scale)")
pf.panel(ax_a, "a")
rowplot(ax_b, [d[c + "1"] for c, _ in chan], [""] * len(chan))
ax_b.set_xlim(0, None)
ax_b.set_xlabel("ARL$_1$ after the change (months)")
ax_b.tick_params(axis="y", length=0)
pf.panel(ax_b, "b")
ax_b.texts[-1].set_x(-0.04)

# (c) operating points at triage level 2 (sigma-rule level 2; EWMA, CUSUM single level)
ops = [("M2_tri2", "M2 (proposed)", "o", pf.INK, pf.INK, 4.6), ("NBcomplete_tri2", "NB complete pooling", "^", "white", pf.INK, 4),
       ("NBnone_tri2", "NB no pooling", "s", "white", pf.INK, 3.6), ("M0_tri2", "Poisson M0", "o", "white", pf.INK, 4),
       ("xgb_tri2", "XGBoost", "D", pf.INK, pf.INK, 3.6), ("lstm_tri2", "LSTM", "v", pf.GREY, pf.GREY, 4),
       ("tft_tri2", "TFT", "x", pf.INK, pf.INK, 4), ("sigma_2", "$\\sigma$-rule, level 2", "P", "white", pf.INK, 4.6),
       ("ewma", "EWMA", "D", "white", pf.GREY, 3.6), ("cusum", "CUSUM", "s", pf.GREY, pf.GREY, 3.6)]
hc = [Line2D([], [], ls="none", marker=mk, mfc=fc, mec=ec, color=ec, ms=ms, mew=0.7, label=name)
      for k, name, mk, fc, ec, ms in ops]
for k, name, mk, fc, ec, ms in ops[::-1]:                     # M2 drawn last, on top of the baselines it coincides with
    x, y = d[f"{k}_fa_stable"].mean(), d[f"{k}_dr1"].mean()
    ax_c.plot([x], [y], ls="none", marker=mk, mfc=fc, mec=ec, color=ec, ms=ms, mew=0.7)
    print(f"L2 {name}: detection {y:.3f}, false alarms {x:.3f}")
ax_c.set_xlim(0, 0.55)
ax_c.set_ylim(0, 1)
ax_c.set_xlabel("False alarms per item-year, stable windows")
ax_c.set_ylabel("Detection within one month")
ax_c.legend(handles=hc, loc="center left", bbox_to_anchor=(0.0, 0.53), ncol=2, handletextpad=0.2, columnspacing=0.6,
            borderaxespad=0.2)
pf.panel(ax_c, "c")

# (d) per cell: baseline minus M2
base = [("M0", "Poisson M0", "o", "white", pf.INK), ("NBnone", "NB no pooling", "s", "white", pf.INK),
        ("NBcomplete", "NB complete pooling", "^", "white", pf.INK), ("xgb", "XGBoost", "D", pf.INK, pf.INK),
        ("lstm", "LSTM", "v", pf.GREY, pf.GREY), ("tft", "TFT", "x", pf.INK, pf.INK)]
reg_order = {"stable": 0, "gradual": 1, "abrupt": 2}
cc = cells.assign(_r=cells["regime"].map(reg_order)).sort_values(["alpha", "mu", "_r"]).reset_index(drop=True)
labels = [f"{a:g}, {m:g}, {r}" for a, m, r in zip(cc["alpha"], cc["mu"], cc["regime"])]
ax_d.axvline(0, color=pf.INK, lw=0.6, zorder=0)
for k, name, mk, fc, ec in base:
    x = cc[f"{k}_aucpr"] - cc["M2_aucpr"]
    ax_d.plot(x, np.arange(len(cc)), ls="none", marker=mk, mfc=fc, mec=ec, color=ec, ms=3.6, mew=0.7)
for y in (2.5, 5.5, 8.5):
    ax_d.axhline(y, color=pf.LIGHT, lw=0.4, zorder=0)
ax_d.set_yticks(range(len(cc)))
ax_d.set_yticklabels(labels)
ax_d.set_ylim(len(cc) - 0.5, -0.5)
ax_d.yaxis.tick_right()
ax_d.yaxis.set_label_position("right")
ax_d.spines["right"].set_visible(True)
ax_d.spines["left"].set_visible(False)
ax_d.set_ylabel("Cell ($\\alpha'$, $\\bar\\mu$, regime)")
ax_d.set_xlabel("AUC-PR of the baseline minus AUC-PR of M2")
handles = [Line2D([], [], ls="none", marker=mk, mfc=fc, mec=ec, color=ec, ms=3.6, mew=0.7, label=name)
           for _, name, mk, fc, ec in base]
ax_d.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, handletextpad=0.2, columnspacing=0.8)
pf.panel(ax_d, "d")
ax_d.texts[-1].set_x(-0.04)
pf.save(fig, out)
print("saved", os.path.basename(out), "replications", len(d))
