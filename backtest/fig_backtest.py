"""fig_backtest.py - Fig. 4 of the manuscript: the backtest on the Korean import panel.
(a) three-month shock probability p_it(3) of M2 for six items with documented export controls, against the caution,
    alert and severe ranges of (16); trade events (E1) as ticks on the base line, export-restriction implementations
    (E2) as vertical lines, their announcement months dotted;
(b) the level-shift statistic q_t of each group with the bootstrap threshold and the persistent q-alarm months;
(c) operating points at triage level 2 on the two event definitions: shocks flagged a month ahead against alarm
    months per item-year, every method (Fig. 3c of the simulation study drawn for the real panel).
usage: python fig_backtest.py OUT_DIR PANEL_CSV FIG_OUT [--items 281111,811292,...]
       e.g. python backtest/fig_backtest.py results/backtest/main backtest/panel_410.csv figures/out/Fig4_backtest
"""
import argparse, csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("FIG_DIR", os.path.join(HERE, "..", "figures")))
import figstyle as rf

rf.apply()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("panel"); ap.add_argument("fig")
ap.add_argument("--items", default="281111,811292,250410,811010,810194,280530")
ap.add_argument("--names", default="HF (281111);Ga/Ge basket (811292);Natural graphite (250410);Antimony (811010);Tungsten (810194);Rare-earth metals (280530)")
a = ap.parse_args()

Z = np.load(os.path.join(a.out, "paths.npz"), allow_pickle=True)
S = json.load(open(os.path.join(a.out, "summary.json")))
items = [str(x) for x in Z["items"]]; periods = [str(x) for x in Z["periods"]]; DEC = Z["DEC"]
T = len(periods); t0 = int(DEC[0])
P = Z["M2_p"]; Q = Z["M2_q"]; QA = Z["M2_q_alarm"]
qthr = S["q_threshold"]["M2"]
PIS = [0.048, 0.167, 0.5]

# panel rows: events and group of each item
rows = {(r["cmd"], r["period"]): r for r in csv.DictReader(open(a.panel, encoding="utf-8"))}
group_of = {r["cmd"]: r["group"] for r in rows.values()}
groups = S["groups"]
# group q: the run keeps one q per month for the panel? (M2_q is (T,) in the smoke run: the max over groups) -> plot it once
# if a per-group q is stored, use it
QG = Z["M2_q_group"] if "M2_q_group" in Z.files else None

sel = a.items.split(","); names = a.names.split(";")
x = np.arange(T)
yrs = [i for i, p in enumerate(periods) if p.endswith("01")]
ylab = [periods[i][:4] for i in yrs]

fig = plt.figure(figsize=(rf.W2, 125 * rf.MM), constrained_layout=True)
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.15], width_ratios=[1, 1, 1])
axes_a = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
ax_b = fig.add_subplot(gs[2, :])
ax_c1 = fig.add_subplot(gs[3, 0]); ax_c2 = fig.add_subplot(gs[3, 1], sharey=None); ax_c3 = fig.add_subplot(gs[3, 2])

band_cols = [rf.tint(rf.GOLD, 0.85), rf.tint(rf.ORANGE, 0.78), rf.tint(rf.RED, 0.80)]
for ax, code, name in zip(axes_a, sel, names):
    if code not in items:
        ax.set_visible(False); continue
    i = items.index(code)
    p = P[:, i].copy(); p[:t0] = np.nan
    # bands
    ax.axhspan(PIS[0], PIS[1], color=band_cols[0], lw=0); ax.axhspan(PIS[1], PIS[2], color=band_cols[1], lw=0); ax.axhspan(PIS[2], 1.0, color=band_cols[2], lw=0)
    ax.axvspan(-0.5, t0 - 0.5, color="#f0f0f0", lw=0)
    ax.plot(x, p, color=rf.RED, lw=0.9)
    # events
    e1 = [t for t, per in enumerate(periods) if int(rows.get((code, per), {}).get("N1", 0) or 0) > 0]
    e2 = [t for t, per in enumerate(periods) if int(rows.get((code, per), {}).get("N2", 0) or 0) > 0]
    e2a = [t for t, per in enumerate(periods) if int(rows.get((code, per), {}).get("N2a", 0) or 0) > 0]
    ax.plot(e1, np.full(len(e1), -0.04), ls="none", marker="|", color=rf.INK, ms=4, mew=0.7, clip_on=False)
    for t in e2a:
        ax.axvline(t, color=rf.BLUE, lw=0.7, ls=(0, (1.2, 1.2)))
    for t in e2:
        ax.axvline(t, color=rf.BLUE, lw=0.9)
    ax.set_ylim(-0.06, 1.0); ax.set_xlim(-0.5, T - 0.5)
    ax.set_yticks([0, 0.167, 0.5, 1.0]); ax.set_yticklabels(["0", "0.17", "0.5", "1"])
    ax.set_xticks(yrs[::2]); ax.set_xticklabels(ylab[::2])
    ax.text(0.015, 0.97, name, transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=rf.INK, zorder=5,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.2))
    ax.tick_params(length=2)
for ax in axes_a[:3]:
    ax.set_xticklabels([])
axes_a[0].set_ylabel("$p_{it}(3)$"); axes_a[3].set_ylabel("$p_{it}(3)$")

# (b) q
q = Q.copy().astype(float); q[:t0] = np.nan
ax_b.axvspan(-0.5, t0 - 0.5, color="#f0f0f0", lw=0)
if QG is not None:
    for g, gname in enumerate(groups):
        qg = QG[:, g].astype(float); qg[:t0] = np.nan
        ax_b.plot(x, qg, lw=0.9, color=rf.RAMP[5][g], label=gname.replace("_", " "))
    ax_b.legend(ncol=5, loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=7)
else:
    ax_b.plot(x, q, lw=0.9, color=rf.RED, label="$q_t$ (maximum over groups)")
alarm_months = np.flatnonzero(QA)
for t in alarm_months:
    ax_b.axvspan(t - 0.5, t + 0.5, color=rf.tint(rf.RED, 0.75), lw=0)
ax_b.axhline(qthr, color=rf.INK, lw=0.6, ls="--")
ax_b.set_ylim(0, 1.02); ax_b.set_xlim(-0.5, T - 0.5); ax_b.set_ylabel("$q_t$")
ax_b.set_xticks(yrs); ax_b.set_xticklabels(ylab)
ax_b.text(T - 1, qthr + 0.03, f"threshold {qthr:.2f}", ha="right", va="bottom", fontsize=7, color=rf.INK)

# (c) operating points: the three triage levels of the probability-based methods and the sigma-rule joined by a thin line
# (level 3 -> 1, i.e. increasing alarm budget), one point for each chart; for E1 and E2
import pandas as pd
LAB = {"M2": "M2 (proposed)", "M0": "Poisson M0", "xgb": "XGBoost", "sigma": "Sigma-rule", "ewma": "EWMA", "cusum": "CUSUM", "farr": "Farrington-type"}
for ax, tgt, title in ((ax_c1, "N1", "trade interruptions (E1)"), (ax_c2, "N2", "export-restriction implementations (E2)")):
    Mt = pd.read_csv(os.path.join(a.out, f"metrics_{tgt}.csv"))
    for name in ["M2", "M0", "xgb", "sigma", "ewma", "cusum", "farr"]:
        sub = Mt[Mt.method == name]
        if sub.empty:
            continue
        c, m, filled = rf.METHOD[name]
        if name in ("M2", "M0", "xgb", "sigma"):
            pts = [(sub[sub.level == k].iloc[0]["alarm_months_per_itemyear"], sub[sub.level == k].iloc[0]["dr_h1"]) for k in (3, 2, 1)]
            ax.plot([x_ for x_, _ in pts], [y_ for _, y_ in pts], color=c, lw=0.6, alpha=0.7, zorder=1)
            for (x_, y_), k in zip(pts, (3, 2, 1)):
                ax.plot(x_, y_, ls="none", marker=m, color=c, mfc=c if filled else "white", mec=c if not filled else "white",
                        mew=0.5 if filled else 0.9, ms=4.8 if k == 2 else 3.0, label=LAB[name] if k == 2 else None, clip_on=False, zorder=3)
        else:
            r = sub.iloc[0]
            ax.plot(r["alarm_months_per_itemyear"], r["dr_h1"], ls="none", marker=m, color=c, mfc=c if filled else "white",
                    mec=c if not filled else "white", mew=0.5 if filled else 0.9, ms=4.8, label=LAB[name], clip_on=False, zorder=3)
    ax.plot([0, 12], [0, 1], color=rf.RULE, lw=0.6, ls=":", zorder=0)        # an alarm switched on at random
    ax.set_xlim(0, 12); ax.set_ylim(0, 1); ax.set_xlabel("alarm months per item-year")
    ax.set_title(title, fontsize=7.5, loc="right", pad=2)
ax_c1.set_ylabel("flagged a month\nahead")
ax_c2.set_yticklabels([])
ax_c3.axis("off")
h, l = ax_c1.get_legend_handles_labels()
ax_c3.legend(h, l, loc="center left", fontsize=7, handletextpad=0.4, borderpad=0.2, frameon=False, title="level 2 (large) and levels 3, 1 (small)", title_fontsize=7)

rf.letters(fig, [axes_a[0], ax_b, ax_c1], "abc")
for ext in ("pdf", "png"):
    fig.savefig(a.fig + "." + ext, dpi=600)
print("written", a.fig + ".pdf/.png")
