"""Result figures: Figs. 2 and 3 of the manuscript (parameter recovery, alarm performance), Fig. S3 of the extended
results (hawkes_dispersion) and two figures that the manuscript does not use (sri_curves, sigma_rule).

usage: python make_figs_color.py OUT_DIR [sri_curves sigma_rule hawkes_dispersion parameter_recovery alarm_performance]
data: sri_curves, sigma_rule and hawkes_dispersion are analytic; Fig. 2 reads results/analysis/A_replications.csv;
Fig. 3 reads results/analysis/B_replications.csv and B_cells.csv, the Farrington-type baseline (results/followup/out/
A3_farrington_replications.csv) and the negative-binomial-designed CUSUM (results/followup/out/
B1_nbcharts_replications_group.csv), and code/tools/arl.py. Paths default to the repository layout and can be
overridden with the environment variables ANALYSIS, TOOLS and FOLLOWUP.
The schematic figures (Fig. 1 and Figs. S1, S2 and S4 of the extended results) are drawn by schem19.py, Fig. 4 by
backtest/fig_backtest.py and Fig. 5 by code/tools/cost_curves.py; all share the style module figstyle.py.
Sizes: 180 mm wide (double column), Fig. 2 140 mm; saved at that exact width so all figures share one type scale."""
import json
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy.stats import nbinom, poisson

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as rf
from figstyle import (BLUE, GOLD, GREEN, INK, MM, MUTED, PURPLE, RED, RULE, SUB, TEAL, W15, W2, METHOD,
                     tint, letters, series, save, box, arrow, canvas, rowmajor, constrained)

rf.apply()
import matplotlib.pyplot as plt

REPO = os.path.dirname(HERE)
ANALYSIS = os.environ.get("ANALYSIS", os.path.join(REPO, "results", "analysis"))
TOOLS = os.environ.get("TOOLS", os.path.join(REPO, "code", "tools"))
FOLLOWUP = os.environ.get("FOLLOWUP", os.path.join(REPO, "results", "followup", "out"))
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
AP = "\\alpha\\prime"                  # alpha-prime in mathtext (a plain ' is set as a raised superscript)
DOT = (0, (1, 1.6))                    # dotted reference lines
DASH = (0, (4, 2))                     # dashed target / nominal lines
LIM, LIMC = (0, (3.0, 1.6)), "#8B9199"  # Poisson limits (sri_curves): thin dashed, grey in (a), series colour in (b)
WHITE_LEG = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=1.0, borderpad=0.3)


def ticks(ax, axis, values, labels=None):
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(values))
    lab = dict(zip(values, labels)) if labels else {}
    a.set_major_formatter(FuncFormatter(lambda v, _: lab.get(v, f"{v:g}")))


def sri(mu, a):
    mu = np.asarray(mu, float)
    return np.exp(-mu) if np.isinf(a) else (1.0 + mu / a) ** (-a)


def two_panels(h_mm, w=W2):
    fig, ax = plt.subplots(1, 2, figsize=(w, h_mm * MM), layout="constrained")
    constrained(fig, wspace=0.06)
    return fig, ax


# ============================================================================ sri_curves (not in the manuscript)
def sri_curves():
    fig, ax = two_panels(64)
    mu = np.logspace(-2, 1, 400)
    for k, a in enumerate([0.2, 0.5, 1, 2, 10]):
        series(ax[0], mu, sri(mu, a), k, 5, label=f"${AP}={a:g}$", marks=5, mrange=(0.6, 8), ymin=0.045)
    ax[0].plot(mu, np.exp(-mu), color=LIMC, lw=0.7, ls=LIM, zorder=1.5, label=f"Poisson limit (${AP}\\to\\infty$)")
    ax[0].set_xscale("log")
    ticks(ax[0], "x", [0.01, 0.1, 1, 10])
    ax[0].set_xlim(0.01, 10)
    ax[0].set_xlabel("Expected shocks per period, $\\mu = R\\,C\\,\\Delta$")
    ax[0].legend(loc="lower left", handlelength=2.4)
    ag = np.logspace(np.log10(0.05), 2, 400)
    for k, m in enumerate([0.1, 0.5, 1, 2]):
        c = series(ax[1], ag, (1 + m / ag) ** (-ag), k, 4, label=f"$\\mu={m:g}$")
        ax[1].axhline(np.exp(-m), color=c, lw=0.7, ls=LIM, zorder=0)
    ax[1].set_xscale("log")
    ticks(ax[1], "x", [0.1, 1, 10, 100])
    ax[1].set_xlim(0.05, 100)
    ax[1].set_xlabel(f"Dispersion parameter ${AP}$")
    for m in [0.1, 0.5, 1, 2]:
        ax[1].text(95, np.exp(-m) + 0.03, f"$\\mu={m:g}$", ha="right", va="bottom", fontsize=7.5)
    for a in ax:
        a.set_ylim(0, 1.02)
        a.set_ylabel("SRI = $P(N=0)$")
    letters(fig, ax)
    save(fig, os.path.join(OUT, "extra_sri_curves"))


# ============================================================================ sigma_rule (not in the manuscript)
def p_alarm(mu, a, k=2.0):
    nstar = int(np.floor(mu + k * np.sqrt(mu))) + 1
    if np.isinf(a):
        return poisson.sf(nstar - 1, mu), nstar
    return nbinom.sf(nstar - 1, a, a / (a + mu)), nstar


def sigma_rule():
    fig, ax = two_panels(70)
    agrid = np.logspace(-1, 2, 300)
    XINF = 300.0
    hs = []
    for k, m in enumerate([0.05, 0.2, 0.5, 1.0, 2.0]):
        fa = np.array([p_alarm(m, a)[0] for a in agrid])
        fa_p, nstar = p_alarm(m, np.inf)
        c = series(ax[0], agrid, fa / fa_p, k, 5, label=f"$\\mu={m:g}$, alarm if $N\\geq{nstar}$")
        hs.append(ax[0].lines[-1])
        series(ax[1], agrid, 1.0 / fa, k, 5)
        ax[1].plot([XINF], [1 / fa_p], ls="none", marker="o", ms=4.4, mfc=c, mec="white", mew=0.5, clip_on=False,
                   zorder=4)
    ax[0].axhline(1, color=MUTED, lw=0.8, ls=DOT, zorder=0)
    ax[0].set_ylabel("Relative false-alarm probability")
    ax[1].set_ylabel("In-control ARL$_0$ (periods)")
    ax[1].set_yscale("log")
    ticks(ax[1], "y", [5, 10, 20, 50])
    ax[1].yaxis.set_minor_locator(NullLocator())
    ax[1].axhline(24, color=INK, lw=0.8, ls=DASH, zorder=0)
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel(f"True dispersion ${AP}$")
    ticks(ax[0], "x", [0.1, 1, 10, 100])
    ticks(ax[1], "x", [0.1, 1, 10, 100, XINF], ["0.1", "1", "10", "100", "$\\infty$"])
    ax[0].set_xlim(0.1, 100)
    ax[1].set_xlim(0.1, XINF * 1.3)
    ax[1].xaxis.set_minor_locator(FixedLocator([m * 10.0 ** e for e in (-1, 0, 1) for m in range(2, 10)]))
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax[1].transData, ax[1].transAxes)
    xa, xb = 150.0, 200.0
    ax[1].plot([xa, xb], [0, 0], color="white", lw=2.0, transform=tr, clip_on=False, zorder=5,
               solid_capstyle="butt")
    for xx in (xa, xb):
        ax[1].plot([xx / 1.06, xx * 1.06], [-0.018, 0.018], color=INK, lw=0.6, transform=tr, clip_on=False,
                   zorder=6)
    ax[1].legend(handles=[Line2D([], [], ls="none", marker="o", ms=4.4, mfc=MUTED, mec="white", mew=0.5,
                                label=f"Poisson (${AP}\\to\\infty$)"),
                          Line2D([], [], color=INK, lw=0.8, ls=DASH, label="Target ARL$_0$ = 24")],
                 loc="lower right")
    fig.legend(handles=rowmajor(hs, 3), loc="outside upper center", ncol=3, handlelength=2.4)
    letters(fig, ax)
    save(fig, os.path.join(OUT, "extra_sigma_rule"))


# ============================================================================ Fig. S3 of the extended results: Hawkes dispersion
def hawkes_dispersion():
    g = lambda y: 1.0 - (1.0 - np.exp(-y)) / y
    D = lambda n, kD: 1.0 + n * (2.0 - n) / (1.0 - n) ** 2 * g(kD)
    kD = np.logspace(np.log10(0.05), np.log10(50), 400)
    fig, ax = two_panels(66)
    hs = []
    for k, n in enumerate([0.1, 0.3, 0.6, 0.8]):
        c = series(ax[0], kD, D(n, kD), k, 4, label=f"$n={n:g}$")
        hs.append(ax[0].lines[-1])
        ax[0].axhline(1 / (1 - n) ** 2, color=c, lw=0.8, ls=DOT, zorder=0)
        series(ax[1], kD, 1.0 / (D(n, kD) - 1.0), k, 4)
    hs.append(Line2D([], [], color=MUTED, lw=0.8, ls=DOT, label="Large-window asymptote $1/(1-n)^2$"))
    for a in ax:
        a.set_xscale("log")
        a.set_yscale("log")
        ticks(a, "x", [0.1, 1, 10])
        a.set_xlim(0.05, 50)
        a.set_xlabel("Period length / relaxation time, $\\kappa\\Delta$")
    ticks(ax[0], "y", [1, 2, 5, 10, 20])
    ticks(ax[1], "y", [0.1, 1, 10, 100])
    ax[0].set_ylabel("Index of dispersion, $D(n,\\kappa\\Delta)$")
    ax[1].set_ylabel(f"${AP}_{{\\mathrm{{match}}}}\\,/\\,(\\Lambda\\Delta)$")
    fig.legend(handles=hs, loc="outside upper center", ncol=5, handlelength=2.4)
    letters(fig, ax)
    save(fig, os.path.join(OUT, "FigS3_hawkes_dispersion"))


# ============================================================================ Fig. 2: parameter recovery (Study A)
def parameter_recovery():
    d = pd.read_csv(os.path.join(ANALYSIS, "A_replications.csv"))
    d = d[np.isfinite(d["alpha_cover"])]
    d = d[(d.regime == "gradual") & (pd.to_numeric(d.alpha, errors="coerce") == 2)]
    fig, axes = two_panels(66, w=W15)
    VAR = (("hier", "M2", "Hierarchical"), ("none", "NBnone", "No pooling"), ("complete", "NBcomplete", "Complete pooling"))
    rows = []
    for v, mkey, vname in VAR:
        c, mk, _ = METHOD[mkey]
        for (I, T) in ((10, 60), (30, 120)):
            sub = d[(d.variant == v) & (d.I == I) & (d["T"] == T)]
            big = (I, T) == (30, 120)
            g = sub.groupby("mu")
            mus = np.array(sorted(g.groups))
            cov = g["alpha_cover"].mean().reindex(mus).values
            wid = g["alpha_width_log"].mean().reindex(mus).values
            sty = dict(ls="-" if big else (0, (3.5, 2)), color=c, marker=mk, mfc=c if big else "none",
                       mec="white" if big else c, mew=0.5 if big else 1.0, ms=5.4 if big else 5.0, lw=1.1,
                       zorder=3 if big else 4)
            axes[0].plot(mus, cov, **sty)
            axes[1].plot(mus, wid, **sty)
            rows.append((vname, I, T, np.round(cov, 3).tolist(), np.round(wid, 3).tolist(), g.size().tolist()))
    for ax in axes:
        ax.set_xscale("log")
        ticks(ax, "x", [0.05, 0.2, 0.5])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(0.04, 0.62)
        ax.set_xlabel("Base rate $\\bar\\mu$ per item-month")
    axes[0].axhline(0.9, color=MUTED, lw=0.8, ls=DASH, zorder=0)
    axes[0].set_ylabel(f"Coverage of 90% interval for ${AP}$")
    axes[0].set_ylim(0, 1.04)
    axes[1].set_ylabel(f"Width of 90% interval, log ${AP}$")
    axes[1].set_yscale("log")
    ticks(axes[1], "y", [0.1, 0.2, 0.5, 1, 2, 5])
    axes[1].yaxis.set_minor_locator(NullLocator())
    axes[1].set_ylim(0.1, 5)
    hs = [Line2D([], [], color=METHOD[m][0], marker=METHOD[m][1], mfc=METHOD[m][0], mec="white", mew=0.5, ms=5.4,
                 lw=1.1, label=n) for _, m, n in VAR]
    hs += [Line2D([], [], color=SUB, marker="o", mfc=SUB, mec="white", mew=0.5, ms=4.8, lw=1.1,
                  label="$I = 30$, $T = 120$"),
           Line2D([], [], color=SUB, ls=(0, (3.5, 2)), marker="o", mfc="none", mec=SUB, mew=1.0, ms=5.0, lw=1.1,
                  label="$I = 10$, $T = 60$")]
    fig.legend(handles=rowmajor(hs, 3), loc="outside upper center", ncol=3, handlelength=2.6, columnspacing=1.2)
    letters(fig, axes)
    save(fig, os.path.join(OUT, "Fig2_parameter_recovery"))
    for r in rows:
        print("parameter_recovery", r)


# ============================================================================ Fig. 3: alarm performance (Study B)
def alarm_performance():
    sys.path.insert(0, TOOLS)
    import arl as ARL
    d = pd.read_csv(os.path.join(ANALYSIS, "B_replications.csv"))
    farr = pd.read_csv(os.path.join(FOLLOWUP, "A3_farrington_replications.csv"))     # the Farrington-type row of the alarm table is in the figure too
    d = d.merge(farr[[c for c in farr.columns if c in ("cell", "rep") or c.startswith("farr_")]], on=["cell", "rep"], how="left")
    nbc = pd.read_csv(os.path.join(FOLLOWUP, "B1_nbcharts_replications_group.csv"))   # v1.24: NB-designed CUSUM (B1)
    d = d.merge(nbc[[c for c in nbc.columns if c in ("cell", "rep") or c.startswith("nbcusum_")]], on=["cell", "rep"], how="left")
    cells = pd.read_csv(os.path.join(ANALYSIS, "B_cells.csv"))
    fig = plt.figure(figsize=(W2, 138 * MM), layout="constrained")
    constrained(fig, hspace=0.0)
    top, bot = fig.subfigures(2, 1, height_ratios=[1, 1.36], hspace=0.0)
    ax_a, ax_b = top.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1]))
    ax_c, ax_d = bot.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1]))
    top.get_layout_engine() if hasattr(top, "get_layout_engine") else None

    def point(ax, x, y, key, ms=4.8):
        c, mk, filled = METHOD[key]
        ax.plot([x], [y], ls="none", marker=mk, mfc=c if filled else "white", mec="white" if filled else c,
                mew=0.5 if filled else 1.0, ms=ms, zorder=4)

    chan = [("M2_q_arl", "M2", "$q$ channel, M2"), ("NBcomplete_q_arl", "NBcomplete", "$q$ channel, NB complete pooling"),
            ("NBnone_q_arl", "NBnone", "$q$ channel, NB no pooling"), ("M0_q_arl", "M0", "$q$ channel, Poisson M0"),
            ("sigma_arl", "sigma", "sigma-rule, level 1 (plug-in SRI)"), ("ewma_arl", "ewma", "EWMA"),
            ("cusum_arl", "cusum", "CUSUM"), ("nbcusum_arl", "nbcusum", "CUSUM, negative-binomial design"),
            ("farr_arl", "farr", "Farrington-type threshold")]
    for i, (col, key, _) in enumerate(chan):
        share = col.replace("_arl", "_stable_alarm_units") if col.endswith("q_arl") else col.split("_")[0] + "_stable_alarm_units"
        pr = ARL.pooled(d, col + "0", share, ARL.UNITS["q" if "_q_" in col else "item"])
        c = METHOD[key][0]
        ax_a.plot([pr["lo"], pr["hi"]], [i, i], color=tint(c, 0.5), lw=2.6, solid_capstyle="butt", zorder=2)
        point(ax_a, pr["est"], i, key)
        print(f"alarm_performance ARL0 {col}: {pr['est']:.1f} [{pr['lo']:.1f}, {pr['hi']:.1f}]")
    ax_a.set_xscale("log")
    ax_a.set_xlim(1, 1000)
    ticks(ax_a, "x", [1, 3, 10, 30, 100, 300, 1000])
    ax_a.xaxis.set_minor_locator(NullLocator())
    ax_a.axvline(24, color=INK, lw=0.7, ls=DASH, zorder=0)
    ax_a.set_xlabel("ARL$_0$ in stable windows (months)")
    for i, (col, key, _) in enumerate(chan):
        v = d[col + "1"].values.astype(float)
        v = v[np.isfinite(v)]
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        c = METHOD[key][0]
        ax_b.plot([q1, q3], [i, i], color=tint(c, 0.5), lw=2.6, solid_capstyle="butt", zorder=2)
        point(ax_b, med, i, key)
        print(f"alarm_performance ARL1 {col}: median {med:.1f} IQR [{q1:.1f}, {q3:.1f}]")
    ax_b.set_xlim(0, 46)
    ax_b.set_xlabel("Delay from the change to the first alarm (months)")
    for ax, lab in ((ax_a, [n for *_, n in chan]), (ax_b, [""] * len(chan))):
        ax.set_yticks(range(len(chan)))
        ax.set_yticklabels(lab)
        ax.set_ylim(len(chan) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0)
    # (c) operating points at triage level 2
    ops = [("M2_tri2", "M2", "M2 (proposed)", 5.8), ("NBcomplete_tri2", "NBcomplete", "NB complete pooling", 5.2),
           ("NBnone_tri2", "NBnone", "NB no pooling", 4.8), ("M0_tri2", "M0", "Poisson M0", 5.0),
           ("xgb_tri2", "xgb", "XGBoost", 4.6), ("lstm_tri2", "lstm", "LSTM", 5.0), ("tft_tri2", "tft", "TFT", 5.2),
           ("sigma_2", "sigma", "sigma-rule, level 2", 5.6), ("ewma", "ewma", "EWMA", 4.6),
           ("cusum", "cusum", "CUSUM", 4.8), ("nbcusum", "nbcusum", "CUSUM, negative-binomial design", 4.8),
           ("farr", "farr", "Farrington-type threshold", 4.8)]
    first = ["M2_tri2"]                           # M2 underneath and larger: NB complete pooling sits on the same point
    for col, key, name, ms in sorted(ops, key=lambda o: o[0] not in first):
        x, y = d[f"{col}_fa_stable"].mean(), d[f"{col}_dr1"].mean()
        point(ax_c, x, y, key, ms=7.6 if col == "M2_tri2" else ms)
        print(f"alarm_performance L2 {name}: detection {y:.3f}, false alarms {x:.3f}")
    ax_c.set_xlim(0, 0.55)
    ax_c.set_ylim(0, 1)
    ax_c.set_xlabel("False alarms per item-year, stable windows")
    ax_c.set_ylabel("Detection within one month")
    ax_c.legend(handles=[rf.method_handle(key, name, ms=ms) for _, key, name, ms in ops], loc="upper left",
                bbox_to_anchor=(0.0, -0.20), ncol=2, handletextpad=0.25, columnspacing=0.9, labelspacing=0.5,
                borderaxespad=0.0)
    # (d) per cell: baseline minus M2
    base = [("M0", "Poisson M0"), ("NBnone", "NB no pooling"), ("NBcomplete", "NB complete pooling"),
            ("xgb", "XGBoost"), ("lstm", "LSTM"), ("tft", "TFT")]
    reg = {"stable": 0, "gradual": 1, "abrupt": 2}
    cc = cells.assign(_r=cells["regime"].map(reg)).sort_values(["alpha", "mu", "_r"]).reset_index(drop=True)
    labels = [f"{a:g}, {m:g}, {r}" for a, m, r in zip(cc["alpha"], cc["mu"], cc["regime"])]
    for y in (2.5, 5.5, 8.5):
        ax_d.axhline(y, color=RULE, lw=0.5, zorder=0)
    ax_d.axvline(0, color=INK, lw=0.7, zorder=1)
    for key, name in base:
        x = (cc[f"{key}_aucpr"] - cc["M2_aucpr"]).values
        for yy, xx in enumerate(x):
            point(ax_d, xx, yy, key, ms=4.6)
    ax_d.set_yticks(range(len(cc)))
    ax_d.set_yticklabels(labels)
    ax_d.set_ylim(len(cc) - 0.5, -0.5)
    ax_d.yaxis.tick_right()
    ax_d.yaxis.set_label_position("right")
    ax_d.spines["right"].set_visible(True)
    ax_d.spines["left"].set_visible(False)
    ax_d.tick_params(axis="y", length=0)
    ax_d.set_ylabel(f"Cell (${AP}$, $\\bar\\mu$, regime)")
    ax_d.set_xlabel("AUC-PR of the baseline minus AUC-PR of M2")
    ax_d.legend(handles=[rf.method_handle(k, n, ms=4.8) for k, n in base], loc="upper center",
                bbox_to_anchor=(0.5, -0.17), ncol=3, handletextpad=0.25, columnspacing=1.0)
    letters(fig, [ax_a, ax_b, ax_c, ax_d], align=["ylabel", 0.0, "ylabel", 0.0], groups=[(0, 2)])
    save(fig, os.path.join(OUT, "Fig3_alarm_performance"))


FIGS = dict(sri_curves=sri_curves, sigma_rule=sigma_rule, parameter_recovery=parameter_recovery,
            alarm_performance=alarm_performance, hawkes_dispersion=hawkes_dispersion)
if __name__ == "__main__":
    for name in (sys.argv[2:] or FIGS):
        FIGS[name]()
        print("done", name)
