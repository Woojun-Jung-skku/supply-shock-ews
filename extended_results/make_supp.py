"""make_supp.py — Supplementary material (Tables S1-S12) built from the archived result files of this repository:
the analysis CSVs (results/analysis), the replication-level JSONL results (results/A, results/B, results/R) and the
follow-up arms (results/followup/out: A1, A3, A7; results/followup: A2, A5, A6). Writes supplementary.tex
(XeLaTeX) and the tables as CSV under OUT_DIR/tables; Tables S13-S16 come from make_supp_extra.py.
Paths default to the repository layout and can be overridden with the environment variables REPO, ANALYSIS,
RESULTS, FOLLOWUP, FOLLOWUP_MAC and CODE; SUPP_TITLE and SUPP_SUBTITLE set the two header lines of page 1.
usage: python make_supp.py OUT_DIR"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))                 # <repository>/extended_results
REPO = os.environ.get("REPO", os.path.dirname(HERE))
AN = os.environ.get("ANALYSIS", os.path.join(REPO, "results", "analysis"))
RES = os.environ.get("RESULTS", os.path.join(REPO, "results"))
FU = os.environ.get("FOLLOWUP", os.path.join(REPO, "results", "followup", "out"))
FUM = os.environ.get("FOLLOWUP_MAC", os.path.join(REPO, "results", "followup"))
CODE = os.environ.get("CODE", os.path.join(REPO, "code"))
sys.path.insert(0, CODE); sys.path.insert(0, os.path.join(CODE, "tools")); sys.path.insert(0, os.path.join(CODE, "followup"))
import config as C
C.apply_profile("t4_min")
import arl as ARL
import table5b as T5
import analyze as AN_
import study_b

OUT = sys.argv[1]
TAB = os.path.join(OUT, "tables")
os.makedirs(TAB, exist_ok=True)

# The two lines under "Supplementary material" on page 1. The defaults are neutral; the PDF that accompanies a submission is
# built with the manuscript title and the journal line in the environment variables SUPP_TITLE and SUPP_SUBTITLE.
TITLE = os.environ.get("SUPP_TITLE", "A hierarchical gamma–Poisson early-warning model of supply shocks with cost-based alarms")
SUBTITLE = os.environ.get("SUPP_SUBTITLE", "Supplementary material to the manuscript")
B = pd.read_csv(os.path.join(AN, "B_replications.csv"))
A = pd.read_csv(os.path.join(AN, "A_replications.csv"))
A3 = pd.read_csv(os.path.join(FU, "A3_farrington_replications.csv"))
A5 = pd.read_csv(os.path.join(FUM, "A5_M2_24m_replications.csv"))

tex = []
csv_files = []


GREEK = {"α": r"\alpha", "β": r"\beta", "δ": r"\delta", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "π": r"\pi",
         "σ": r"\sigma", "τ": r"\tau", "η": r"\eta", "ζ": r"\zeta", "ρ": r"\rho", "φ": r"\phi", "Σ": r"\Sigma"}
MATH_TOKENS = [("log C_i", r"$\log C_i$"), ("C_i", "$C_i$"), ("m_t", "$m_t$"), ("δ_t", r"$\delta_t$"), ("κ_w", r"$\kappa_w$"), ("τ_C", r"$\tau_C$"),
               ("σ_η", r"$\sigma_\eta$"), ("τ_ζ", r"$\tau_\zeta$"), ("m_α", r"$m_\alpha$"), ("w^AHP", r"$w^{\mathrm{AHP}}$"), ("μ̄", r"$\bar{\mu}$"),
               ("α′", r"$\alpha'$"), ("ARL₀", "ARL$_0$"), ("ARL₁", "ARL$_1$"), ("π_1", r"$\pi_1$"), ("π_2", r"$\pi_2$"), ("π_3", r"$\pi_3$"),
               ("R̂", r"$\hat{R}$"), ("λ = z√ν", r"$\lambda = z\sqrt{\nu}$"), ("≤", r"$\le$"), ("≥", r"$\ge$"), ("∞", r"$\infty$"), ("∈", r"$\in$"),
               ("×", r"$\times$"), ("√", r"$\surd$")]


import re as _re
_TOK = {a: b for a, b in MATH_TOKENS}
_PAT = _re.compile("|".join(_re.escape(a) for a in sorted(_TOK, key=len, reverse=True)))


def mathify(s):
    s = _PAT.sub(lambda m: _TOK[m.group(0)], str(s))        # one pass, longest token first
    out, inmath = [], False
    for ch in s:                                              # bare Greek letters outside math -> math
        if ch == "$":
            inmath = not inmath
            out.append(ch)
        elif ch in GREEK and not inmath:
            out.append("$" + GREEK[ch] + "$")
        else:
            out.append(ch)
    return "".join(out)


def esc(s):
    """LaTeX-escape text outside $...$ after converting the notation to math."""
    parts = mathify(s).split("$")
    for i in range(0, len(parts), 2):
        t = parts[i].replace("%", r"\%").replace("&", r"\&").replace("_", r"\_").replace("#", r"\#").replace("{", r"\{").replace("}", r"\}")
        parts[i] = _re.sub(r"(?<![\w.])-(?=\d)", "−", t)      # minus sign before numbers
    return "$".join(parts)


def f(x, d=3):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "–"
        return f"{x:.{d}f}"
    except (TypeError, ValueError):
        return "–"


def mse(t, d=3):
    """(mean, se, n) -> 'mean (se)'; se omitted when undefined."""
    m, s = t[0], t[1]
    if m is None or not np.isfinite(m):
        return "–"
    return f"{m:.{d}f} ({s:.{d}f})" if np.isfinite(s) else f"{m:.{d}f}"


def table(label, caption, header, rows, align=None, size=r"\footnotesize", landscape=False, note=None, csvname=None):
    """One LaTeX table (booktabs); also saves the rows as CSV."""
    n = len(header)
    align = align or ("l" + "r" * (n - 1))
    body = [" & ".join(esc(h) for h in header) + r" \\ \midrule"]
    for r in rows:
        if r == "MIDRULE":
            body.append(r"\midrule")
        elif isinstance(r, tuple) and r[0] == "SUB":
            body.append(r"\multicolumn{%d}{l}{\itshape %s} \\" % (n, esc(r[1])))
        else:
            body.append(" & ".join(esc(c) for c in r) + r" \\")
    env_open = r"\begin{landscape}" if landscape else ""
    env_close = r"\end{landscape}" if landscape else ""
    nl = chr(10)
    note_tex = "" if note is None else (nl + r"\par\vspace{3pt}{\scriptsize " + esc_caption(note) + "}")
    tex.append(env_open + nl + r"\noindent\begin{minipage}{\linewidth}" + nl + r"\captionof{table}{" + esc_caption(caption) + "}" + r"\label{" + label + "}" + nl
               + r"\centering" + size + nl + r"\begin{tabular}{" + align + "}" + nl + r"\toprule" + nl + nl.join(body) + nl
               + r"\bottomrule" + nl + r"\end{tabular}" + note_tex + nl + r"\end{minipage}" + nl + r"\vspace{22pt}" + nl + env_close + nl)
    if csvname:
        pd.DataFrame([r for r in rows if r != "MIDRULE" and not (isinstance(r, tuple) and r[0] == "SUB")], columns=header).to_csv(os.path.join(TAB, csvname), index=False)
        csv_files.append(csvname)


def esc_caption(s):
    return esc(s)


# ------------------------------------------------------------------------------------------------ S1: triage levels
METHOD_LABEL = {"M2 (proposed)": "M2 (proposed)", "M0 Poisson": "Poisson M0", "NB no pooling": "NB no pooling",
                "NB complete pooling": "NB complete pooling", "sigma-rule (plug-in SRI)": "Sigma-rule (plug-in SRI)",
                "EWMA": "EWMA", "CUSUM": "CUSUM", "XGBoost": "XGBoost", "LSTM": "LSTM", "TFT": "TFT"}
ORDER = ["M2 (proposed)", "NB complete pooling", "NB no pooling", "M0 Poisson", "XGBoost", "LSTM", "TFT", "sigma-rule (plug-in SRI)", "EWMA", "CUSUM"]


def farr_row(d):
    def ms(v):
        v = pd.to_numeric(v, errors="coerce").dropna().values
        return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan, int(v.size)) if v.size else (np.nan, np.nan, 0)
    pr = ARL.pooled(d, "farr_arl0", "farr_stable_alarm_units", ARL.UNITS["item"])
    return dict(dr1=ms(d.farr_dr1), dr3=ms(d.farr_dr3), fa_stable=ms(d.farr_fa_stable), lead=ms(d.farr_lead),
                lead_iqr=(d.farr_lead_q25.mean(), d.farr_lead_q75.mean()), arl0=(pr["est"], pr["se"], pr["n"], pr["lo"], pr["hi"]),
                arl1=ms(d.farr_arl1))


def lead_iqr(d, prefix, level):
    """mean of the replication quartiles of the lead time (as Table 6)."""
    recs = []
    return None


def triage_rows(d, level, with_farr=None):
    rows_ = {r["method"]: r for r in T5.table(d, level)}
    out = []
    for m in ORDER:
        r = rows_[m]
        out.append([METHOD_LABEL[m] + (f", level {level}" if m.startswith("sigma") else ""), mse(r["dr1"], 2), mse(r["dr3"], 2), mse(r["fa_stable"], 2), mse(r["lead"], 1)])
    if with_farr is not None:
        fr = farr_row(with_farr)
        out.append(["Farrington-type threshold", mse(fr["dr1"], 2), mse(fr["dr3"], 2), mse(fr["fa_stable"], 2), mse(fr["lead"], 1)])
    return out


for level, cost in ((1, "1:20"), (2, "1:5"), (3, "1:1")):
    rows = triage_rows(B, level, A3)
    table(f"tab:S1-{level}", f"Table S1.{level}. Triage at level {level} (cost ratio {cost}, threshold $\\pi_{level}$ = { {1: '0.048', 2: '0.167', 3: '0.5'}[level] }, two consecutive months), Study B pooled over the twelve cells (65 replications; mean over replications and Monte Carlo standard error in brackets): share of shocks flagged within one and within three months before they occur, false alarms per item-year in the stable windows and median lead time in months. The sigma-rule is shown at its own level {level}; EWMA, CUSUM and the Farrington-type threshold have one level, so their rows repeat across the three tables. {'This table repeats Table 6(a) of the main text for comparison.' if level == 2 else 'Level 2 is Table 6(a) of the main text (Table S1.2).'}",
          ["Method", "Detected ≤ 1 month", "Detected ≤ 3 months", "False alarms per item-year", "Lead time (median)"], rows, csvname=f"S1_triage_level{level}.csv")

tex.append(r"\clearpage")
# ------------------------------------------------------------------------------------------------ S2: by regime
for reg in ("stable", "gradual", "abrupt"):
    d = B[B.regime == reg]
    rows_ = {r["method"]: r for r in T5.table(d, 2)}
    rows = []
    for m in ORDER:
        r = rows_[m]
        arl0 = (f"{r['arl0'][0]:.1f}" if r["arl0"][0] < 10 else f"{r['arl0'][0]:.0f}") if "arl0" in r and np.isfinite(r["arl0"][0]) else "–"
        arl1 = mse(r["arl1"], 1) if "arl1" in r and reg != "stable" else "–"
        rows.append([METHOD_LABEL[m] + (", level 2" if m.startswith("sigma") else ""), mse(r["aucpr"]), mse(r["brier"]), mse(r["ece"]), mse(r["dr1"], 2), mse(r["dr3"], 2), mse(r["fa_stable"], 2), mse(r["lead"], 1), arl0, arl1,
                     mse(r["conf_cover"]) if "conf_cover" in r else "–", mse(r["conf_size"], 2) if "conf_size" in r else "–"])
    fr = farr_row(A3[A3.regime == reg])
    rows.append(["Farrington-type threshold", "–", "–", "–", mse(fr["dr1"], 2), mse(fr["dr3"], 2), mse(fr["fa_stable"], 2), mse(fr["lead"], 1), f"{fr['arl0'][0]:.0f}", mse(fr["arl1"], 1) if reg != "stable" else "–", "–", "–"])
    n = int(len(d))
    note = ("In the stable cells the whole monitoring span (82 decision months) is a stable window and no change occurs, so ARL$_1$ is undefined; the detection and false-alarm columns refer to the shocks and stable windows of these cells."
            if reg == "stable" else "Stable windows are the 24 months before the change at month 60; ARL$_1$ is the delay from the change to the first (persistent) alarm.")
    table(f"tab:S2-{reg}", f"Table S2.{dict(stable=1, gradual=2, abrupt=3)[reg]}. Alarm performance in the {reg} cells of Study B (four cells, {n} replications; triage at level 2; otherwise as Table 6). ARL₀ is pooled over replications (stable unit-months over alarm months); for the sigma-rule the event columns are at its level 2 and the run lengths at its first level, as in Table 6.",
          ["Method", "AUC-PR", "Brier", "ECE", "Det. ≤1 mo", "Det. ≤3 mo", "FA/item-yr", "Lead", "ARL₀", "ARL₁", "Set cov.", "Set size"], rows,
          size=r"\scriptsize", landscape=True, note=note, csvname=f"S2_table6_{reg}.csv")

# ------------------------------------------------------------------------------------------------ S2.4: ARL0 decomposition
def _decomp(d, col, share, unit):
    size = ARL.window_size(d.regime.values, ARL.UNITS[unit]).astype(float)
    al = ARL.alarm_months(d[col].values, d[share].values, size)
    ok = np.isfinite(al)
    return float(size[ok].sum()), float(al[ok].sum())


rows = []
chans = [("q channel, M2", "M2_q_arl0", "M2_q_stable_alarm_units", "q"), ("q channel, NB complete pooling", "NBcomplete_q_arl0", "NBcomplete_q_stable_alarm_units", "q"),
         ("q channel, NB no pooling", "NBnone_q_arl0", "NBnone_q_stable_alarm_units", "q"), ("q channel, Poisson M0", "M0_q_arl0", "M0_q_stable_alarm_units", "q"),
         ("Sigma-rule, level 1", "sigma_arl0", "sigma_stable_alarm_units", "item"), ("EWMA", "ewma_arl0", "ewma_stable_alarm_units", "item"), ("CUSUM", "cusum_arl0", "cusum_stable_alarm_units", "item")]
for lab, col, share, unit in chans:
    line = [lab]
    tu = ta = 0.0
    for reg in ("stable", "gradual", "abrupt"):
        u, a = _decomp(B[B.regime == reg], col, share, unit)
        tu += u; ta += a
        line.append(f"{u:.0f} / {a:.0f} = {u / max(a, 1):.1f}")
    line.append(f"{tu:.0f} / {ta:.0f} = {tu / max(ta, 1):.1f}")
    rows.append(line)
line = ["Farrington-type threshold"]; tu = ta = 0.0
for reg in ("stable", "gradual", "abrupt"):
    u, a = _decomp(A3[A3.regime == reg], "farr_arl0", "farr_stable_alarm_units", "item")
    tu += u; ta += a
    line.append(f"{u:.0f} / {a:.0f} = {u / max(a, 1):.1f}")
line.append(f"{tu:.0f} / {ta:.0f} = {tu / max(ta, 1):.1f}")
rows.append(line)
table("tab:S2-4", "Table S2.4. How the pooled ARL₀ of Table 6(b) arises from the regimes: stable-window unit-months / alarm months = ARL₀ for each change channel and regime, and their sums over the twelve cells. Unit-months are groups × stable decision months × replications for the q channels (3 × 82 × 20 in the stable cells, 3 × 24 × 25 and 3 × 24 × 20 in the gradual and abrupt cells) and items × months × replications (30 × …) for the item-level charts; the pooled value is the ratio of the summed unit-months to the summed alarm months, i.e. the unit-month-weighted harmonic mean of the regime values, not their arithmetic mean.",
      ["Channel", "Stable (u-mo / alarm-mo)", "Gradual", "Abrupt", "All twelve cells (Table 6(b))"], rows, size=r"\scriptsize", csvname="S2d_arl0_decomposition.csv")

# ------------------------------------------------------------------------------------------------ S3: per cell
cells = pd.read_csv(os.path.join(AN, "B_cells.csv")).sort_values(["regime", "alpha", "mu"])
meths = [("M2", "M2"), ("NBcomplete", "NB compl."), ("NBnone", "NB none"), ("M0", "M0"), ("xgb", "XGBoost"), ("lstm", "LSTM"), ("tft", "TFT")]
rows = []
for _, c in cells.iterrows():
    rows.append([f"{c.regime}, α′ = {c.alpha:g}, μ̄ = {c.mu:g}", int(c.n), f(c.M2_base_rate, 2)] + [f(c[f"{k}_aucpr"]) for k, _ in meths])
table("tab:S3a", "Table S3.1. AUC-PR of the event probabilities by cell of Study B (mean over the replications of the cell; n = replications). The event share is the share of item-months with at least one shock within three months (not the base rate $\\bar{\\mu}$ of Table 4).",
      ["Cell", "n", "Event share"] + [lab for _, lab in meths], rows, csvname="S3a_aucpr_by_cell.csv")
rows = []
for _, c in cells.iterrows():
    rows.append([f"{c.regime}, α′ = {c.alpha:g}, μ̄ = {c.mu:g}"] + [f"{c[f'{k}_tri2_dr1']:.2f} / {c[f'{k}_tri2_fa_stable']:.2f}" for k, _ in meths])
table("tab:S3b", "Table S3.2. Triage at level 2 by cell of Study B: share of shocks flagged within one month / false alarms per item-year in the stable windows (means over the replications of the cell).",
      ["Cell"] + [lab for _, lab in meths], rows, size=r"\scriptsize", csvname="S3b_triage_by_cell.csv")

# ------------------------------------------------------------------------------------------------ S4: Study A by factor
rows = []
for fac, lab, fmt in (("alpha", "α′", lambda v: "∞" if str(v) == "inf" else f"{float(v):g}"), ("mu", "μ̄", lambda v: f"{float(v):g}"), ("regime", "regime", str), ("I", "I", lambda v: f"{int(v)}"), ("T", "T", lambda v: f"{int(v)}")):
    d = pd.read_csv(os.path.join(AN, f"A_hier_by_{fac}.csv"))
    if fac == "alpha":
        d = d.assign(_k=d.alpha.astype(str).map(lambda v: 1e9 if v == "inf" else float(v))).sort_values("_k")
    rows.append(("SUB", f"by {lab}"))
    for _, r in d.iterrows():
        rows.append([f"{lab} = {fmt(r[fac])}", int(r.n), f(r.failed, 3), f(r.alpha_cover, 2), f(r.beta_cover, 2), f(r.w_cover, 2), f(r.kappa_w_cover, 2), f(r.logC_cover, 2), f(r.m_t_cover, 2), f(r.delta_t_cover, 2), f(r.logC_rmse, 3), f(r.m_t_rmse, 3)])
table("tab:S4", "Table S4. Study A, hierarchical model (538 fits): realised coverage of 90% posterior intervals and RMSE by factor level (all fits of the level, including the 4.6% that failed a convergence check; n = fits). Table 5 of the main text gives the same quantities by problem size.",
      ["Level", "n", "Failed", "α′", "β", "w", "κ_w", "log C", "m_t", "δ_t", "RMSE log C", "RMSE m_t"], rows, size=r"\scriptsize", csvname="S4_studyA_by_factor.csv")

# ------------------------------------------------------------------------------------------------ S5: pooling comparison (Fig. 4)
def load_recs(cell, variant, root=RES):
    p = os.path.join(root, "A", f"{cell}__{variant}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


rows = []
for (I, T) in ((10, 60), (30, 120)):
    rows.append(("SUB", f"(I, T) = ({I}, {T})"))
    for mu in ("0.05", "0.2", "0.5"):
        cell = f"A-a2-mu{mu}-gradual-I{I}-T{T}"
        for variant, lab in (("hier", "hierarchical"), ("complete", "complete pooling"), ("none", "no pooling")):
            rr = load_recs(cell, variant)
            if not rr:
                continue
            a50 = np.array([x["recovery"]["alpha"]["q50"] for x in rr]).ravel(); a05 = np.array([x["recovery"]["alpha"]["q05"] for x in rr]).ravel(); a95 = np.array([x["recovery"]["alpha"]["q95"] for x in rr]).ravel()
            geo = float(np.exp(np.log(a50).mean()))
            rows.append([f"μ̄ = {mu}, {lab}", len(rr), f(geo, 2), f"{100 * (geo / 2 - 1):+.0f}%", f(((a05 <= 2) & (2 <= a95)).mean(), 2), f(np.mean(np.log(a95) - np.log(a05)), 2),
                         f(np.mean([x["recovery"]["logC_summary"]["cover"] for x in rr]), 2), f(np.mean([x["recovery"]["logC_summary"]["rmse"] for x in rr]), 3), int(sum(x["failed"] for x in rr))])
table("tab:S5", "Table S5. Pooling comparison of Section 5.2 (i) and Fig. 4 (gradual regime, α′ = 2): posterior median of α′ (geometric mean over replications and, for the hierarchical and no-pooling variants, over groups or items; truth 2), its relative error, realised coverage of the 90% interval, mean log-scale interval width, and coverage and RMSE of log C_i, by base rate, size and variant (n = replications; the reference cell has 20).",
      ["Cell, variant", "n", "α′ median", "Rel. error", "α′ coverage", "log width", "log C coverage", "RMSE log C", "Failed fits"], rows, size=r"\scriptsize", csvname="S5_pooling_fig6.csv")

tex.append(r"\clearpage")
# ------------------------------------------------------------------------------------------------ S6: A1 priors
import a1_priors as A1
recs = [json.loads(l) for l in open(os.path.join(FU, "A1_priors.jsonl"))]
base = [x for x in load_recs("A-a2-mu0.2-gradual-I30-T120", "hier") if x["rep"] in A1.REPS]
ARM_LABEL = {"m_alpha_log1": "m_α centred at log 1", "m_alpha_log5": "m_α centred at log 5", "innov_x2": "innovation scales × 2 (half-N(0, 0.2))",
             "innov_x5": "innovation scales × 5 (half-N(0, 0.5))", "ahp_reversed": "AHP prior reversed, w^AHP = (0.375, 0.625)"}
rows = []
for arm, rr in [("baseline (manuscript priors)", base)] + [(a, [x for x in recs if x["arm"] == a]) for a in A1.ARMS]:
    s = A1._arm_stats(rr)
    w50 = np.array([x["recovery"]["w"]["q50"] for x in rr]).mean(0)
    rows.append([ARM_LABEL.get(arm, arm), s["n"], s["failed"], f(s["alpha_geo"], 2), f(s["alpha_cover"], 2), f(s["alpha_logwidth"], 2), f"({w50[0]:.2f}, {w50[1]:.2f})", f(s["w_cover"], 2), f(s["kappa_med"], 2),
                 f"{s['beta_bias']:+.3f}", f(s["beta_cover"], 2), f(s["logC_rmse"], 3), f(s["logC_cover"], 2), f(s["m_rmse"], 3), f(s["m_cover"], 2), f(s["sig_eta"], 3), f(s["tau_zeta"], 4)])
table("tab:S6", "Table S6. Prior sensitivity (Section 5.4): the hierarchical model refitted to the data of replications 0–4 of the reference cell (α′ = 2, μ̄ = 0.2, gradual, I = 30, T = 120) under five alternative priors; everything else as in Study A. Posterior medians averaged over replications (α′ as a geometric mean over replications and groups), realised coverage of 90% intervals, mean log-scale width of the α′ interval, mean bias of β, RMSE and coverage of log C_i and of the level path m_t, and posterior medians of σ_η and τ_ζ. Truth: α′ = 2, β = (0.4, 0.3, 0.2), w = (0.5, 0.3), κ_w = 0.8.",
      ["Arm", "n", "Failed", "α′", "α′ cov.", "α′ log width", "w", "w cov.", "κ_w", "β bias", "β cov.", "RMSE log C", "log C cov.", "RMSE m_t", "m_t cov.", "σ_η", "τ_ζ"], rows, size=r"\scriptsize", landscape=True, csvname="S6_prior_sensitivity.csv")

tex.append(r"\clearpage")
# ------------------------------------------------------------------------------------------------ S7: A7 tau_C
recs7 = [json.loads(l) for l in open(os.path.join(FU, "A7_tauC.jsonl"))]
rows = []
for variant, lab in (("complete", "complete pooling"), ("hier", "hierarchical")):
    for tau in (0.0, 0.3, 0.6):
        if tau == 0.3:
            rr = [x for x in load_recs("A-a2-mu0.2-gradual-I30-T120", variant) if x["rep"] in (0, 1, 2, 3, 4)]
        else:
            rr = [x for x in recs7 if x["variant"] == variant and x["tau_C"] == tau]
        if not rr:
            continue
        a50 = np.array([x["recovery"]["alpha"]["q50"] for x in rr]).ravel(); a05 = np.array([x["recovery"]["alpha"]["q05"] for x in rr]).ravel(); a95 = np.array([x["recovery"]["alpha"]["q95"] for x in rr]).ravel()
        geo = float(np.exp(np.log(a50).mean()))
        pt = {0.0: "2.00", 0.3: "1.61", 0.6: "1.00"}[tau] if variant == "complete" else "–"
        rows.append([lab, f"{tau:.1f}", len(rr), int(sum(x["failed"] for x in rr)), f(geo, 2), f"{100 * (geo / 2 - 1):+.0f}%", f"{a50.min():.2f}–{a50.max():.2f}", f(((a05 <= 2) & (2 <= a95)).mean(), 2),
                     f(np.mean(np.log(a95) - np.log(a05)), 2), f(np.mean([x["recovery"]["logC_summary"]["cover"] for x in rr]), 2), f(np.mean([x["recovery"]["logC_summary"]["rmse"] for x in rr]), 2), pt])
table("tab:S7", "Table S7. Complete pooling and the item-effect scale τ_C (Section 5.2 (i)): replications 0–4 of the reference cell re-drawn with the item effects scaled to τ_C ∈ {0, 0.6} (τ_C = 0.3 is the Study A design), fitted with complete pooling and with the hierarchical model. Posterior median of α′ (geometric mean over replications and, for the hierarchical model, groups; truth 2), its range, 90% coverage and log-scale width, and coverage and RMSE of log C_i. The last column is the pseudo-true dispersion of complete pooling — the value that maximises the expected log-likelihood of the misspecified model under the true process on the item means of the cell, with the level free (1.56 and 0.87 by variance matching).",
      ["Model", "τ_C", "n", "Failed", "α′ median", "Rel. error", "Range", "α′ coverage", "log width", "log C coverage", "RMSE log C", "Pseudo-true α′"], rows, size=r"\scriptsize", csvname="S7_tauC.csv")

# ------------------------------------------------------------------------------------------------ S8: A2 beta = 0
r2 = [json.loads(l) for l in open(os.path.join(FUM, "R", "R-beta0__latent.jsonl"))]
ref = json.load(open(os.path.join(CODE, "followup", "ref", "reference_numbers.json")))["B_reference_cell"]
b50 = np.array([x["recovery"]["beta"]["q50"] for x in r2]); b05 = np.array([x["recovery"]["beta"]["q05"] for x in r2]); b95 = np.array([x["recovery"]["beta"]["q95"] for x in r2])
w50 = np.array([x["recovery"]["w"]["q50"] for x in r2]).mean(0); k50 = np.mean([x["recovery"]["kappa_w"]["q50"] for x in r2])
rowsA2 = []
for x in r2:
    o = AN_.model_row("M2", x["M2"]); o["regime"] = "gradual"
    for ch in ("ewma", "cusum"):
        c = x["charts"][ch]
        o.update({f"{ch}_dr1": c["events"]["dr_h1"], f"{ch}_fa_stable": c["events"]["fa_per_itemyear_stable"], f"{ch}_arl0": c["run_length"]["arl0"], f"{ch}_arl1": c["run_length"].get("arl1"), f"{ch}_stable_alarm_units": c["run_length"].get("stable_alarm_units")})
    rowsA2.append(o)
d2 = pd.DataFrame(rowsA2)
def ms(v):
    v = pd.to_numeric(v, errors="coerce").dropna().values
    return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan, int(v.size))
pr2 = ARL.pooled(d2, "M2_q_arl0", "M2_q_stable_alarm_units", ARL.UNITS["q"])
rows = [["β posterior median (truth 0)", f"({b50.mean(0)[0]:+.3f}, {b50.mean(0)[1]:+.3f}, {b50.mean(0)[2]:+.3f})", "(0.405, 0.302, 0.190) for a truth of (0.4, 0.3, 0.2)"],
        ["90% intervals of β covering the truth", f"{((b05 <= 0) & (0 <= b95)).mean():.2f}", "0.93"],
        ["w posterior median (truth (0.5, 0.3))", f"({w50[0]:.2f}, {w50[1]:.2f})", "(0.53, 0.31)"],
        ["κ_w posterior median (truth 0.8)", f"{k50:.2f}", "0.84"],
        ["Recovery fits failed", f"{sum(x['recovery_fit']['failed'] for x in r2)} of {len(r2)}", "0 of 5"],
        "MIDRULE",
        ["AUC-PR", mse(ms(d2.M2_aucpr)), mse(tuple(ref["M2_aucpr"]))], ["Brier score", mse(ms(d2.M2_brier)), mse(tuple(ref["M2_brier"]))], ["ECE", mse(ms(d2.M2_ece)), mse(tuple(ref["M2_ece"]))],
        ["Event share", mse(ms(d2.M2_base_rate)), mse(tuple(ref["M2_base_rate"]))],
        ["Detected ≤ 1 month (level 2)", mse(ms(d2.M2_tri2_dr1), 2), mse(tuple(ref["M2_tri2_dr1"]), 2)], ["Detected ≤ 3 months", mse(ms(d2.M2_tri2_dr3), 2), mse(tuple(ref["M2_tri2_dr3"]), 2)],
        ["False alarms per item-year", mse(ms(d2.M2_tri2_fa_stable), 2), mse(tuple(ref["M2_tri2_fa_stable"]), 2)], ["Lead time (median, months)", mse(ms(d2.M2_tri2_lead), 1), mse(tuple(ref["M2_tri2_lead"]), 1)],
        ["ARL₀ of the q channel (pooled, 90% interval)", f"{pr2['est']:.0f} ({pr2['lo']:.0f}–{pr2['hi']:.0f})", f"{ref['M2_q_arl0_pooled'][0]:.0f} ({ref['M2_q_arl0_pooled'][1]:.0f}–{ref['M2_q_arl0_pooled'][2]:.0f})"],
        ["ARL₁ of the q channel", mse(ms(d2.M2_q_arl1), 1), mse(tuple(ref["M2_q_arl1"]), 1)],
        "MIDRULE"]
for ch, lab in (("ewma", "EWMA"), ("cusum", "CUSUM")):
    pr = ARL.pooled(d2, f"{ch}_arl0", f"{ch}_stable_alarm_units", ARL.UNITS["item"])
    rows.append([f"{lab}: detected ≤ 1 month / false alarms / ARL₀", f"{d2[ch + '_dr1'].mean():.2f} / {d2[ch + '_fa_stable'].mean():.2f} / {pr['est']:.1f}", f"{ref[ch + '_dr1'][0]:.2f} / {ref[ch + '_fa_stable'][0]:.2f} / {ref[ch + '_arl0_pooled'][0]:.1f}"])
table("tab:S8", "Table S8. Indicators without information (Section 5.4): the robustness protocol (recovery fit, M2 alarm pipeline, EWMA and CUSUM) on the reference design with β = 0 in the truth, five replications, against the Study B reference cell (β = (0.4, 0.3, 0.2), ten replications). Means over replications with Monte Carlo standard errors in brackets; the recovery rows of the reference column are the Study A hierarchical fits of replications 0–4 of the reference cell (the baseline of Table S6).",
      ["Quantity", "β = 0 (5 replications)", "Reference (β = (0.4, 0.3, 0.2))"], rows, align="lll", size=r"\footnotesize", csvname="S8_beta0.csv")

# ------------------------------------------------------------------------------------------------ S9: A3 per cell
rows = []
for (reg, a, mu), d in A3.groupby(["regime", "alpha", "mu"]):
    fr = farr_row(d)
    rows.append([f"{reg}, α′ = {a:g}, μ̄ = {mu:g}", len(d), mse(fr["dr1"], 2), mse(fr["dr3"], 2), mse(fr["fa_stable"], 2), f"{fr['lead'][0]:.1f} ({fr['lead_iqr'][0]:.1f}–{fr['lead_iqr'][1]:.1f})", f"{fr['arl0'][0]:.1f} ({fr['arl0'][3]:.1f}–{fr['arl0'][4]:.1f})", mse(fr["arl1"], 1) if reg != "stable" else "–", f"{d.thr_mean.mean():.2f}", f"{d.mu_hat_mean.mean():.3f}"])
fr = farr_row(A3)
rows.append("MIDRULE")
rows.append(["All twelve cells", len(A3), mse(fr["dr1"], 2), mse(fr["dr3"], 2), mse(fr["fa_stable"], 2), f"{fr['lead'][0]:.1f} ({fr['lead_iqr'][0]:.1f}–{fr['lead_iqr'][1]:.1f})", f"{fr['arl0'][0]:.1f} ({fr['arl0'][3]:.1f}–{fr['arl0'][4]:.1f})", mse(fr["arl1"], 1), f"{A3.thr_mean.mean():.2f}", f"{A3.mu_hat_mean.mean():.3f}"])
table("tab:S9", "Table S9. Farrington-type threshold (Section 5.1, Table 6) by cell of Study B: a rolling 36-month quasi-Poisson regression per item (intercept and a linear trend retained when significant at 5%, one Anscombe down-weighting of past outliers), a negative-binomial predictive with the estimated dispersion, and an alarm when the current month's count exceeds its (1 − 1/24) quantile. Detection shares, false alarms per item-year, median lead time (mean of the replication quartiles in brackets), pooled ARL₀ with its 90% bootstrap interval, ARL₁, and the mean threshold and fitted mean.",
      ["Cell", "n", "Det. ≤ 1 mo", "Det. ≤ 3 mo", "FA / item-year", "Lead (IQR)", "ARL₀ (90%)", "ARL₁", "Threshold", "Fitted μ"], rows, size=r"\scriptsize", csvname="S9_farrington_by_cell.csv")

# ------------------------------------------------------------------------------------------------ S10: A5 24-month schedule
rows = []
pr24 = ARL.pooled(A5, "M2_q_arl0", "M2_q_stable_alarm_units", ARL.UNITS["q"]); pr6 = ARL.pooled(B, "M2_q_arl0", "M2_q_stable_alarm_units", ARL.UNITS["q"])
A5i = A5.set_index(["cell", "rep"]); Bi = B.set_index(["cell", "rep"]).loc[A5i.index]
for k, lab, dgt in (("M2_aucpr", "AUC-PR", 3), ("M2_brier", "Brier score", 3), ("M2_ece", "ECE", 3), ("M2_tri2_dr1", "Detected ≤ 1 month (level 2)", 3), ("M2_tri2_dr3", "Detected ≤ 3 months", 3),
                    ("M2_tri2_fa_stable", "False alarms per item-year", 3), ("M2_tri2_lead", "Lead time (median, months)", 1), ("M2_q_arl1", "ARL₁ of the q channel", 1), ("M2_conf_cover", "Conformal coverage", 3), ("M2_conf_size", "Conformal set size", 2)):
    x = A5i[k] - Bi[k]
    diff = f"{x.mean():+.{dgt}f}".replace("-0." + "0" * dgt, "0." + "0" * dgt).replace("+0." + "0" * dgt, "0." + "0" * dgt)
    rows.append([lab, mse(ms(A5[k]), dgt), mse(ms(B[k]), dgt), f"{diff} ({x.std(ddof=1) / np.sqrt(x.notna().sum()):.{dgt}f})"])
rows.append(["ARL₀ of the q channel (pooled, 90% interval)", f"{pr24['est']:.0f} ({pr24['lo']:.0f}–{pr24['hi']:.0f})", f"{pr6['est']:.0f} ({pr6['lo']:.0f}–{pr6['hi']:.0f})", "–"])
rows.append(["Origin fits that failed after the refit", f"{100 * A5.M2_fit_failed_origins.mean() / 4:.1f}%", f"{100 * B.M2_fit_failed_origins.mean() / 14:.1f}%", "–"])
rows.append(["Fitting time per replication (s)", f"{A5.M2_fit_seconds.mean():.0f}", f"{B.M2_fit_seconds.mean():.0f}", "–"])
table("tab:S10a", "Table S10.1. Refit schedule (Section 5.3 (iv)): M2 refitted every 24 months (origins 36, 60, 84, 108, the schedule of the model-based baselines) against the six-month schedule of Table 6, on the same 65 Study B replications (same data and same origin-36 fit; the particle filter propagates the posterior between refits). Means over replications with Monte Carlo standard errors; the last column is the mean paired difference (24-month minus six-month) with its standard error.",
      ["Quantity", "24-month refit", "Six-month refit (Table 6)", "Paired difference"], rows, align="llll", size=r"\footnotesize", csvname="S10a_refit_schedule.csv")
rows = []
for reg in ("stable", "gradual", "abrupt"):
    d24, d6 = A5[A5.regime == reg], B[B.regime == reg]
    p24 = ARL.pooled(d24, "M2_q_arl0", "M2_q_stable_alarm_units", ARL.UNITS["q"]); p6 = ARL.pooled(d6, "M2_q_arl0", "M2_q_stable_alarm_units", ARL.UNITS["q"])
    rows.append([reg, len(d24), f"{d24.M2_aucpr.mean():.3f}", f"{d6.M2_aucpr.mean():.3f}", f"{d24.M2_tri2_dr1.mean():.3f}", f"{d6.M2_tri2_dr1.mean():.3f}", f"{d24.M2_tri2_fa_stable.mean():.3f}", f"{d6.M2_tri2_fa_stable.mean():.3f}",
                 f"{p24['est']:.0f} ({p24['lo']:.0f}–{p24['hi']:.0f})", f"{p6['est']:.0f} ({p6['lo']:.0f}–{p6['hi']:.0f})", f"{d24.M2_q_arl1.mean():.1f}" if reg != "stable" else "–", f"{d6.M2_q_arl1.mean():.1f}" if reg != "stable" else "–"])
table("tab:S10b", "Table S10.2. Refit schedule by regime: the 24-month (24) and six-month (6) schedules of Table S10.1 within the stable, gradual and abrupt cells (means over the replications of the regime; ARL₀ pooled with its 90% interval).",
      ["Regime", "n", "AUC-PR 24", "AUC-PR 6", "Det. ≤1 mo 24", "Det. ≤1 mo 6", "FA 24", "FA 6", "ARL₀ 24", "ARL₀ 6", "ARL₁ 24", "ARL₁ 6"], rows, size=r"\scriptsize", csvname="S10b_refit_by_regime.csv")


tex.append(r"\clearpage")
# ------------------------------------------------------------------------------------------------ S11: A6 horseshoe2
def a6_stats(rr):
    cp = [x["changepoint"] for x in rr]
    return dict(n=len(rr), failed=sum(x["failed"] for x in rr), rhat=sum(x["rhat_max"] >= 1.01 for x in rr), ess=sum(min(x["ess_bulk_min"], x["ess_tail_min"]) <= 400 for x in rr), div=sum(x["n_div"] > 0 for x in rr),
                rhat_med=float(np.median([x["rhat_max"] for x in rr])), ess_med=float(np.median([min(x["ess_bulk_min"], x["ess_tail_min"]) for x in rr])),
                err=float(np.mean([c["abs_err"] for c in cp])), cover=float(np.mean([c["cover"] for c in cp])), pmax=float(np.mean([c["max_prob"] for c in cp])), pmin=min(c["max_prob"] for c in cp), pmx=max(c["max_prob"] for c in cp),
                m_rmse=float(np.mean([x["recovery"]["m_t"]["rmse"] for x in rr])), m_cover=float(np.mean([x["recovery"]["m_t"]["cover"] for x in rr])), steps=float(np.median([x["attempts"][-1]["mean_steps"] for x in rr])))
rows = []
for cell, mu in (("A-a2-mu0.2-abrupt-I30-T120", "0.2"), ("A-a2-mu0.5-abrupt-I30-T120", "0.5")):
    rows.append(("SUB", f"abrupt, α′ = 2, μ̄ = {mu}, (I, T) = (30, 120)"))
    for lab, rr in (("Gaussian innovations (Study A)", load_recs(cell, "hier")), ("horseshoe, 2,000 draws, target 0.95 (Study A)", load_recs(cell, "horseshoe")),
                    ("horseshoe, reparameterised, 4,000 draws, target 0.99", [json.loads(l) for l in open(os.path.join(FUM, "A6", f"{cell}__horseshoe2.jsonl"))])):
        s = a6_stats(rr)
        rows.append([lab, s["n"], s["failed"], s["rhat"], s["ess"], s["div"], f(s["rhat_med"], 4), f"{s['ess_med']:.0f}", f(s["err"], 2), f(s["cover"], 2), f"{s['pmax']:.2f} ({s['pmin']:.2f}–{s['pmx']:.2f})", f(s["m_rmse"], 3), f(s["m_cover"], 2), f"{s['steps']:.0f}"])
table("tab:S11", "Table S11. Abrupt changes (Section 5.2 (iii)): ten replications of the two strong-data abrupt cells fitted with Gaussian innovations, with the regularised horseshoe of Study A (four chains of 400 warm-up and 500 draws, target acceptance 0.95) and with the same horseshoe prior reparameterised as λ = z√ν (z half-normal, ν inverse-gamma(1/2, 1/2)) and sampled with 1,000 warm-up and 1,000 draws per chain at target acceptance 0.99, tree depth 10, refit at the same settings. Fits failing the convergence check after the refit, by reason (R̂ ≥ 1.01, bulk or tail ESS ≤ 400, divergent transitions), median of the worst R̂ and of the minimum ESS; change-date statistics (mean absolute error of the posterior median date in months, coverage of the 90% interval, mean and range over replications of the posterior probability of the true month in the best-identified group); RMSE and coverage of the level path; median leapfrog steps per iteration.",
      ["Innovations and sampler", "n", "Failed", "R̂", "ESS", "Div.", "R̂ max (med.)", "ESS min (med.)", "|Date error|", "Date cov.", "P(true month)", "RMSE m_t", "m_t cov.", "Steps"], rows, size=r"\scriptsize", landscape=True, csvname="S11_horseshoe.csv")

tex.append(r"\clearpage")
# ------------------------------------------------------------------------------------------------ S12: fit diagnostics
rows = []
for line in open(os.path.join(AN, "fitstats.txt")).read().strip().splitlines()[1:]:
    if line.startswith("rhat/ess"):
        continue
    parts = line.split()
    if parts[0] == "B":
        rows.append([" ".join(parts[:2]), parts[2], parts[3], "–", "–", "–", parts[4], "–", parts[5] + " (per origin, incl. refits)", "–"])
    else:
        name = " ".join(parts[:3]); vals = parts[3:]
        rows.append([name] + vals)
table("tab:S12", "Table S12. Fit diagnostics (Section 5.4) by study and model: number of fits, share passing the convergence check at the first attempt, share of the failed first attempts with each reason (a fit can have several), share still failing after the one refit, mean leapfrog steps per iteration, and median seconds of a first fit and of a refit on one CPU core. Study B counts origin fits.",
      ["Group", "Fits", "Pass 1st", "R̂", "ESS", "Div.", "Final fail", "Steps", "s / fit", "s / refit"], rows, size=r"\scriptsize", csvname="S12_fit_diagnostics.csv")

# ------------------------------------------------------------------------------------------------ document
preamble = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=20mm]{geometry}
\usepackage{fontspec}
\setmainfont{TeX Gyre Pagella}
\usepackage{booktabs,pdflscape,caption,array,url,amsmath,graphicx}
\setlength{\tabcolsep}{3.5pt}
\captionsetup{font=small,labelfont=bf,justification=justified,singlelinecheck=false,labelformat=empty,skip=4pt}
\usepackage{unicode-math}
\setmathfont{Latin Modern Math}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\pagestyle{plain}
\begin{document}
"""
intro = r"""{\large\bfseries Supplementary material: extended version}

{\bfseries """ + esc(TITLE) + r"""}

""" + esc(SUBTITLE) + r"""

\textbf{Numbering.} This document is the extended version of the supplementary material. It keeps its own numbers for notes, tables and figures (Supplementary Notes 1--17, Tables S1--S29 and Figs.~S1--S4). Its references to sections, tables, figures and equations of the main text follow the longer version of the manuscript for which it was written. The supplementary material of the manuscript is the shorter document in \path{supplementary/}. The README files of the repository convert between the two sets of numbers.

This supplement gives, in Supplementary Notes 1--6, the implementation details that the main text summarises (sampler settings and convergence criteria, baseline specifications, the pseudo-true dispersion of the pooling comparison, the refit schedule, prior and indicator sensitivity, computational cost), and tabulates results that the main text cites but does not print: the triage results at levels 1 and 3 (Section 4.4, Section 5.3 (ii)), the Study B results by regime and by cell (Section 5.3), the Study A coverages by factor level and the pooling comparison behind Fig.~4 (Section 5.2), the sensitivity runs of Sections 5.2 (i), 5.2 (ii), 5.2 (iii), 5.3 (iii), 5.3 (iv) and 5.4 (prior sensitivity, the item-effect scale $\tau_C$, dispersion that differs across groups, uninformative indicators, the Farrington-type threshold by cell, the control charts designed under the negative binomial, the 24-month refit schedule, the reparameterised horseshoe and the severity submodel), and the fit diagnostics of Section 5.4. Supplementary Notes 7--17 hold material that the main text refers to but does not print: the yield-model lineage and the gamma--Poisson family with the source model and the variable correspondence (Note 7, Table S17, Figs.~S1 and S4), the severity submodel (Note 8), the industry instances of the intensity--exposure--dispersion structure (Note 9, Table S18, Fig.~S2), the expected-cost analysis of Section 5.6 with standard errors, paired comparisons, break-even ratios and the alternative cost model (Note 10, Tables S19.1--S22), the numerical verification of the Hawkes window-count dispersion of Supplementary Note 16 (Note 11, Table S23, Fig.~S3), the alarm metrics of the robustness arms of Section 5.4 (Note 12, Table S24), the modelling choices and parameter-recovery details of Sections 3.3 and 5.2 (Note 13), the backtest on the Korean import panel of Section 5.5---data sources, event construction, the event-definition decision, alarm metrics at every level and target, the export-restriction implementations one by one, and the sensitivity of the results to the event definitions (Note 14, Tables S25--S28)---the construction of the conformal prediction sets of Section 4.3 (Note 15), the derivation of the Hawkes window-count dispersion of Section 3.4 (Note 16), and the rare arm of the simulation study of Sections 5.1, 5.3 and 5.6 (Note 17, Tables S29.1--S29.4). All numbers are computed from the archived replication-level result files of the repository with the scripts listed at the end; conventions are those of the main text.

\textbf{Conventions.} Entries are means over replications with Monte Carlo standard errors in brackets unless stated otherwise. ARL$_0$ is the pooled ratio of stable-window unit-months to alarm months over the replications --- so a value pooled over cells is the unit-month-weighted harmonic mean of the cell or regime values, not their arithmetic mean (Table S2.4 shows the sums) --- with a 90\% bootstrap interval over replications (units: groups for the $q$ channels, items for the sigma-rule, EWMA, CUSUM and the Farrington-type threshold; stable windows: the 82 decision months of a stable cell, the 24 decision months before the change at month 60 elsewhere). ARL$_1$ is the mean delay from the change to the first alarm (first persistent alarm for the $q$ channels), censored at the end of monitoring. Detection within $h$ months counts a shock as detected when the level is in force in the month before it; false alarms are alarm onsets in stable windows not followed by a shock within three months, per item-year. Posterior intervals are 90\% equal-tailed; coverage is the realised share of intervals containing the truth. Failed fits are those that fail the convergence check of Section 4.1 after the one refit; they are included in all tables unless stated otherwise.

\textbf{Contents.} Supplementary Notes 1--6 (details of Sections 4.1, 5.1, 5.2 (i), 5.3 (iv) and 5.4), then Tables S1.1--S1.3 (triage at levels 1, 2, 3), S2.1--S2.3 (Table 6 by regime) and S2.4 (decomposition of the pooled ARL$_0$ by regime), S3.1--S3.2 (Study B by cell), S4 (Study A by factor level), S5 (pooling comparison, Fig.~4), S6 (prior sensitivity), S7 (item-effect scale $\tau_C$), S8 (indicators without information, $\beta = 0$), S9 (Farrington-type threshold by cell), S10.1--S10.2 (24-month refit schedule), S11 (horseshoe innovations, abrupt cells), S12 (fit diagnostics), S13 (dispersion that differs across groups), S14 (control charts designed under the negative binomial), S15 (the severity submodel and its loss-fraction intervals), S16 (heterogeneity that is not gamma); then Supplementary Notes 7--13 with Tables S17 (variable correspondence), S18 (industry instances), S19.1--S19.3 and S20--S22 (expected cost: means with standard errors by subset, savings and sign tests, level switching, break-even ratios and the value of monitoring by method, alarm budget and detection at like-for-like operating points, a risk-averse reading with the 90th percentile and CVaR of cost, the classification cost model), S23 (Hawkes dispersion verification) and S24 (alarm metrics of the robustness arms), Figs.~S1--S4; then Supplementary Note 14 with Tables S25.1--S25.2 (composition and items of the Korean import panel), S26.1--S26.3 (alarm metrics on the panel by target and level), S27 (the export-restriction implementations one by one) and S28 (sensitivity to the event definitions); then Supplementary Notes 15 (conformal prediction sets, Section 4.3) and 16 (window-count dispersion of a Hawkes process, Section 3.4); then Supplementary Note 17 (the rare arm of Sections 5.1, 5.3 and 5.6) with Tables S29.1 (alarm performance by cell and pooled), S29.2 (expected cost and the band of cost ratios in which each policy pays), and S29.3 and S29.4 (replication-level values).

\input{supplementary_notes}
\clearpage
"""
files = r"""
\clearpage
\input{supplementary_extra}
\input{supplementary_part2}
\clearpage
\textbf{Result files and scripts.} Study A, B and robustness results are the replication-level files \path{results/A/}, \path{results/B/} and \path{results/R/} (\path{<cell>__<variant>.jsonl}, one line per replication), summarised by \path{tools/analyze.py} into \path{analysis/A_replications.csv}, \path{B_replications.csv}, \path{R_replications.csv} and \path{A_hier_by_*.csv}; Table 6 and Tables S1--S2 come from \path{tools/table5b.py} and the fit diagnostics from \path{tools/fitstats.py}. The follow-up arms are \path{followup/a1_priors.py} (Table S6; \path{A1_priors.jsonl}), \path{followup/a7_tauC.py} (Table S7; \path{A7_tauC.jsonl}), \path{followup/a3_farrington.py} (Table S9; \path{A3_farrington_replications.csv}) and \path{followup/mac_arms.py} with its arms \texttt{a2}, \texttt{a5} and \texttt{a6} (Tables S8, S10 and S11; \path{R/R-beta0__latent.jsonl}, \path{B24/<cell>__M2.jsonl}, \path{A6/<cell>__horseshoe2.jsonl}). The remaining arms are \path{followup/b1_nbcharts.py} and \path{b1_diag.py} (Table S14; \path{B1_nbcharts_replications_group.csv}, \path{_panel.csv}, \path{B1_nbcharts_diag.csv}), \path{followup/b2_alpha_groups.py} (Table S13; \path{B2_alpha_groups.jsonl}), \path{followup/b5_severity.py} (Table S15; \path{B5/R/<cell>__latent.jsonl}) and \path{followup/b3_lognormal.py} (Table S16; \path{B3/R/R-lognormal__latent.jsonl}; the \texttt{frailty} argument of \path{dgp.simulate}). The expected-cost analysis (Section 5.6; Tables S19--S22) uses \path{tools/extract_alarm_share.py} (alarm shares and onset counts from the Study B JSONL files), \path{tools/validate_regen.py} (exact regeneration of the simulated panels from their seeds, checked against the archived metrics), \path{tools/cost_curves.py} (cost curves, savings, sign tests, break-even ratios, sensitivity; outputs under \path{results/analysis/cost/}) and \path{tools/cost_tail.py} (Table S21.4). The backtest (Section 5.5; Note 14) is run by \path{backtest/backtest_multi.py} on the panel \path{backtest/panel_410.csv}, with outputs under \path{results/backtest/}. The rare arm (Note 17) is run by \path{run_rare_full.py}; its replication-level files are \path{results_rare/B/B-a2-mu0.02-<regime>-I100__all.jsonl}, in the Study B record format, and its tables and statements are computed by the scripts of \path{rare_arm/}: \path{rare_cost.py} (expected cost, bands, savings and paired comparisons with the definitions of Section 5.6), \path{tables_full.py} and \path{analyze_full.py} (the Table 6 metrics and the replication-level detail). This document and the CSV form of every table are produced by \path{extended_results/make_supp.py} (Tables S1--S12), \path{extended_results/make_supp_extra.py} (Tables S13--S16), \path{extended_results/gen_supp_part2.py} (Notes 7--13, 15 and 16; Tables S17--S24), \path{extended_results/gen_supp_backtest.py} (Note 14, Tables S25--S28) and \path{rare_arm/gen_note17.py} (Note 17, Tables S29.1--S29.4).
\end{document}
"""
open(os.path.join(OUT, "supplementary.tex"), "w", encoding="utf8").write(preamble + intro + "\n".join(tex) + files)
print("tables:", len(tex), "csv:", csv_files)
