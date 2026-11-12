# -*- coding: utf-8 -*-
"""gen_supp_part2.py — the second part of the supplementary material: Supplementary Notes 7–13, 15 and 16 with Tables
S17–S24 and Figs. S1–S4 (the yield-model lineage, the severity submodel, the industry instances, the expected-cost
analysis of Section 5.6, the Hawkes verification, the robustness arms, the modelling choices, the conformal sets and
the Hawkes derivation) and the reference list. Writes supplementary_part2.tex (input by supplementary.tex), which in
turn inputs supplementary_backtest.tex (Note 14, gen_supp_backtest.py) and supplementary_note17_rare_arm.tex (Note 17,
rare_arm/gen_note17.py), and the CSV form of Tables S17–S24. Figures S1–S4 are copied from figures/out_final/ to
supplementary/figs/.

usage: python3 gen_supp_part2.py COST_OUT_DIR SUPP_DIR [FIG_DIR] [ANALYSIS_DIR]
  COST_OUT_DIR  results/analysis/cost (cost_table_A.csv, cost_savings_A.csv, cost_levels_A.csv, cost_breakeven_A.csv,
                cost_table_B.csv, cost_sensitivity_A.csv, cost_curves_grid_A.csv, cost_inputs_by_replication.csv,
                cost_tail_A.csv, cost_tail_bands.csv)
  SUPP_DIR      repository supplementary/ (writes supplementary_part2.tex and tables/S17..S24 csv)
"""
import os, sys, shutil
import pandas as pd
import numpy as np

COST, SUPP = sys.argv[1], sys.argv[2]
TAB = os.path.join(SUPP, "tables"); os.makedirs(TAB, exist_ok=True)
FIGS = os.path.join(SUPP, "figs"); os.makedirs(FIGS, exist_ok=True)
REPO = os.path.dirname(os.path.abspath(SUPP.rstrip("/")))
FIGSRC = sys.argv[3] if len(sys.argv) > 3 else os.path.join(REPO, "figures", "out_final")   # optional third argument: folder with the main-text figure PDFs
ANDIR = sys.argv[4] if len(sys.argv) > 4 else os.path.join(REPO, "results", "analysis")       # optional fourth argument: results/analysis (R_cells.csv, B_cells.csv)
for src, dst in (("FigS1_lineages.pdf", "FigS1_lineages.pdf"), ("FigS2_industry_map.pdf", "FigS2_industry_map.pdf"), ("FigS3_hawkes_dispersion.pdf", "FigS3_hawkes_dispersion.pdf"),
                 ("FigS4_three_layer_correspondence.png", "FigS4_correspondence.png")):
    p = os.path.join(FIGSRC, src)
    if not os.path.exists(p):
        sys.exit(f"figure not found: {p}")
    shutil.copy(p, os.path.join(FIGS, dst))

LABEL = {"M2": "M2 (proposed)", "NBcomplete": "NB complete pooling", "NBnone": "NB no pooling", "M0": "Poisson M0", "xgb": "XGBoost", "lstm": "LSTM",
         "tft": "TFT", "sigma": "Sigma-rule (plug-in SRI)", "ewma": "EWMA", "cusum": "CUSUM", "nbcusum": "CUSUM, negative-binomial design",
         "farr": "Farrington-type threshold", "never": "Never act", "always": "Always act", "trivial": "Cheaper of never act and always act, in each replication"}
ORDER = ["M2", "NBcomplete", "NBnone", "M0", "xgb", "lstm", "tft", "sigma", "ewma", "cusum", "nbcusum", "farr", "never", "always"]
R = [2, 5, 10, 20, 50]

T = pd.read_csv(os.path.join(COST, "cost_table_A.csv"))
S = pd.read_csv(os.path.join(COST, "cost_savings_A.csv"))
L = pd.read_csv(os.path.join(COST, "cost_levels_A.csv"))
BE = pd.read_csv(os.path.join(COST, "cost_breakeven_A.csv"))
TB = pd.read_csv(os.path.join(COST, "cost_table_B.csv"))
SE = pd.read_csv(os.path.join(COST, "cost_sensitivity_A.csv"))

out = []
P = out.append


def table_env(label, caption, header, rows, align, size=r"\scriptsize"):
    P(r"\noindent\begin{minipage}{\linewidth}")
    P(r"\captionof{table}{" + caption + r"}\label{" + label + "}")
    P(r"\centering" + size)
    P(r"\begin{tabular}{" + align + "}")
    P(r"\toprule")
    P(" & ".join(header) + r" \\ \midrule")
    for r in rows:
        P(" & ".join(str(c) for c in r) + r" \\")
    P(r"\bottomrule")
    P(r"\end{tabular}")
    P(r"\end{minipage}")
    P(r"\vspace{14pt}")
    P("")


def csv_out(name, header, rows):
    pd.DataFrame(rows, columns=header).to_csv(os.path.join(TAB, name), index=False)


# ================================================================ Note 7
P(r"\clearpage")
P(r"\textbf{Supplementary Note 7. The yield-model lineage and the gamma–Poisson family (Sections 1 and 3.1 of the main text).}")
P("")
P(r"\textit{The die-yield lineage.} The earliest die-yield model assumes defects that are Poisson-distributed over the wafer with density $D_{0}$, so that a die of area $A$ is defect-free with probability $\exp(-AD_{0})$. Observed yields exceeded this prediction for large dies, and the explanation---defect density varies across wafers and across regions of a wafer---led Murphy (1964) to average the Poisson yield over a triangular distribution of $D_{0}$ and Seeds (1967) and Price (1970) over an exponential distribution. Stapper (1973) and Stapper et al.\ (1983) generalised the mixture to a gamma distribution, which yields the negative-binomial formula (Y2) below with a cluster parameter $\alpha$. The negative-binomial model became standard because $\alpha$ could be estimated from wafer maps and because it nests the earlier forms (Cunningham, 1990; Stapper, 1989). Defect density itself follows a learning curve that is well approximated by a log-linear decline in cumulative volume (Wright, 1936), and yield engineers use the fitted $(D_{0},\alpha)$ to set scrap and disposition limits and to select among model variants with information criteria. Fig.~S1(a) summarises this lineage, each model drawn as the distribution of defect density over which it averages the Poisson yield.")
P("")
P(r"\textit{The source model.} For a die of area $A$ on a wafer with defect density $D_{0}$ and cluster parameter $\alpha$, the source model of Section 3.1 of the main text can be stated in four expressions,")
P(r"\begin{align}")
P(r"Y_{\mathrm{Poisson}} &= \exp(-AD_{0}), \tag{Y1}" + "\\\\")
P(r"Y_{\mathrm{NB}} &= \Big(1 + \frac{AD_{0}}{\alpha}\Big)^{-\alpha}, \tag{Y2}" + "\\\\")
P(r"\log D_{0}(v) &= a_{0} - b_{0}\log v, \tag{Y3}" + "\\\\")
P(r"\text{scrap rate} &= 1 - Y_{\mathrm{NB}}\ \text{(die level)}, \tag{Y4}")
P(r"\end{align}")
P(r"where $v$ is cumulative volume and (Y3) is the learning curve. In the source, $(D_{0},\alpha)$ are estimated from inspection data, (Y3) is fitted by least squares and scrap limits are set as control limits on yield; Sections 3.2--4.4 of the main text replace each of these steps, and Fig.~S4 draws the three-layer correspondence---detection, model, alarm---between the two readings.")
P("")
P(r"\noindent\begin{minipage}{\linewidth}\centering\includegraphics[width=\linewidth]{figs/FigS4_correspondence.png}")
P(r"\captionof{figure}{Fig. S4. Three-layer correspondence between semiconductor yield engineering (source) and supply-chain early warning (target): inline inspection and event screening (detection), negative-binomial yield and mixed-Poisson reliability (model), disposition limits and the cost-based alarm (alarm).}\label{fig:S4}\end{minipage}")
P(r"\vspace{14pt}")
P("")
P(r"\textit{The gamma–Poisson family across domains.} The negative-binomial yield model is one instance of a much older idea: a Poisson process whose intensity is itself gamma-distributed, so that heterogeneity of the intensity produces over-dispersion and the probability of zero events is the Laplace transform of the intensity distribution evaluated at the exposure. Greenwood and Yule's accident proneness (1920), the CreditRisk+ model of portfolio default counts (Credit Suisse Financial Products, 1997), the dispersion parameter of superspreading (Lloyd-Smith et al., 2005), gamma frailty in survival and reliability analysis (Vaupel et al., 1979; Hougaard, 2000) and the general theory of mixed Poisson processes (Grandell, 1997) all rest on this construction; software-reliability growth models (Goel and Okumoto, 1979) play the role of the learning curve, and count-regression practice made the negative binomial the default for over-dispersed counts (Lawless, 1987; Cameron and Trivedi, 2013). Fig.~S1(b) summarises this genealogy around the zero-probability identity at its centre, with the supply-shock case of the main text below. Table S17 gives the variable correspondence between the source (die yield) and target (supply reliability) readings that Fig.~S4 draws (Section 1 of the main text).")
P("")
P(r"\noindent\begin{minipage}{\linewidth}\centering\includegraphics[width=\linewidth]{figs/FigS1_lineages.pdf}")
P(r"\captionof{figure}{Fig. S1. (a) Die-yield model lineage from the Poisson formula to the negative binomial; (b) the gamma–Poisson family across domains.}\label{fig:S1}\end{minipage}")
P(r"\vspace{14pt}")
P("")
rows17 = [
    ["defect density (defects per unit area)", "$D_{0}$", "risk intensity (expected shocks per period for a reference item)", "$R_{it}$", "intensity"],
    ["die area", "$A$", "exposure coefficient (dimensionless multiplier of expected shocks)", "$C_{i}$", "exposure"],
    ["cluster parameter", r"$\alpha$", "dispersion parameter (within-period clustering of intensity)", r"$\alpha'_{g}$", "dispersion"],
    ["defects per die", "–", "shock events per item-period", "$N_{it}$", "count"],
    ["expected defects per die", "$AD_{0}$", "expected shocks per item-period", r"$\mu_{it} = R_{it}C_{i}\Delta$", "mean"],
    ["die yield", "$Y$", r"supply reliability index, $P(N_{it} = 0)$", r"$\mathrm{SRI}_{it}$", "reliability"],
    ["scrap rate", "–", "supply disruption rate (Supplementary Note 8)", r"$\mathrm{SDR}_{it}$", "loss"],
    ["learning curve in cumulative volume", "(Y3)", "group-level dynamic log-intensity", r"$m_{t}^{g},\delta_{t}^{g}$", "dynamics"],
]
table_env("tab:S17", "Table S17. Variable correspondence between the source (die yield) and target (supply reliability) models (Fig.~S4; Section 1 of the main text).",
          ["Source (yield)", "Symbol", "Target (supply)", "Symbol", "Role"], rows17, r"p{3.1cm}lp{5.9cm}ll", size=r"\footnotesize")
csv_out("S17_correspondence.csv", ["Source (yield)", "Symbol", "Target (supply)", "Symbol", "Role"], rows17)

# ================================================================ Note 8
P(r"\textbf{Supplementary Note 8. Severity and the supply disruption rate (Section 4.4 of the main text).}")
P("")
P(r"The SRI answers ``will there be a shock?''; the disruption rate answers ``how much supply is lost?'' Because a period without shocks has zero loss, the loss variable has a point mass at zero and the natural model is a hurdle,")
P(r"\begin{align}\mathrm{SDR}_{it} &= \mathbf{1}\{N_{it}\ge 1\}\,S_{it}, \tag{S8.1}\\ S_{it}\mid N_{it}\ge 1 &\sim \mathrm{Beta}\big(\phi\,m_{it},\ \phi\,(1-m_{it})\big), \tag{S8.2}\\ \mathrm{logit}\,m_{it} &= \gamma_{0} + \gamma_{1}\log\mu_{it} + \boldsymbol{\gamma}_{2}^{\top}\mathbf{x}_{i,t-1}\ (+\,\gamma_{3}\log N_{it}), \tag{S8.3}\\ E[\mathrm{SDR}_{it}] &= (1-\mathrm{SRI}_{it})\,m_{it},\qquad E[\mathrm{Loss}_{it}] = \mu_{it}\,E[S_{it}], \tag{S8.4}\end{align}")
P(r"where (S8.2)–(S8.3) is a beta regression (Ferrari and Cribari-Neto, 2004) for the loss fraction given a shock, with precision $\phi$, and the second equality in (S8.4) assumes losses add across shocks; the optional term in parentheses in (S8.3) lets the loss fraction depend on the number of shocks realised in the period. The two outputs of the framework---probability of disruption and loss given disruption---thus come from (4) of the main text and (S8.2) respectively. If complete stoppage ($S = 1$) can occur, a one-inflated beta or the compression of Smithson and Verkuilen (2006) is used (Ospina and Ferrari, 2010). For heavy-tailed severities the beta body is complemented by a generalised-Pareto tail fitted to exceedances of the observed loss fraction over its 90th percentile by the peaks-over-threshold method (Pickands, 1975; Coles, 2001), once at least ten exceedances are available (Section 4.4 of the main text). The priors are $\phi \sim \mathrm{Gamma}(2, 0.1)$ and $\gamma \sim N(0,1)$; the severity submodel is fitted at every monitoring origin, and its recovery of the design and the coverage of its loss-fraction intervals before and after conformalisation are reported in Table S15.")
P("")

# ================================================================ Note 9
P(r"\textbf{Supplementary Note 9. Industry instances of the intensity–exposure–dispersion structure (Section 6.2 of the main text).}")
P("")
P(r"The intensity–exposure–dispersion structure of Proposition 1 is not specific to critical materials. Table S18 instantiates $(\lambda, s, \alpha) \equiv (R, C, \alpha')$ for five industries at two levels: the process level, where the source model already operates in a form adapted to each industry, and the supply level, where the framework applies to the industry's own critical inputs. Fig.~S2 shows the dual application. The detection layer differs in every row; the model and alarm layers do not.")
P("")
rows18 = [
    ["lithium-ion batteries", "electrode vision, X-ray CT, field battery-management data", "electrode defect density", "cell area / capacity", "coating-roll and lot correlation", "cell yield / lot scrap", "precursors, separators"],
    ["displays", "pixel inspection images (mura)", "pixel defect density", "panel area", "spatial mura clustering", "panel yield / substrate breakage", "emitter materials"],
    ["biopharmaceuticals", "bioreactor sensors, GMP deviation records", "contamination / deviation intensity", "batch scale and value", "raw-material lot and facility correlation", "batch purity / batch rejection", "media, chromatography resins"],
    ["AI data centres", "GPU telemetry (ECC errors, thermal)", "node failure intensity", "cluster and job size", "rack- and power-level common cause", "job completion / node uptime", "high-bandwidth memory, accelerators"],
    ["space systems", "component qualification tests, on-orbit telemetry", "component failure intensity", "mission duration, orbital environment", "common-cause failures", "mission success / vehicle loss", "space-grade components"],
]
P(r"\begin{landscape}")
table_env("tab:S18", "Table S18. Industry instances of the intensity–exposure–dispersion structure.",
          ["Industry", "Detection source", r"Intensity $\lambda$ ($= R$)", "Exposure $s$ ($= C$)", r"Clustering $\alpha$ ($= \alpha'$)", "Process-level output", "Supply-level instance"],
          rows18, r"p{2.4cm}p{4.2cm}p{3.0cm}p{3.0cm}p{3.4cm}p{3.2cm}p{3.2cm}", size=r"\scriptsize")
P(r"\noindent\begin{minipage}{\linewidth}\centering\includegraphics[width=0.9\linewidth]{figs/FigS2_industry_map.pdf}")
P(r"\captionof{figure}{Fig. S2. Industry instance map: the intensity–exposure–dispersion structure at process level and at supply level.}\label{fig:S2}\end{minipage}")
P(r"\end{landscape}")
csv_out("S18_industry_instances.csv", ["Industry", "Detection source", "Intensity", "Exposure", "Clustering", "Process-level output", "Supply-level instance"], rows18)

# ================================================================ Note 10
P(r"\textbf{Supplementary Note 10. Expected cost of the alarm policies (Section 5.6 of the main text).}")
P("")
P(r"\textit{Inputs.} All quantities come from the archived replication-level results of Study B (\path{results/B/<cell>__all.jsonl}, 65 replications in 12 cells) and the follow-up baselines (\path{A3_farrington_replications.csv}, \path{B1_nbcharts_replications_group.csv}): for method $m$ at level $k$ in replication $j$, the one-month detection rate $\mathrm{DR}_{mkj}$ (\texttt{dr\_h1}), the share of decision item-months with the level in force (\texttt{alarm\_share}; alarm months per item-year $\mathrm{AM} = 12\times$ share), the number of alarm onsets (\texttt{n\_onsets}) and the false onsets per item-year over the whole window (\texttt{fa\_per\_itemyear}) and in the stable windows (\texttt{fa\_per\_itemyear\_stable}, the Table 6 quantity). The number of realised shock item-months in the detection window ($\tau = 37,\dots,118$) is counted exactly by regenerating each simulated panel from its archived seed (\path{dgp.simulate}, \path{dgp.seed_of('B', cell, rep)}); the regeneration reproduces the archived base rates and the EWMA/CUSUM metrics of all 65 replications to four decimals (\path{code/tools/validate_regen.py}).")
P("")
P(r"\textit{Model A (main text).} $\mathrm{EC}_{mkj}(r) = \mathrm{AM}_{mkj} + r\,(1-\mathrm{DR}_{mkj})\,E_{j}$, with $E_{j}$ the shock item-months per item-year; never acting costs $rE_{j}$, always acting 12. For methods with several operating points the reported value at each $r$ is the cheapest of their levels (the lower envelope of the level-wise mean-cost lines); charts and the Farrington-type threshold have one line. Model A charges every alarm month, including those preceding a detected shock. The alternative that charges only alarm months not followed by a shock cannot be computed from the archived outputs, which hold alarm months and detection rates but not the alarm paths.")
P("")
P(r"\textit{Model B (the classification framing of eq.\ (19) and of Table 6).} $\mathrm{EC}^{B}_{mkj}(r) = \mathrm{FA}_{mkj} + r\,(1-\mathrm{DR}_{mkj})\,E_{j}$ with $\mathrm{FA}$ the false onsets per item-year over the whole window. An onset is counted once however long the alarm stays on, so the always-on policy has almost no onsets and a cost of zero, and every probability-based method is cheapest at its lowest threshold for every $r$ (Table S22). Model B is reported to tie the cost analysis to the false-alarm definition of Table 6; the conclusions of the main text rest on Model A.")
P("")
P(r"\textit{Statistics.} Means over the 65 replications (the reference cell has 10) with Monte Carlo standard errors; paired differences M2 $-$ $m$ at each method's cheapest level with two-sided sign tests; savings $1 - \mathrm{EC}_{\mathrm{M2}}/\mathrm{EC}_{m}$ on the means; break-even ratios $r^{*}$ where the envelope curves cross (log grid of 600 points on $[1, 100]$); the value of monitoring $1 - \mathrm{EC}_{\mathrm{M2}} / \min(rE, 12)$. Scripts: \path{code/tools/extract_alarm_share.py}, \path{code/tools/cost_curves.py}; outputs under \path{results/analysis/cost/}.")
P("")
# Table S19: EC with SE by subset
header19 = ["Method"] + [f"$r = {r}$" for r in R]
for sub, tag in (("All cells", "all twelve cells (n = 65)"), ("mu 0.05", r"cells with $\bar{\mu}$ = 0.05 (n = 30)"), ("mu 0.2", r"cells with $\bar{\mu}$ = 0.2 (n = 35)")):
    rows = []
    t = T[T.subset == sub]
    for m in ORDER:
        tt = t[t.method == m].set_index("r")
        cells = []
        for r in R:
            lv = tt.loc[r, "level"]
            cells.append(f"{tt.loc[r, 'EC']:.2f} ({tt.loc[r, 'SE']:.2f})" + (f" [L{int(lv)}]" if pd.notna(lv) else ""))
        rows.append([LABEL[m]] + cells)
    sfx = {"All cells": ".1", "mu 0.05": ".2", "mu 0.2": ".3"}[sub]
    table_env(f"tab:S19{sfx}", f"Table S19{sfx}. Expected cost per item-year (Model A; units of one alarm-month), {tag}: mean over replications (Monte Carlo standard error) at the method's cheapest level [L$k$] for each avoidable loss per shock $r$. Never act = $rE$; always act = 12.",
              header19, rows, "l" + "r" * len(R), size=r"\scriptsize")
    csv_out(f"S19{sfx.replace('.', '_')}_cost_modelA_{sub.replace(' ', '')}.csv", header19, rows)
# Table S20: paired savings (All cells)
rows = []
for m in ORDER[1:] + ["trivial"]:
    t = S[(S.subset == "All cells") & (S.method == m)].set_index("r")
    cells = []
    for r in (2, 5, 10, 20, 50):
        p = t.loc[r, "sign_p"]
        cells.append(f"{100 * t.loc[r, 'saving']:+.1f}\\% ({100 * t.loc[r, 'M2_cheaper_share']:.0f}\\%" + (("; $<$0.001)" if p < 0.001 else f"; {p:.2f})") if pd.notna(p) else ")"))
    rows.append([LABEL[m]] + cells)
P(r"\begin{landscape}")
table_env("tab:S20", "Table S20. Saving of M2 relative to each policy, all twelve cells (Model A): $1 - \\mathrm{EC}_{\\mathrm{M2}}/\\mathrm{EC}_{m}$ on the means (share of the 65 replications in which M2 is cheaper; two-sided sign-test p-value, ties excluded), each method at its cheapest level. The last row takes the cheaper of never acting and always acting in each replication, so its benchmark is the mean of $\\min(rE_j, 12)$. This benchmark never costs more than $\\min(rE, 12)$ at the mean shock frequency of the subset, which Table S21.1 and the main text use for the value of monitoring. The savings in the last row are therefore never larger than the value of monitoring.",
          ["Compared with"] + [f"$r = {r}$" for r in R], rows, "l" + "l" * len(R), size=r"\scriptsize")
csv_out("S20_savings_vs_M2.csv", ["Compared with"] + [f"r={r}" for r in R], rows)
P(r"\end{landscape}")
# Table S21: break-even, switching, value of monitoring
rows = []
for sub, name in (("All cells", "All cells"), ("mu 0.05", r"$\bar{\mu}$ = 0.05"), ("mu 0.2", r"$\bar{\mu}$ = 0.2"), ("stable", "Stable"), ("gradual", "Gradual"), ("abrupt", "Abrupt"), ("alpha 0.5", r"$\alpha'$ = 0.5"), ("alpha 2", r"$\alpha'$ = 2")):
    l = L[L.subset == sub].iloc[0]
    def rstar(m):
        b = BE[(BE.subset == sub) & (BE.method == m)].iloc[0]
        return b.r_star.replace(";", ", ")
    sw = l.switches.split(";")
    s32 = sw[0].split(":")[0] if len(sw) > 0 and sw[0] else "–"; s21 = sw[1].split(":")[0] if len(sw) > 1 else "–"
    rows.append([name, int(l.n), s32, s21, f"{100 * l.max_value_of_monitoring:.0f}\\% at $r$ = {l.r_at_max:.1f}", l.r_range_vom_above_10pct.replace("-", "–"),
                 rstar("never"), rstar("always"), rstar("xgb"), rstar("lstm"), rstar("cusum")])
table_env("tab:S21_1", "Table S21.1. Level switching points of M2 (the $r$ at which level 2 becomes cheaper than level 3, and level 1 cheaper than level 2), the value of monitoring $1 - \\mathrm{EC}_{\\mathrm{M2}}/\\min(rE, 12)$ (its maximum and the range of $r$ over which it exceeds 10\\%), and break-even ratios $r^{*}$ at which the M2 envelope crosses the never-act line, the always-act line and the envelopes of XGBoost, the LSTM and the CUSUM (Model A; `none' = no crossing on $[1, 100]$).",
          ["Subset", "n", r"L3$\to$L2", r"L2$\to$L1", "Max value", "Value $>$ 10\\%", r"$r^{*}$ never", r"$r^{*}$ always", r"$r^{*}$ XGBoost", r"$r^{*}$ LSTM", r"$r^{*}$ CUSUM"],
          rows, "lrrrllrrrrr", size=r"\scriptsize")
csv_out("S21_1_levels_breakeven.csv", ["Subset", "n", "L3->L2", "L2->L1", "Max value of monitoring", "r range >10%", "r* never", "r* always", "r* xgb", "r* lstm", "r* cusum"], rows)

# Table S21.2: value of monitoring by method and subset (max over r of 1 - EC_m / min(rE, 12)), from the curve grid
G = pd.read_csv(os.path.join(COST, "cost_curves_grid_A.csv")); rg = G["r"].values
rows = []
for m in ORDER[:-2]:
    cells = []
    for sub in ("All cells", "mu 0.05", "mu 0.2"):
        triv = np.minimum(G[f"{sub}|never"].values, G[f"{sub}|always"].values)
        v = 1 - G[f"{sub}|{m}"].values / triv; i_ = int(v.argmax())
        cells.append(f"{100 * v[i_]:.1f}\\% at $r$ = {rg[i_]:.1f}")
    rows.append([LABEL[m]] + cells)
table_env("tab:S21_2", "Table S21.2. Value of monitoring by method: the largest saving of each policy's lower envelope relative to the better of never acting and always acting, $\\max_r\\,[1 - \\mathrm{EC}_m(r)/\\min(rE, 12)]$, and the $r$ at which it occurs (Model A; all cells and the two base-rate subsets). The maximum lies at $r = 12/E$ for every policy because the trivial envelope peaks there.",
          ["Method", "All cells", r"$\bar{\mu}$ = 0.05", r"$\bar{\mu}$ = 0.2"], rows, "llll", size=r"\footnotesize")
csv_out("S21_2_value_of_monitoring_by_method.csv", ["Method", "All cells", "mu 0.05", "mu 0.2"], rows)
# Table S21.3: alarm budget and one-month detection by method and level (replication means), with the ratio to a random alarm of equal budget
X = pd.read_csv(os.path.join(COST, "cost_inputs_by_replication.csv"))
KEYS = [("M2_tri1", "M2, level 1"), ("M2_tri2", "M2, level 2"), ("M2_tri3", "M2, level 3"), ("NBcomplete_tri3", "NB complete pooling, level 3"), ("NBnone_tri3", "NB no pooling, level 3"),
        ("M0_tri3", "Poisson M0, level 3"), ("xgb_tri3", "XGBoost, level 3"), ("lstm_tri3", "LSTM, level 3"), ("tft_tri3", "TFT, level 3"),
        ("sigma_2", "Sigma-rule, limit 2"), ("sigma_3", "Sigma-rule, limit 3"), ("ewma", "EWMA"), ("cusum", "CUSUM"), ("nbcusum", "CUSUM, negative-binomial design"), ("farr", "Farrington-type threshold")]
rows = []
for k, lab in KEYS:
    am = X[f"{k}_am"].values; dr = X[f"{k}_dr1"].values; on = am.mean() / 12
    rows.append([lab, f"{am.mean():.2f}", f"{100 * on:.0f}\\%", f"{100 * dr.mean():.0f}\\%", f"{dr.mean() / on:.2f}"])
table_env("tab:S21_3", "Table S21.3. Alarm budget and detection at a like-for-like operating point (Study B, all cells, means over the 65 replications): alarm months per item-year, the share of decision item-months in which the level is in force, the one-month detection rate (share of shock item-months with the level in force in the preceding month) and the ratio of that rate to the share in force, which is the detection rate relative to an alarm switched on at random for the same share of months (1 = no skill).",
          ["Policy and level", "Alarm months / item-year", "Share in force", r"Detected $\le$ 1 month", "Ratio to random"], rows, "lrrrr", size=r"\footnotesize")
csv_out("S21_3_alarm_budget_detection.csv", ["Policy and level", "Alarm months per item-year", "Share in force", "Detected <= 1 month", "Ratio to random"], rows)
# Table S21.4: a risk-averse reading — 90th percentile and CVaR_0.90 of the per-replication cost, tail envelope over levels
TL = pd.read_csv(os.path.join(COST, "cost_tail_A.csv")); TBAND = pd.read_csv(os.path.join(COST, "cost_tail_bands.csv"))
rows = []
t = TL[TL.subset == "All cells"].set_index("method")
for m in ORDER:
    cells = []
    for r in R:
        q = t.loc[m, f"q90_r{r}"]; c = t.loc[m, f"cvar90_r{r}"]
        lq = t.loc[m, f"q90_level_r{r}"] if f"q90_level_r{r}" in t.columns else np.nan
        lc = t.loc[m, f"cvar90_level_r{r}"] if f"cvar90_level_r{r}" in t.columns else np.nan
        tag = (f" [L{int(lq)}/L{int(lc)}]" if pd.notna(lq) and pd.notna(lc) else "")
        cells.append(f"{q:.2f} / {c:.2f}{tag}")
    rows.append([LABEL[m]] + cells)
b = TBAND[TBAND.subset == "All cells"].set_index("measure")
note = (f"Under the mean, M2 beats both trivial policies for $r$ in [{b.loc['mean','band_lo']:.1f}, {b.loc['mean','band_hi']:.1f}] (largest saving "
        f"{100*b.loc['mean','max_saving']:.1f}\\% at $r$ = {b.loc['mean','r_at_max']:.1f}); under the 90th percentile for $r$ in [{b.loc['q90','band_lo']:.1f}, "
        f"{b.loc['q90','band_hi']:.1f}] ({100*b.loc['q90','max_saving']:.1f}\\% at {b.loc['q90','r_at_max']:.1f}); under CVaR$_{{0.90}}$ for $r$ in "
        f"[{b.loc['cvar90','band_lo']:.1f}, {b.loc['cvar90','band_hi']:.1f}] ({100*b.loc['cvar90','max_saving']:.1f}\\% at {b.loc['cvar90','r_at_max']:.1f}).")
table_env("tab:S21_4", "Table S21.4. A risk-averse reading of the cost curves (Model A, all cells, 65 replications): the 90th percentile and the conditional mean beyond it (CVaR$_{0.90}$) of the per-replication cost $\\mathrm{EC}_j(r)$, each at the level that minimises that measure [L$k$ for the percentile / for the CVaR]; never acting is the same statistic of $rE_j$, always acting is 12 in every replication. " + note,
          ["Method"] + [f"$r = {r}$" for r in R], rows, "l" + "l" * len(R), size=r"\scriptsize")
csv_out("S21_4_tail_risk.csv", ["Method"] + [f"q90 / CVaR90 at r={r}" for r in R], rows)
shutil.copy(os.path.join(COST, "cost_tail_bands.csv"), os.path.join(TAB, "S21_4_tail_bands.csv"))   # an exact copy, no float round trip
# Table S22: Model B
rows = []
t = TB[TB.subset == "All cells"]
for m in ORDER:
    tt = t[t.method == m].set_index("r")
    cells = []
    for r in R:
        lv = tt.loc[r, "level"]
        cells.append(f"{tt.loc[r, 'EC']:.2f}" + (f" [L{int(lv)}]" if pd.notna(lv) else ""))
    rows.append([LABEL[m]] + cells)
table_env("tab:S22", "Table S22. Expected cost per item-year under Model B (false onsets per item-year over the whole window plus $r$ times the missed shocks per item-year), all twelve cells, each method at its cheapest level [L$k$]. Always acting has no false onsets and no misses, hence a cost of zero; every probability-based method is cheapest at level 1 for every $r$.",
          header19, rows, "l" + "r" * len(R), size=r"\scriptsize")
csv_out("S22_cost_modelB.csv", header19, rows)
# sensitivity sentence
se = SE[SE.subset == "All cells"]
ALARM = [m for m in ORDER if m not in ("never", "always")]
sg = lambda v: f"{v:+.1f}".replace("-", "−").replace("+0.0", "0.0")
chg = []
for r in (5, 20):
    tt = se[se.r == r].set_index("method")
    d = {m: 100 * (tt.loc[m, "EC_dr3"] / tt.loc[m, "EC_dr1"] - 1) for m in ALARM}
    nv = 100 * (tt.loc["never", "EC_dr3"] / tt.loc["never", "EC_dr1"] - 1)
    chg.append((r, min(d.values()), max(d.values()), nv))
X = pd.read_csv(os.path.join(COST, "cost_inputs_by_replication.csv"))
hd = {c: X[c].mean() for c in ("M2_tri1_am", "M2_tri2_am", "M2_tri1_dr1", "M2_tri2_dr1", "head1_am", "head2_am", "head1_dr1", "head2_dr1")}
h2 = {r: 100 * (se[(se.r == r) & (se.method == "head2")].EC_dr1.iloc[0] / T[(T.subset == "All cells") & (T.method == "M2") & (T.r == r)].EC_level2.iloc[0] - 1) for r in R}
h1 = {r: 100 * (se[(se.r == r) & (se.method == "head1")].EC_dr1.iloc[0] / T[(T.subset == "All cells") & (T.method == "M2") & (T.r == r)].EC_level1.iloc[0] - 1) for r in R}
P(r"\textit{Sensitivity.} Counting a shock as detected when the level was in force at any time in the three months before it (\texttt{dr\_h3}, with $E$ counted over $\tau = 39,\dots,118$) lowers the Model A cost of every alarm policy, by "
  + f"{-chg[0][2]:.0f}\\% to {-chg[0][1]:.0f}\\% at $r = 5$ and by {-chg[1][2]:.0f}\\% to {-chg[1][1]:.0f}\\% at $r = 20$ (the largest changes are those of the sigma-rule and the EWMA; never acting rises by {chg[0][3]:.1f}\\% because $E$ is counted over a shorter window), and leaves every ordering of the main text in place. "
  + f"The two-channel headline level of M2 (triage level or persistent $q$-alarm, whichever is higher) adds {hd['head2_am'] - hd['M2_tri2_am']:.2f} alarm months per item-year to level 2 and {hd['head1_am'] - hd['M2_tri1_am']:.2f} to level 1 and raises the one-month detection rate of level 2 from {hd['M2_tri2_dr1']:.3f} to {hd['head2_dr1']:.3f}; its Model A cost is within {max(abs(v) for v in h1.values()):.1f}\\% of the triage channel at level 1 and within {max(abs(v) for v in h2.values()):.0f}\\% at level 2 ({sg(h2[2])}\\%, {sg(h2[5])}\\%, {sg(h2[10])}\\%, {sg(h2[20])}\\% and {sg(h2[50])}\\% at $r$ = 2, 5, 10, 20 and 50). "
  + r"Weighting the loss by the mean loss fraction (0.17 under the log-normal severity of Table 4, 0.12 under the Pareto severity; Table S15) rescales $r$ and leaves the curves otherwise unchanged.")
P("")

# ================================================================ Note 11
P(r"\textbf{Supplementary Note 11. Numerical verification of the Hawkes window-count dispersion (Section 3.4 of the main text and Supplementary Note 16).}")
P("")
P(r"Table S23 verifies (10)–(11) of the main text by exact cluster simulation (Hawkes and Oakes, 1974) of an exponential-kernel Hawkes process with $\Lambda\Delta = 0.5$; 20 replicates of 3,000 windows per cell after burn-in. Fig.~S3 plots the index of dispersion (10) against $\kappa\Delta$ for four branching ratios and the moment-matched dispersion per unit mean (11).")
P("")
rows23 = [["0.3", "0.3", "0.491", "1.132", "1.142", "3.73", "3.53"], ["0.3", "1.0", "0.504", "1.383", "1.383", "1.32", "1.31"], ["0.3", "4.0", "0.509", "1.798", "1.785", "0.64", "0.64"],
          ["0.6", "0.3", "0.500", "1.719", "1.714", "0.70", "0.70"], ["0.6", "1.0", "0.508", "2.953", "2.931", "0.26", "0.26"], ["0.6", "4.0", "0.499", "4.857", "4.962", "0.13", "0.13"]]
table_env("tab:S23", "Table S23. Numerical verification of (10)–(11) of the main text by exact cluster simulation of an exponential-kernel Hawkes process with $\\Lambda\\Delta = 0.5$; 20 replicates of 3,000 windows per cell after burn-in.",
          ["$n$", r"$\kappa\Delta$", "mean count", "$D$ (empirical)", "$D$ (10)", r"$\alpha'_{match}$ (empirical)", r"$\alpha'_{match}$ (11)"], rows23, "lllllll", size=r"\footnotesize")
csv_out("S23_hawkes_verification.csv", ["n", "kappa*Delta", "mean count", "D empirical", "D eq10", "alpha_match empirical", "alpha_match eq11"], rows23)
P(r"\noindent\begin{minipage}{\linewidth}\centering\includegraphics[width=\linewidth]{figs/FigS3_hawkes_dispersion.pdf}")
P(r"\captionof{figure}{Fig. S3. (a) Index of dispersion (10) of Hawkes window counts against $\kappa\Delta$ for four branching ratios, large-window asymptotes dotted; (b) moment-matched dispersion per unit mean, (11).}\label{fig:S3}\end{minipage}")
P(r"\vspace{14pt}")
P("")
# ================================================================ Note 12
P(r"\textbf{Supplementary Note 12. Alarm metrics of the robustness arms (Section 5.4 of the main text).}")
P("")
P(r"Table S24 tabulates, for the robustness arms of Section 5.4 that keep the gamma–Poisson truth (the Hawkes arms are in Section 5.4 and Note 11), the M2 alarm metrics that the main text quotes: AUC-PR, Brier score and expected calibration error of the three-month event probability, the one-month detection rates at triage levels 2 and 3, the stable-window false alarms per item-year at level 2, the sequential coverage of the conformal sets, the delay of the $q$ channel after the change, the posterior median of $\alpha'$ and the RMSE of $\log C_i$. Every arm is the reference design ($\alpha' = 2$, $\bar{\mu} = 0.2$, gradual, $I = 30$, $T = 120$) with one factor changed, five replications; the first row is the Study B reference cell (ten replications). Source: \path{results/analysis/R_cells.csv} and \path{B_cells.csv}.")
P("")
RC = pd.read_csv(os.path.join(ANDIR, "R_cells.csv")); BC = pd.read_csv(os.path.join(ANDIR, "B_cells.csv"))
ARMS = [("R-measerr0.5", "latent", "measurement error in $\\mathbf{z}$ ($\\varsigma$ = 0.5)"), ("R-miss15", "latent", "15\\% missing indicators, latent AR(1)"),
        ("R-miss15", "locf", "15\\% missing indicators, last value carried forward"), ("R-pareto", "latent", "Pareto severity"),
        ("R-cold20", "latent", "20\\% cold-start items"), ("R-lag2", "latent", "indicator lag 2 in the truth, 1 in the model")]
COLS = ["aucpr", "brier", "ece", "tri2_dr1", "tri3_dr1", "tri2_fa_stable", "conf_cover", "q_arl1", "alpha_q50_med", "logC_rmse"]
FMT = ["%.3f", "%.3f", "%.3f", "%.2f", "%.2f", "%.2f", "%.3f", "%.1f", "%.2f", "%.2f"]
rows24 = []
ref = BC[BC.cell == "B-a2-mu0.2-gradual"].iloc[0]
rows24.append(["reference cell (Study B)", int(ref.n)] + [(f % ref["M2_" + c]) if ("M2_" + c) in ref.index else "–" for f, c in zip(FMT, COLS)])
for cell, arm, lab in ARMS:
    q = RC[(RC.cell == cell) & (RC.arm == arm)].iloc[0]
    rows24.append([lab, int(q.n)] + [f % q[c] for f, c in zip(FMT, COLS)])
hdr24 = ["Arm", "n", "AUC-PR", "Brier", "ECE", "DR $\\le$ 1 m, L2", "DR $\\le$ 1 m, L3", "FA (stable), L2", "Set cov.", "$q$ ARL$_1$", "$\\alpha'$ median", "RMSE $\\log C$"]
table_env("tab:S24", "Table S24. Alarm metrics of M2 in the robustness arms of Section 5.4 (five replications each; the reference cell of Study B, ten replications, for comparison): AUC-PR, Brier score and ECE of the three-month event probability; one-month detection rate at triage levels 2 and 3; stable-window false alarms per item-year at level 2; sequential coverage of the conformal sets (target 0.90); mean delay of the $q$ channel after the change (months); posterior median of $\\alpha'$ (truth 2) and RMSE of $\\log C_i$.",
          hdr24, rows24, r">{\raggedright\arraybackslash}p{2.9cm}r" + "r" * len(COLS), size=r"\scriptsize")      # the first column wraps: the table fits the text width
csv_out("S24_robustness_alarm_metrics.csv", ["Arm", "n"] + COLS, rows24)

# ================================================================ Note 13
P(r"\textbf{Supplementary Note 13. Modelling choices and the details of the parameter-recovery results (Sections 3.3 and 5.2 of the main text).}")
P("")
P(r"\textit{Three choices in the intensity dynamics (7)–(9).} First, there is no additional item-period noise term in (7): a log-normal disturbance on $\log R$ and the gamma multiplier in (2) both represent unexplained intensity variation and are not separately identified at small $\mu$ (Table S16 measures the price of fitting the gamma form to a log-normal truth). Second, the level and slope are indexed by group, not item: the item-specific level and the exposure $u_i$ in (14) are identified only through their sum, and item-level slopes cannot be estimated from rare counts (Section 3.2); static item differences therefore live in $C_i$ and the trend in the group, which also lets a cold-start item inherit its group's trajectory. Third, $\sigma_\eta = \tau_\zeta = 0$ reduces (8)–(9) to the deterministic log-linear trend (Y3), so the dynamic model nests the source model (Section 4.1).")
P("")
P(r"\textit{Reference cell: two decisions fed back into the design.} In the single replication of the reference cell reported in Section 5.2, the deterioration probability $P(\delta_t^g > 0.01 \mid \mathcal{D})$ needs a floor: without it the probability floats around one half, with it it averages 0.23 before $t_0$ and 0.76 after $t_0 + 12$. Even the smoothed posterior detects a $+0.02$ per month slope change only seven to eight months after onset, so the slope is a secondary flag and the primary statistics are the level-based $p$ and $q$; and sparse (horseshoe) innovations buy nothing in a gradual regime, so the Gaussian model is the base.")
P("")
P(r"\textit{(i) Pooling.} Complete pooling attributes the residual item heterogeneity to dispersion: coverage of $\log C_i$ falls to 0.13–0.49, and as the information grows its $\alpha'$ intervals contract around a value that is too small—at $\bar{\mu} = 0.5$ and $I = 30$ they cover the truth in 20\% of cases (Fig.~4 of the main text; Table S5). The bias is predictable: an omitted item effect $u_i \sim N(0, \tau_C^2)$ multiplies the gamma variable in (2), and the pseudo-true dispersion of the misspecified model is 1.61 at $\tau_C = 0.3$ and 1.00 at $\tau_C = 0.6$ for the item means of the reference cell (Supplementary Note 3; Table S7). No pooling keeps nominal coverage but learns little about the item-level $\alpha'_i$, whose log-scale interval width stays between 1.7 and 3.2.")
P("")
P(r"\textit{(ii) Dispersion information.} The width of the 90\% interval for $\log\alpha'$ falls by a factor of 2.4–4.9 from $\bar{\mu} = 0.05$ to $\bar{\mu} = 0.5$ at $I = 30$ and by 1.6–3.9 at $I = 10$ (Fig.~4b)—more slowly than the $8/\mu^2$ scaling of Section 3.2 implies, because at small $\bar{\mu}$ the hierarchical prior supplies the information that the counts lack; doubling $T$ narrows it by a further 14–42\%. When the dispersion differs across groups—$\alpha'_g$ = (1, 2, 4) and (0.5, 2, 8)—the hierarchy orders the groups correctly in every replication and covers each group's truth in 93\% and 67\% of cases, whereas complete pooling settles at 1.62 and 1.09 and covers them in 20\% and 0\% (Table S13).")
P("")
P(r"\textit{(iii) Abrupt changes.} With strong data ($I = 30$, $\bar{\mu} \geq 0.2$) the horseshoe variant locates the change sharply—posterior probability 0.65–0.90 on the true month, against 0.06–0.10 with Gaussian innovations—but its fits fail a convergence check in 18 of these 20 replications because the posterior is multimodal in the month of the jump; with weak data it spreads the change over neighbouring months. With either innovation model the 90\% interval for the change date contains the true month in every replication (Table S11).")
P("")
P(r"\input{supplementary_backtest}")
P("")
P(r"\textbf{Supplementary Note 15. Conformal prediction sets (Sections 4.3 and 5.3 (iv) of the main text).}")
P("")
P(r"For the observable count $N_{i,t+h}$ the framework forms distribution-free prediction sets from the posterior predictive mass function by the adaptive-prediction-sets construction of Romano et al.\ (2020): the candidate counts are ordered by predictive mass, the set is the smallest initial segment whose cumulative mass reaches the calibrated level, and the level is calibrated on the conformity scores of past item-periods. Because exchangeability fails under regime change, the per-item miscoverage level is adjusted online by adaptive conformal inference (Gibbs and Candès, 2021), which raises the level after a miss and lowers it after a cover, so that the realised sequential coverage tracks the 90\% target; most sets are $\{0\}$ or $\{0,1\}$, and the set size is itself informative. Conformalised quantile regression (Romano et al., 2019) does the same for the loss fraction given a shock (Supplementary Note 8, Table S15). In Study B the sets cover 91.8\% of the counts under M2 with 2.03 counts on average and 90.6\% under the Poisson model with 1.68 (Table 6b of the main text); the gain from modelling over-dispersion lies in the calibration of the event probabilities and in run-length control, not in the prediction sets.")
P("")
P(r"\textbf{Supplementary Note 16. Window-count dispersion of an exponential-kernel Hawkes process (Section 3.4 of the main text).}")
P("")
P(r"Let $\lambda(t) = \nu + \sum_{t_{k} < t} a\, e^{-b(t - t_{k})}$ with $n = a/b < 1$, stationary rate $\Lambda = \nu/(1 - n)$ and $\kappa = b - a = b(1 - n)$. Hawkes (1971) gives the complete covariance density of the stationary process as $\Lambda\,\delta(\tau) + c(\tau)$ with")
P(r"\begin{equation}")
P(r"c(\tau) = \frac{\Lambda\, a\,(2b - a)}{2\kappa}\, e^{-\kappa|\tau|}, \qquad \tau \neq 0. \tag{S16.1}")
P(r"\end{equation}")
P(r"The variance of the count in a window of length $\Delta$ is the double integral of the covariance density over the window,")
P(r"\begin{equation}")
P(r"\mathrm{Var}[N(\Delta)] = \Lambda\Delta + 2\int_{0}^{\Delta} (\Delta - \tau)\, c(\tau)\, d\tau. \tag{S16.2}")
P(r"\end{equation}")
P(r"Using $\int_{0}^{\Delta} (\Delta - \tau)\, e^{-\kappa\tau}\, d\tau = \Delta/\kappa - (1 - e^{-\kappa\Delta})/\kappa^{2}$,")
P(r"\begin{equation}")
P(r"\mathrm{Var}[N(\Delta)] = \Lambda\Delta\left[1 + \frac{a(2b - a)}{\kappa^{2}}\left(1 - \frac{1 - e^{-\kappa\Delta}}{\kappa\Delta}\right)\right], \tag{S16.3}")
P(r"\end{equation}")
P(r"and substituting $a = nb$, $\kappa = b(1 - n)$ gives $a(2b - a)/\kappa^{2} = n(2 - n)/(1 - n)^{2}$, which is (10) of the main text. The large-window limit $1/(1 - n)^{2}$ follows from $1 + n(2 - n)/(1 - n)^{2} = 1/(1 - n)^{2}$, and the small-window expansion from $g(y) = y/2 - y^{2}/6 + O(y^{3})$. The expression agrees with the variance formula of Da Fonseca and Zaatour (2014); Supplementary Note 11 verifies (10)--(11) by exact cluster simulation (Table S23, Fig.~S3).")
P("")
P(r"\input{supplementary_note17_rare_arm}")
P("")
P(r"\vspace{10pt}")
P("")
P(r"\textbf{References cited in Supplementary Notes 7–11, 15 and 16.} Cameron, A.C., Trivedi, P.K., 2013. Regression Analysis of Count Data, 2nd ed. Cambridge University Press, Cambridge. Coles, S., 2001. An Introduction to Statistical Modeling of Extreme Values. Springer, London. Credit Suisse Financial Products, 1997. CreditRisk+: A Credit Risk Management Framework. London. Cunningham, J.A., 1990. IEEE Trans. Semicond. Manuf. 3, 60–71. Da Fonseca, J., Zaatour, R., 2014. J. Futures Mark. 34, 548–579. Ferrari, S., Cribari-Neto, F., 2004. J. Appl. Stat. 31, 799–815. Gibbs, I., Candès, E., 2021. Adv. Neural Inf. Process. Syst. 34, 1660–1672. Goel, A.L., Okumoto, K., 1979. IEEE Trans. Reliab. R-28, 206–211. Grandell, J., 1997. Mixed Poisson Processes. Chapman \& Hall, London. Greenwood, M., Yule, G.U., 1920. J. R. Stat. Soc. 83, 255–279. Hawkes, A.G., 1971. Biometrika 58, 83–90. Hawkes, A.G., Oakes, D., 1974. J. Appl. Probab. 11, 493–503. Hougaard, P., 2000. Analysis of Multivariate Survival Data. Springer, New York. Lawless, J.F., 1987. Can. J. Stat. 15, 209–225. Lloyd-Smith, J.O., et al., 2005. Nature 438, 355–359. Murphy, B.T., 1964. Proc. IEEE 52, 1537–1545. Ospina, R., Ferrari, S.L.P., 2010. Stat. Pap. 51, 111–126. Pickands, J., 1975. Ann. Stat. 3, 119–131. Price, J.E., 1970. Proc. IEEE 58, 1290–1291. Romano, Y., Patterson, E., Candès, E., 2019. Adv. Neural Inf. Process. Syst. 32, 3538–3548. Romano, Y., Sesia, M., Candès, E., 2020. Adv. Neural Inf. Process. Syst. 33, 3581–3591. Seeds, R.B., 1967. IEEE Int. Conv. Rec. 6, 60–61. Smithson, M., Verkuilen, J., 2006. Psychol. Methods 11, 54–71. Stapper, C.H., 1973. IEEE Trans. Electron Devices 20, 655–657. Stapper, C.H., 1989. Microelectron. J. 20, 129–151. Stapper, C.H., Armstrong, F.M., Saji, K., 1983. Proc. IEEE 71, 453–470. Vaupel, J.W., Manton, K.G., Stallard, E., 1979. Demography 16, 439–454. Wright, T.P., 1936. J. Aeronaut. Sci. 3, 122–128.")
P("")
open(os.path.join(SUPP, "supplementary_part2.tex"), "w", encoding="utf-8").write("\n".join(out))
print("written", os.path.join(SUPP, "supplementary_part2.tex"), "tables S17–S24 ->", TAB)
