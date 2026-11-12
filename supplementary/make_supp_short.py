#!/usr/bin/env python3
"""make_supp_short.py - the supplementary material of the manuscript (Supplementary Notes 1-7, Tables S1-S16).

It keeps the settings that are needed to reproduce the results and the tables that the main text cites but
does not print. The table bodies are taken from the extended version in ../extended_results, whose generators
write them from the archived result files. The notes are written here. Numbers of tables, notes, equations and
sections follow the manuscript. README.md in this folder maps every table to its source.

    python make_supp_short.py OUT_DIR        (writes OUT_DIR/supplementary.tex for XeLaTeX)

The two lines under the heading of page 1 are set with the environment variables SUPP_TITLE and SUPP_SUBTITLE,
and SUPP_LONG sets the folder of the extended version.
"""
import csv, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import longtex

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
TITLE = os.environ.get("SUPP_TITLE", "A hierarchical gamma--Poisson early-warning model of supply shocks with cost-based alarms")
SUBTITLE = os.environ.get("SUPP_SUBTITLE", "Supplementary material to the manuscript")
T = longtex.tables()
TABLES_CSV = os.path.join(longtex.SUPP, "tables")

def drop_cols(tabular, drop, spec):
    """remove 1-based columns from a tabular without \\multicolumn rows"""
    out = []
    for line in tabular.split("\n"):
        if line.startswith("\\begin{tabular}"): out.append("\\begin{tabular}{" + spec + "}"); continue
        if "&" in line:
            body, tail = line.rsplit("\\\\", 1)
            cells = body.split("&"); assert "multicolumn" not in line, line
            cells = [c for k, c in enumerate(cells, 1) if k not in drop]
            line = "&".join(cells).rstrip() + " \\\\" + tail
        out.append(line)
    return "\n".join(out)

def block(n, caption, tabular, size="footnotesize", note="", fit=False):
    if fit: tabular = "\\resizebox{\\linewidth}{!}{%\n" + tabular + "}"
    s ="\\vspace{6pt}\n\\noindent\\begin{minipage}{\\linewidth}\n\\captionof{table}{Table S%d. %s}\n\\centering\\%s\n%s\n" % (n, caption, size, tabular)
    if note: s += "\n\\vspace{2pt}\\raggedright " + note + "\n"
    return s + "\\end{minipage}\n\\vspace{8pt}\n"

def regime_table():
    """Table S3: the Study B results by regime, from the three regime tables of the extended supplement"""
    R = {}
    for reg in ("stable", "gradual", "abrupt"):
        rows = list(csv.reader(open(os.path.join(TABLES_CSV, f"S2_table6_{reg}.csv"), encoding="utf-8")))
        h = rows[0]; R[reg] = {r[0]: dict(zip(h, r)) for r in rows[1:]}
    v = lambda x: re.sub(r"\s*\(.*\)$", "", x).replace("–", "--") if x not in ("", "–") else "--"
    meth = [r for r in R["stable"]]
    lines = ["\\begin{tabular}{l*{4}{r}*{4}{r}*{4}{r}}", "\\toprule",
             " & \\multicolumn{4}{c}{Stable} & \\multicolumn{4}{c}{Gradual} & \\multicolumn{4}{c}{Abrupt} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}",
             "Method & AUC-PR & ECE & Det. & ARL$_0$ & AUC-PR & ECE & Det. & ARL$_1$ & AUC-PR & ECE & Det. & ARL$_1$ \\\\ \\midrule"]
    for m_ in meth:
        c = [m_.replace("%", "\\%")]
        for reg, last in (("stable", "ARL₀"), ("gradual", "ARL₁"), ("abrupt", "ARL₁")):
            d = R[reg][m_]; c += [v(d["AUC-PR"]), v(d["ECE"]), v(d["Det. ≤1 mo"]), v(d[last])]
        lines.append(" & ".join(c) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)

def rare_part_b():
    s = open(os.path.join(longtex.SUPP, "supplementary_note17_rare_arm.tex"), encoding="utf-8").read()
    k = s.index("\n(b) Change channel and prediction sets\n"); a = s.index("\\begin{tabular}", k); b = s.index("\\end{tabular}", a) + len("\\end{tabular}")
    return drop_cols(s[a:b], {7, 8}, "llcccc")

NB_K = r"""\[
k \;=\; \hat{\alpha}\,\log\!\left[\frac{\hat{\alpha} + 1.5\hat{\mu}}{\hat{\alpha} + \hat{\mu}}\right] \Big/ \log\!\left[\frac{1.5\,(\hat{\alpha} + \hat{\mu})}{\hat{\alpha} + 1.5\hat{\mu}}\right],
\]"""

PRE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=20mm]{geometry}
\usepackage{fontspec}
\setmainfont{TeX Gyre Pagella}
\usepackage{booktabs,caption,array,url,amsmath,graphicx}
\setlength{\tabcolsep}{3.5pt}
\captionsetup{font=small,labelfont=bf,justification=justified,singlelinecheck=false,labelformat=empty,skip=4pt}
\usepackage{unicode-math}
\setmathfont{Latin Modern Math}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\pagestyle{plain}
\begin{document}
"""

def main():
    P = []; A = P.append
    A(PRE)
    A("{\\large\\bfseries Supplementary material}\\\\[2pt]\n{\\bfseries %s}\\\\[1pt]\n{\\small %s}\n" % (TITLE, SUBTITLE))
    A(r"""This supplement has two parts. Supplementary Notes 1--3 give the settings that are needed to reproduce the results: the sampler and its convergence criteria, the baselines and the severity submodel. Supplementary Notes 4--7 tabulate results that the main text cites but does not print: further results of the simulation study, the comparisons of expected cost, the backtest and the rare arm. All numbers are computed from the archived result files of the repository.

\textbf{Conventions.} Entries are means over replications, with Monte Carlo standard errors in brackets where they are shown. Detection within one month counts a shock as detected when the level is in force in the month before it. False alarms are alarm onsets in the stable windows that no shock follows within three months, per item-year. ARL$_0$ is the ratio of unit-months in the stable windows to alarm months, pooled over the replications. The units are groups for the $q$ channels and items for the other rules. ARL$_1$ is the mean delay from the change to the first alarm. Posterior intervals are 90\% equal-tailed intervals.
""")
    # ------------------------------------------------------------------ Note 1
    A(r"""\textbf{Supplementary Note 1. Sampler settings and convergence criteria (Section 4.1).}
Posteriors are sampled with the No-U-Turn sampler in NumPyro. Each fit uses four chains with 400 warm-up and 500 sampling iterations, a target acceptance probability of 0.95 and a maximum tree depth of 10. The random-walk and hierarchical terms use non-centred parameterisations. The trend is sampled in reparameterised coordinates, namely the least-squares level and slope of the trend path of each group. This linear map has unit Jacobian, and it removes the ridge between the initial slope and the slope innovations. Chains start at the prior medians with small jitter, except $\sigma_\eta$ and $\tau_\zeta$, which start at 0.03 and 0.01.

A fit is accepted when the rank-normalised $\widehat{R}$ is below 1.01, the bulk and tail effective sample sizes exceed 400 on the reported quantities, and no transition diverges. A fit that fails any criterion is repeated once on the same data with 1,000 warm-up and 1,000 sampling iterations. A fit that still fails is recorded as failed, and it is never replaced by a new draw of the data. Posterior predictive checks compare four statistics with their replicate distributions: the proportion of zero counts, the maximum count, the index of dispersion by group, and the autocorrelation of the counts at lags one to three.
""")
    # ------------------------------------------------------------------ Note 2
    A(r"""\textbf{Supplementary Note 2. Baseline specifications (Section 5.1).}
\textit{Sigma-rule.} The plug-in SRI is the SRI at the posterior means of $\mu$ and $\alpha'$ from the M2 filter. Its limits lie 1$\sigma$, 1.5$\sigma$ and 2$\sigma$ below the mean of the preceding 24 months.

\textit{Control charts.} The EWMA uses $\lambda = 0.2$, and the Poisson CUSUM uses the reference value of Lucas (1985) for a 1.5-fold shift. Both charts freeze their in-control means at months 13--36, and their limits are set for a nominal Poisson ARL$_0$ of 24 months. The CUSUM that is designed under the negative binomial uses the likelihood-ratio reference value for the same shift,
""" + NB_K + r"""
where the dispersion $\hat{\alpha}$ is the pooled moment estimate within the group of the item. Its limit is the $(1 - 1/24)$ quantile of the in-control distribution of the statistic, which is simulated from the fitted negative binomial.

\textit{Farrington-type threshold.} At every decision month and for every item, a quasi-Poisson regression with an intercept and a linear trend is fitted to the preceding 36 months. Past outliers are down-weighted once, and the estimated dispersion turns the fitted mean into a negative-binomial predictive distribution. The item raises an alarm when the count of the current month exceeds the $(1 - 1/24)$ quantile of this distribution.

\textit{Machine-learning baselines.} XGBoost uses a depth of 3 or 6, a learning rate of 0.1 and up to 300 rounds. The LSTM and the temporal fusion transformer have 16 hidden units on windows of twelve months. All three are trained on the indicators, the exposure measurements and the count history of the item, with the same rolling-origin splits. Their hyperparameters are chosen on the last twelve labelled months of the training range of each origin.
""")
    # ------------------------------------------------------------------ Note 3
    A(r"""\textbf{Supplementary Note 3. Severity submodel (Section 4.3).}
A period without a shock has no loss, so the loss variable has a point mass at zero. We therefore use a hurdle model,
\begin{align}\mathrm{SDR}_{it} &= \mathbf{1}\{N_{it}\ge 1\}\,S_{it}, \tag{S3.1}\\ S_{it}\mid N_{it}\ge 1 &\sim \mathrm{Beta}\big(\phi\,m_{it},\ \phi\,(1-m_{it})\big), \tag{S3.2}\\ \mathrm{logit}\,m_{it} &= \gamma_{0} + \gamma_{1}\log\mu_{it} + \boldsymbol{\gamma}_{2}^{\top}\mathbf{x}_{i,t-1}. \tag{S3.3}\end{align}
Equations (S3.2)--(S3.3) are a beta regression for the loss fraction $S_{it}$ given a shock, with precision $\phi$. For heavy-tailed severities, a generalised-Pareto tail complements the beta body. The tail is fitted by the peaks-over-threshold method to the exceedances of the observed loss fraction over its 90th percentile, once at least ten exceedances are available. The priors are $\phi \sim \mathrm{Gamma}(2, 0.1)$ and $\gamma \sim N(0,1)$. The submodel is fitted at every monitoring origin.
""")
    # ------------------------------------------------------------------ Note 4
    A(r"""\textbf{Supplementary Note 4. Further results of the simulation study (Sections 5.2--5.4).}
Tables S1 and S2 give the triage results at levels 1 and 3, which the cost evaluation of Section 5.6 uses. Table S3 gives the results by regime, and Table S4 gives the triage results at level 2 by cell. Table S5 gives the coverage of the posterior intervals in Study A by factor level.

\textit{Refit schedule.} We also refitted M2 every 24 months instead of every six, on the same 65 replications. The AUC-PR is then 0.600 against 0.602, and the ARL$_0$ in the stable windows is 32 months against 35. The ARL$_1$ is 20.6 against 20.8 months, and the other metrics differ by at most 0.005. The fitting time falls to a quarter.

\textit{Priors and indicator information.} We varied the priors of Section 4.1 in the reference cell, with five replications each. Centring the dispersion hyperprior at $\log 1$ or $\log 5$ instead of $\log 2$ moves the posterior median of $\alpha'$ from 2.15 to 2.12 and 2.19. Doubling or quintupling the scales of the trend innovations leaves the level path unchanged. An AHP prior that points the wrong way, $\mathbf{w}^{\mathrm{AHP}} = (0.375, 0.625)$, pulls the posterior median of $\mathbf{w}$ from $(0.53, 0.31)$ to $(0.47, 0.39)$ for a truth of $(0.5, 0.3)$. However, it leaves the quantities that drive the alarms essentially unchanged. When the indicator information is removed from the truth ($\boldsymbol{\beta} = 0$), the alarm pipeline gives an AUC-PR of 0.80, against 0.81 in the reference cell.

\textit{Robustness arms.} Table S6 gives the alarm metrics of M2 in the robustness arms. Each arm is the reference design with one factor changed, and it has five replications. In a further arm, the gamma multiplier of (2) is replaced in the truth by a lognormal multiplier with the same mean and variance. The fitted model then recovers the variance-matched dispersion (posterior median 2.42, against 2.35 under the gamma truth), and the AUC-PR is 0.805 against 0.807.
""")
    A(block(1, r"Triage at level 1 (cost ratio 1:20, threshold $\pi_1$ = 0.048, two consecutive months) in Study B, pooled over the twelve cells (65 replications). The columns give the share of shocks flagged within one and within three months before they occur, the false alarms per item-year in the stable windows and the median lead time in months. The sigma-rule is shown at its own level 1. EWMA, CUSUM and the Farrington-type threshold have one level, so their rows are the same in Table S2. Level 2 is in Table 4(a) of the main text.", T["1.1"]["tabular"]))
    A(block(2, r"Triage at level 3 (cost ratio 1:1, threshold $\pi_3$ = 0.5, two consecutive months) in Study B. The columns are those of Table S1, and the sigma-rule is shown at its own level 3.", T["1.3"]["tabular"]))
    A(block(3, r"Study B by regime: AUC-PR and ECE of the event probabilities, share of shocks flagged within one month at level 2 (Det.), ARL$_0$ in the stable cells and ARL$_1$ after the change, in months. Entries are means over the replications of the regime (four cells each; 20 replications in the stable regime, 25 in the gradual regime and 20 in the abrupt regime). The rows of the model-based methods give the ARL of their $q$ channel. For the sigma-rule, the detection column is at its level 2 and the run lengths are at its first limit, as in Table 4 of the main text.", regime_table(), size="scriptsize"))
    A(block(4, r"Triage at level 2 by cell of Study B. Each entry gives the share of shocks flagged within one month and, after the slash, the false alarms per item-year in the stable windows (means over the replications of the cell).", T["3.2"]["tabular"], size="scriptsize"))
    A(block(5, r"Study A, hierarchical model (538 fits): realised coverage of the 90\% posterior intervals and RMSE by factor level. All fits of a level are included, also the 4.6\% that failed a convergence check ($n$ = number of fits; Failed = share of failed fits).", T["4"]["tabular"], size="scriptsize"))
    t24 = drop_cols(T["24"]["tabular"], {9}, ">{\\raggedright\\arraybackslash}p{3.4cm}rrrrrrrrrr")
    A(block(6, r"Alarm metrics of M2 in the robustness arms of Section 5.4 (five replications each; the first row is the reference cell of Study B with ten replications). The columns give the AUC-PR, the Brier score and the ECE of the three-month event probability, the detection rate within one month at triage levels 2 and 3 (L2, L3), the false alarms per item-year in the stable windows at level 2, the mean delay of the $q$ channel after the change in months, the posterior median of $\alpha'$ (truth 2) and the RMSE of $\log C_i$.", t24, size="scriptsize"))
    # ------------------------------------------------------------------ Note 5
    A(r"""\textbf{Supplementary Note 5. Expected cost of the alarm policies (Section 5.6).}
\textit{Inputs.} All quantities come from the archived results of Study B. For method $m$ at level $k$ in replication $j$, the first quantity is the detection rate within one month, $\mathrm{DR}_{mkj}$. The second is the share of decision item-months in which the level is in force. This share gives the alarm months per item-year, $\mathrm{AM}_{mkj}$. The number of shock months per item-year, $E_j$, is counted by regenerating each simulated panel from its archived seed.

\textit{Cost model.} The expected cost is $\mathrm{EC}_{mkj}(r) = \mathrm{AM}_{mkj} + r\,(1-\mathrm{DR}_{mkj})\,E_{j}$, which is (18) of the main text. Never acting costs $rE_{j}$, and always acting costs 12. A method with several levels is reported at its cheapest level at each $r$.

\textit{Tables.} Table S7 gives the saving of M2 relative to each policy, with a sign test over the replications. Table S8 gives the values of $r$ at which the cheapest level of M2 switches, together with its value of monitoring and its break-even ratios. Table S9 gives the value of monitoring by method, and Table S10 compares the methods at a similar alarm budget.

\textit{Sensitivity.} Suppose that a shock counts as detected when the level was in force at any time in the three months before it. The cost of every alarm policy then falls, by 1\% to 16\% at $r = 5$ and by 1\% to 21\% at $r = 20$. Even so, every ordering of the main text stays in place. Weighting the loss by the mean loss fraction rescales $r$ and leaves the curves otherwise unchanged.
""")
    A(block(7, r"Saving of M2 relative to each policy in Study B (all twelve cells): $1 - \mathrm{EC}_{\mathrm{M2}}/\mathrm{EC}_{m}$ on the means, each method at its cheapest level. The brackets give the share of the 65 replications in which M2 is cheaper and the $p$-value of a two-sided sign test (ties excluded). The last row takes the cheaper of never acting and always acting in each replication.", T["20"]["tabular"], size="scriptsize", fit=True))
    A(block(8, r"M2 in Study B, for all cells and for the two base rates: the values of $r$ at which the cheapest level switches from 3 to 2 and from 2 to 1, the value of monitoring $1 - \mathrm{EC}_{\mathrm{M2}}/\min(rE, 12)$ (its maximum and the range of $r$ over which it exceeds 10\%), and the break-even ratios $r^{*}$ at which the cost curve of M2 crosses that of never acting, of always acting, of XGBoost, of the LSTM and of the CUSUM (`none' = no crossing on $[1, 100]$).", T["21.1"]["tabular"], size="scriptsize"))
    A(block(9, r"Value of monitoring by method: the largest saving of each policy relative to the better of never acting and always acting, $\max_r\,[1 - \mathrm{EC}_m(r)/\min(rE, 12)]$, and the $r$ at which it occurs, for all cells and for the two base rates.", T["21.2"]["tabular"]))
    A(block(10, r"Alarm budget and detection in Study B (all cells, means over the 65 replications): alarm months per item-year, the share of decision item-months in which the level is in force, the detection rate within one month, and the ratio of this rate to the share in force. The ratio compares the alarm with one that is switched on at random for the same share of months (1 = no skill).", T["21.3"]["tabular"]))
    # ------------------------------------------------------------------ Note 6
    A(r"""\textbf{Supplementary Note 6. The backtest on public trade data (Section 5.5).}
\textit{Data.} The panel holds the monthly imports of the Republic of Korea by partner at the six-digit HS level. We extracted them from the UN Comtrade data API on 26 October 2026 for the 65 items of Table S11. One item, unwrought molybdenum (HS 810291), returned no records, which leaves 64 items. Quantities are net weights in kilograms. They begin in January 2014, because the records of 2013 carry no weights, and the panel begins in February 2014. Decisions are made from month 36 (February 2017) to December 2025, which gives 107 months.

\textit{An item whose code changed.} Bismuth (HS 810600) has no record under this code after December 2021, because the 2022 edition of the Harmonized System divides heading 81.06 into two subheadings. The panel therefore shows zero imports of bismuth in 2022 and has no row for it from January 2023. Seven of its 21 trade interruptions fall in 2022, and they result from the change of code, not from a change in supply. Bismuth stays in every table, because dropping it changes the results little. Without bismuth, the AUC-PR of M2 is 0.611 instead of 0.600, and its calibration error is 0.024 instead of 0.029. Its detection rate within one month at level 2 stays at 0.91.

\textit{Events.} A trade interruption (E1) concerns a partner that supplied at least a quarter of the imports of an item over the trailing twelve months. It is recorded when this partner ships less than half of its own median over the same months. An export-restriction implementation (E2) is recorded when a partner with a trailing share of at least 5\% implements a measure whose HS scope covers the item. The first version of the plan used a share threshold of 5\% for E1 as well. On the panel, that definition fired in 37\% of the decision item-months, which is outside the range of the simulation design. We therefore raised the threshold to 25\% before the main run and fixed it. Table S14 reports two alternative definitions of E1 as sensitivity runs, in which the shipments of the largest partner halve or the total imports of the item halve.

\textit{Measures.} The export-restriction measures come from three sources. The first is the OECD Inventory of Export Restrictions on Industrial Raw Materials (2009--2021), and the second is the Global Trade Alert database (extracted on 26 October 2026). The third is a list of eleven documented measures whose dates were checked against the Global Trade Alert records. From the Global Trade Alert records, we keep the interventions of five export types whose affected jurisdictions include the Republic of Korea. The five types are export bans, export licensing requirements, export quotas, export taxes and other export-related non-tariff measures. We drop measures that list more than 100 six-digit codes, because they are economy-wide frameworks. A measure that appears in more than one source is counted once per partner and month. The resulting table holds 307 measures.

\textit{Suspended measures.} The list of documented measures dates the October 2025 extension of China's rare-earth controls to November 2025, and this entry gives three of the 64 events of Section 5.5. China suspended the extension on 7 November 2025 until 10 November 2026, and the Global Trade Alert records as extracted carry the later date. The tables keep the three events, and Section 5.5 also gives the shares without them.

\textit{Fits.} The protocol is that of Section 5.3, with refits at 18 origins. Each fit uses two chains with 500 warm-up and 600 sampling iterations. A fit was repeated once with 1,000 warm-up and 1,000 sampling iterations when $\widehat{R}$ reached 1.05 or the bulk effective sample size fell below 100 on any stored quantity. Six of the 18 fits of M2 were repeated, and none of the Poisson fits. The fits in use have $\widehat{R} \leq 1.043$ and effective sample sizes of at least 56. XGBoost is pooled across the items. A second variant recalibrates its probabilities by isotonic regression on the twelve labelled months before each origin. Tables S12 and S13 give the alarm metrics at every level for E1 and E2.
""")
    A(block(11, r"The 65 items: six-digit code of HS 2017, description, group, months with positive imports (February 2014 to December 2025), and the item-months with E1 and E2 in the decision window.", T["25.2"]["tabular"], size="tiny"))
    A(block(12, r"Alarm performance on the Korean import panel for trade interruptions (E1): event probabilities for `at least one event within three months' and the alarm at each level. The columns give the events flagged within one and within three months before they occur, the false alarm onsets per item-year, the alarm months per item-year and the lead time in months (median and quartiles). The last rows give the $q$ channel and the headline level of the two models.", T["26.1"]["tabular"], size="tiny"))
    A(block(13, r"Alarm performance on the Korean import panel for export-restriction implementations (E2). The columns are those of Table S12.", T["26.2"]["tabular"], size="tiny"))
    A(block(14, r"Sensitivity of the backtest to the event definitions, at level 2 of the probability-based methods and of the sigma-rule. The first block uses the strict list of E2 (new export controls only, without the annual renewals of catalogues and quotas), evaluated with the fits of the main run. The other blocks use the two alternative definitions of E1, each as a separate run on the same 64 items. AUC-PR and ECE refer to the E1 probabilities and are not reported for the E2 rows. The last column gives the event item-months in the decision window.", T["28"]["tabular"], size="tiny"))
    # ------------------------------------------------------------------ Note 7
    A(r"""\textbf{Supplementary Note 7. The rare arm (Sections 5.1, 5.3 and 5.6).}
\textit{Design.} The rare arm has three cells at a base rate of $\bar{\mu} = 0.02$ shocks per item-month, with $I = 100$ items, $T = 120$ months, three groups and $\alpha' = 2$. The cells differ in the regime: stable, gradual ($+2\%$ per month from $t_0 = 60$) and abrupt ($+\log 3$ at $t_0 = 60$). Each cell has five replications, and everything else follows the protocol of Study B. The realised three-month prevalence of events is 8.2\%, 12.4\% and 16.7\% in the stable, gradual and abrupt cells, and the number of shock months per item-year is 0.34, 0.53 and 0.74. The methods are M2, the Poisson model M0, XGBoost, the sigma-rule, EWMA and CUSUM. All 210 origin fits of M2 passed the convergence check, 106 of them after the one refit.

\textit{Results.} Table S15 gives the metrics of Table 4 of the main text by cell and pooled, and Table S16 gives the expected cost of Section 5.6. The $q$ channel of M2 reached the 24-month target in all fifteen bootstrap calibrations, and the Poisson channel reached it in nine. In the ten replications with a regime change, neither channel raised an alarm in the 24 months before the change. In the stable cell, however, both channels raised alarms more often over the 82 decision months than the bootstrap had predicted.
""")
    r1 = T["29.1"]
    A("\\vspace{6pt}\n\\captionof{table}{Table S15. Alarm performance in the rare arm ($\\bar{\\mu} = 0.02$, $I = 100$, $T = 120$, $\\alpha' = 2$; five replications per cell), by cell and pooled, evaluated as Table 4 of the main text. (a) Event probabilities for `at least one shock within three months' and triage at level 2: detection in the month before the shock and within three months, false alarm onsets per item-year in the stable windows, alarm months per item-year and lead time in months. (b) Change channels: the calibrations that reached the 24-month target in the 200 bootstrap paths, the ARL$_0$ that the bootstrap predicted, the realised ARL$_0$ (median and range over the replications) and ARL$_1$ (mean and range). The ARL$_0$ of the $q$ channels is in group-months (three groups), and that of EWMA and CUSUM is in item-months (100 items). The values 72, 246 and 2400 therefore mean that no alarm occurred in the window.}\n{\\scriptsize\\centering\n(a) Event probabilities and triage at level 2\\\\[2pt]\n" + r1["tabular"] + "\n\n\\vspace{4pt}\n(b) Change channels\\\\[2pt]\n" + rare_part_b() + "\n\n}\n\\vspace{8pt}\n")
    r2 = T["29.2"]
    A("\\captionof{table}{Table S16. Expected cost per item-year in the rare arm (units of one alarm-month), by cell and pooled over the fifteen replications, computed as in Table 6 of the main text: for each policy the mean cost at its cheapest level, with the standard error in brackets. The last three columns give the band of $r$ in which the policy is cheaper than the better of never acting and always acting, its largest saving against that policy, which lies at $r = 12/E$ (34.9, 22.6 and 16.2 by cell; 22.3 pooled), and the range of $r$ over which the saving exceeds 10\\%. Never acting costs $rE$ per item-year ($E$ = 0.34, 0.53 and 0.74 by cell; 0.54 pooled), and always acting costs 12.}\n{\\scriptsize\\centering\n" + r2["tabular"] + "\n\n}\n\\vspace{2pt}{\\footnotesize " + r2["note"] + "}\n")
    A("\n\\end{document}\n")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "supplementary.tex"), "w", encoding="utf-8").write("\n".join(P))
    print("written", os.path.join(OUT, "supplementary.tex"))

if __name__ == "__main__":
    main()
