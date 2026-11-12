"""Supplementary Note 17 (the rare arm) with Tables S29.1-S29.4, in the style of supplementary/make_supp.py. Writes the body
of the note, which supplementary_part2.tex inputs after Supplementary Note 16 (no preamble: build supplementary.tex).
Usage: python rare_arm/gen_note17.py <repo root> <results_rare dir> supplementary/supplementary_note17_rare_arm.tex

Costs follow Section 5.6 of the manuscript exactly as Table 8 and Tables S21.1-S21.2 do (see rare_cost.py):
the cost of a policy at ratio r is the lower envelope of its level-wise MEAN cost lines over the replications of the
subset, and the two trivial policies are priced at the subset mean E, min(r E, 12)."""
import json, glob, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
root, out, texpath = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(root, "code"))
import numpy as np, config as C; C.apply_profile("t4_min"); import dgp
R = np.exp(np.linspace(0.0, np.log(300), 60001)); RS = [2, 5, 10, 20, 50]
def E_of(cell, rep, reg):
    s = dgp.simulate(100, 120, 0.02, 2.0, reg, seed=dgp.seed_of("B", cell, rep)); return (s["N"][:, 36:118] > 0).sum() / (100 * 82 / 12)
cells = {}
for f in sorted(glob.glob(os.path.join(out, "B", "*__all.jsonl"))):
    recs = sorted([json.loads(l) for l in open(f)], key=lambda r: r["rep"]); reg = recs[0]["cell"].split("-")[3]
    cells[reg] = [(r, E_of(r["cell"], r["rep"], reg)) for r in recs]
order = ["stable", "gradual", "abrupt"]; names = {"stable": "Stable", "gradual": "Gradual", "abrupt": "Abrupt", "all": "Pooled"}
allp = [p for reg in order for p in cells[reg]]; groups = [(reg, cells[reg]) for reg in order] + [("all", allp)]
def lv_prob(r, k): return [(r[k]["triage"][f"level{j}"]["dr_h1"], r[k]["triage"][f"level{j}"]["alarm_share"] * 12) for j in (1, 2, 3)]
def lv_sigma(r): return [(r["M2"]["sigma_rule"][f"level{j}"]["dr_h1"], r["M2"]["sigma_rule"][f"level{j}"]["alarm_share"] * 12) for j in (1, 2, 3)]
def lv_chart(r, c): return [(r["charts"][c]["events"]["dr_h1"], r["charts"][c]["events"]["alarm_share"] * 12)]
def lines_at(fn, pairs, rr):
    """cost[k, j]: level k, replication j, at ratio rr"""
    lv = [fn(r) for r, _ in pairs]; E = np.array([e for _, e in pairs])
    return np.array([[lv[j][k][1] + rr * (1 - lv[j][k][0]) * E[j] for j in range(len(pairs))] for k in range(len(lv[0]))])
def cost_at(fn, pairs, rr):
    """(mean, standard error, level index) at the level that is cheapest on average"""
    c = lines_at(fn, pairs, rr); k = int(c.mean(axis=1).argmin()); v = c[k]
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0), k
def curve(fn, pairs):
    lv = [fn(r) for r, _ in pairs]; E = np.array([e for _, e in pairs])
    return np.min([np.mean([lv[j][k][1] + R * (1 - lv[j][k][0]) * E[j] for j in range(len(pairs))], axis=0) for k in range(len(lv[0]))], axis=0)
methods = [("M2 (proposed)", lambda r: lv_prob(r, "M2")), ("Poisson M0", lambda r: lv_prob(r, "M0")), ("XGBoost", lambda r: lv_prob(r, "xgb")),
           ("Sigma-rule (3 limits)", lv_sigma), ("EWMA", lambda r: lv_chart(r, "ewma")), ("CUSUM", lambda r: lv_chart(r, "cusum"))]
def mse(v): v = np.asarray(v, float); return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
def f(x, d=2): return f"{x:.{d}f}"
def band(lvfn, pairs):
    """band of r with a positive saving against min(r E, 12), largest saving, its r, and the range with a saving above 10%"""
    ec = curve(lvfn, pairs); Eb = np.mean([E for _, E in pairs]); triv = np.minimum(R * Eb, 12.0)
    sv = 1 - ec / triv; b = sv > 0; g = sv > 0.10; one = sv > 0.01
    return ((R[b].min(), R[b].max()) if b.any() else None, sv.max() * 100, R[sv.argmax()], (R[g].min(), R[g].max()) if g.any() else None, (R[one].min(), R[one].max()) if one.any() else None)
def paired(fa, fb, pairs, rr):
    """saving of policy a over policy b on the means, 1 - EC_a/EC_b (%), and the replications in which a is cheaper"""
    ca = lines_at(fa, pairs, rr); cb = lines_at(fb, pairs, rr); a = ca[int(ca.mean(axis=1).argmin())]; b = cb[int(cb.mean(axis=1).argmin())]
    return 100 * (1 - a.mean() / b.mean()), int((a < b).sum()), len(a)
M2f, M0f, XGf = methods[0][1], methods[1][1], methods[2][1]
def lead(T):
    med = np.nanmean([t["lead_med"] if t["lead_med"] is not None else np.nan for t in T]); q1 = np.nanmean([t["lead_q25"] if t["lead_q25"] is not None else np.nan for t in T]); q3 = np.nanmean([t["lead_q75"] if t["lead_q75"] is not None else np.nan for t in T])
    return f"{med:.1f} ({q1:.1f}--{q3:.1f})"
# ---------------- numbers quoted in the text (computed, not typed)
bd = {reg: band(M2f, pairs) for reg, pairs in groups}; bx = {reg: band(XGf, pairs) for reg, pairs in groups}; b0 = {reg: band(M0f, pairs) for reg, pairs in groups}
Ebar = {reg: float(np.mean([E for _, E in pairs])) for reg, pairs in groups}
def sw(pairs):
    lv = [M2f(r) for r, _ in pairs]; E = np.array([e for _, e in pairs])
    best = np.argmin([np.mean([lv[j][k][1] + R * (1 - lv[j][k][0]) * E[j] for j in range(len(pairs))], axis=0) for k in range(3)], axis=0)
    return [R[i] for i in range(1, R.size) if best[i] != best[i - 1]]
swp = sw(allp)
cells_only = [(reg, cells[reg]) for reg in order]
p0 = {reg: [paired(M2f, M0f, pairs, rr) for rr in RS] for reg, pairs in groups}       # M2 against the Poisson model
px = {reg: [paired(M2f, XGf, pairs, rr) for rr in RS] for reg, pairs in groups}       # M2 against XGBoost
sg = max(abs(d) for reg in ("stable", "gradual") for d, _, _ in p0[reg])
ab = [d for d, _, _ in p0["abrupt"]]; abw = [w for _, w, _ in p0["abrupt"]]
x20 = [px[reg][3][0] for reg in order]; x50 = [px[reg][4][0] for reg in order]; x2 = [-px[reg][0][0] for reg in ("gradual", "abrupt")]
assert all(px[reg][3][1] == 5 and px[reg][4][1] == 5 for reg in order)
nev5 = max(abs(100 * (cost_at(fn, cells[reg], 5)[0] / (5 * Ebar[reg]) - 1)) for reg in order for fn in (M2f, M0f, XGf))
oracle = float(np.mean([min(am + 20 * (1 - dr) * E for dr, am in XGf(r)) for r, E in allp])); oracle2 = float(np.mean([min(am + 20 * (1 - dr) * E for dr, am in M2f(r)) for r, E in allp]))
cnt = {reg: float(np.mean([(dgp.simulate(100, 120, 0.02, 2.0, reg, seed=dgp.seed_of("B", r["cell"], r["rep"]))["N"][:, 36:118]).sum() / (100 * 82 / 12) for r, _ in cells[reg]])) for reg in order}
def rng(v, d=0):
    a, b = f"{min(v):.{d}f}", f"{max(v):.{d}f}"
    return a if a == b else f"{a}--{b}"
tex = []
tex.append(r"""\textbf{Supplementary Note 17. The rare arm (Sections 5.1, 5.3 and 5.6 of the main text).}

\textbf{Design.} Three cells at a base rate of $\bar{\mu} = 0.02$ shocks per item-month (0.24 per item-year), $I = 100$ items, $T = 120$ months, $G = 3$ groups, $\alpha' = 2$, one per regime---stable, gradual ($+2\%$ per month from $t_0 = 60$) and abrupt ($+\log 3$ at $t_0 = 60$)---with five replications each (cells \texttt{B-a2-mu0.02-\{stable,gradual,abrupt\}-I100}, seeds \texttt{seed\_of("B", cell, rep)}). Everything else is the Study B protocol: the \texttt{t4\_min} design constants (four chains, 400 warm-up and 500 sampling iterations, fourteen origins $36, 42, \dots, 114$, 200 bootstrap paths for the $q$ channel, M0 refitted every four origins), the same decision months, thresholds, persistence rules and metrics (conventions on page 1). The realised three-month event prevalence is 8.2\%, 12.4\% and 16.7\% in the stable, gradual and abrupt cells (24--55\% in Study B), and $E$---shock item-months per item-year over the decision months, an item-month with several shocks counting once---is """ + ", ".join(f"{Ebar[reg]:.2f}" for reg in order[:2]) + f" and {Ebar['abrupt']:.2f}" + r""" (1.13 and 3.15 at the two base rates of Study B); the corresponding numbers of shocks per item-year are """ + ", ".join(f"{cnt[reg]:.2f}" for reg in order[:2]) + f" and {cnt['abrupt']:.2f}" + r""". The methods run are M2, the Poisson model M0, XGBoost, the sigma-rule, EWMA and CUSUM; the NB pooling variants, the Farrington-type threshold, the negative-binomial CUSUM and the two recurrent networks (LSTM, TFT) were not run in this arm. All 210 origin fits of M2 passed the convergence check (106 after the one refit; maximum $\widehat{R}$ 1.0099); no M0 fit failed. Conformal coverage was 0.93--0.94 (target 0.90) with 1.0--1.2 counts per set. The arm was run on two CPU cores in 2-hour chunks resumed from an origin-level cache; the two replications of an earlier pilot run on a different machine reproduce to every recorded digit, as the fixed seeds imply.

\textbf{Reading.} Table S29.1 gives the Table 6 metrics by cell and pooled; Table S29.2 the expected cost of Section 5.6, computed as in Table 8 and Tables S21.1--S21.2; Tables S29.3 and S29.4 the per-replication values behind the paired statements of the main text. \emph{Probabilities and triage.} The ranking advantage of M2 over the Poisson model is confined to the abrupt cell (AUC-PR $+0.014$, five of five replications; stable and gradual within $0.004$); the calibration advantage holds in every changing-regime replication (ECE 0.012 and 0.019 against 0.017 and 0.027) and is absent in the stable cell (0.011 against 0.010); XGBoost ranks worse in all fifteen replications and rarely crosses the level-2 threshold. \emph{Change channel.} The $q$ channel of M2 reached the 24-month target in all fifteen bootstrap calibrations (thresholds 0.90--0.975, persistence 2, predicted ARL$_0$ 24--37) and the Poisson channel in nine (one of five in the gradual cell, where even the 0.99 threshold with persistence 3--4 predicted 15--25); in the ten changing-regime replications neither channel alarmed in the 24-month pre-change window, and M2 signalled the change sooner in every replication (abrupt 9.5 against 17.8 months, gradual 33.5 against 43.7). In the stable cell both channels alarmed more often over the 82-month window than the bootstrap had predicted: realised ARL$_0$ (group-months per alarm month) 15, 31, 7, 246 and 13 for M2 and 4, 12, 246, 246 and 8 for the Poisson model, below the target in three of five replications each (246 = no alarm). \emph{Cost.} Against Study B the band in which M2 beats both trivial policies moves from $r \approx 2.6$--19 to """ + f"${bd['all'][0][0]:.1f}$--{bd['all'][0][1]:.0f}" + r""" (pooled) and its centre $12/E$ from 5.4 to """ + f"{12 / Ebar['all']:.0f}" + r"""; the largest saving against the better trivial policy is """ + rng([bd[reg][1] for reg in order]) + r"""\% by cell and """ + f"{bd['all'][1]:.0f}" + r"""\% over the three cells (Study B: 30\% over its twelve cells, 23--25\% within a base rate; Table S21.1), against """ + rng([b0[reg][1] for reg in order]) + r"""\% for the Poisson model and """ + rng([bx[reg][1] for reg in order]) + r"""\% for XGBoost, and it exceeds 10\% for $r$ between """ + f"{bd['all'][3][0]:.0f} and {bd['all'][3][1]:.0f}" + r""" (pooled). The cheapest level of M2 switches from 3 to 2 near $r \approx """ + f"{swp[0]:.0f}" + r"""$ and from 2 to 1 near $r \approx """ + f"{swp[1]:.0f}" + r"""$ (7 and 18 in Study B). At small ratios there is little to save: at $r = 5$ the three probability-based alarms cost within """ + f"{nev5:.1f}" + r"""\% of never acting in every cell. Paired by replication, at the level that is cheapest on average in the cell (the convention of Table S20; Table S29.4), M2 and the Poisson model cost within """ + f"{sg:.1f}" + r"""\% of each other in the stable and gradual cells at every tabulated ratio, the Poisson model being the cheaper at some; in the abrupt cell M2 is """ + rng(ab, 1) + r"""\% cheaper at every tabulated ratio, in """ + ("four or five" if min(abw) == 4 else str(min(abw))) + r""" of the five replications. Against XGBoost, M2 is """ + rng(x20) + r"""\% cheaper at $r = 20$ and """ + rng(x50) + r"""\% at $r = 50$, in every replication of every cell, and """ + rng(x2) + r"""\% dearer at $r = 2$ in the two changing-regime cells, as in Study B, where the learners are cheaper when misses are cheap. The level of a policy is fixed per cell at each $r$, as in Table 8; choosing it replication by replication, which no policy could do in advance, would lower the pooled cost of XGBoost at $r = 20$ from """ + f"{cost_at(XGf, allp, 20)[0]:.2f} to {oracle:.2f}" + r""" and leave that of M2 at """ + f"{oracle2:.2f}" + r""", narrowing the pooled gap from """ + f"{px['all'][3][0]:.0f}\\% to {100 * (1 - oracle2 / oracle):.0f}\\%" + r""".
""")
# ---------------- Table S29.1 (a)
tex.append(r"""\clearpage
\captionof{table}{Table S29.1. Alarm performance in the rare arm ($\bar{\mu} = 0.02$, $I = 100$, $T = 120$, $\alpha' = 2$; five replications per cell), by cell and pooled, evaluated exactly as Table 6 of the main text. (a) Event probabilities for ``at least one shock within three months'' and triage at level 2 ($\pi_2 = 0.167$, two consecutive months): detection in the month before the shock and within three months, false-alarm onsets per item-year in the stable windows, alarm months per item-year, median lead time (mean of the replication quartiles). (b) Change channel and prediction sets: calibrations reaching the 24-month ARL$_0$ target in the 200 bootstrap paths, the bootstrap-predicted ARL$_0$, the realised ARL$_0$ (median and range over replications; 72 and 246 = no alarm in the 24-month pre-change window or the 82-month stable window), ARL$_1$ (mean over replications; range), conformal coverage and set size.}\label{tab:S29.1}
{\scriptsize
\begin{tabular}{llrrrrrrrl}
\toprule
Cell & Method & AUC-PR & Brier & ECE & Det.\ $\le$1 m & Det.\ $\le$3 m & FA / item-yr & Alarm-mo / item-yr & Lead, median (IQR) \\
\midrule""")
for reg, pairs in groups:
    first = True
    for name, k in (("M2 (proposed)", "M2"), ("Poisson M0", "M0"), ("XGBoost", "xgb")):
        P = [r[k]["prob"] for r, _ in pairs]; T = [r[k]["triage"]["level2"] for r, _ in pairs]
        tex.append(f"{names[reg] if first else ''} & {name} & {np.mean([p['aucpr'] for p in P]):.3f} & {np.mean([p['brier'] for p in P]):.3f} & {np.mean([p['ece'] for p in P]):.3f} & {np.mean([t['dr_h1'] for t in T]):.2f} & {np.mean([t['dr_h3'] for t in T]):.2f} & {np.mean([t['fa_per_itemyear'] for t in T]):.2f} & {np.mean([t['alarm_share']*12 for t in T]):.1f} & {lead(T)} \\\\"); first = False
    T = [r["M2"]["sigma_rule"]["level2"] for r, _ in pairs]
    tex.append(f" & Sigma-rule, level 2 & -- & -- & -- & {np.mean([t['dr_h1'] for t in T]):.2f} & {np.mean([t['dr_h3'] for t in T]):.2f} & {np.mean([t['fa_per_itemyear'] for t in T]):.2f} & {np.mean([t['alarm_share']*12 for t in T]):.1f} & {lead(T)} \\\\")
    for c in ("ewma", "cusum"):
        T = [r["charts"][c]["events"] for r, _ in pairs]
        tex.append(f" & {c.upper()} & -- & -- & -- & {np.mean([t['dr_h1'] for t in T]):.2f} & {np.mean([t['dr_h3'] for t in T]):.2f} & {np.mean([t['fa_per_itemyear'] for t in T]):.2f} & {np.mean([t['alarm_share']*12 for t in T]):.1f} & {lead(T)} \\\\")
    if reg != "all": tex.append(r"\addlinespace")
tex.append(r"""\bottomrule
\end{tabular}

\vspace{4pt}
(b) Change channel and prediction sets

\begin{tabular}{llcccccc}
\toprule
Cell & Channel & Calibrated & Predicted ARL$_0$ & Realised ARL$_0$, median (range) & ARL$_1$, mean (range) & Set coverage & Set size \\
\midrule""")
for reg, pairs in groups:
    first = True
    for name, k in (("$q$ channel, M2", "M2"), ("$q$ channel, Poisson M0", "M0")):
        a0 = np.array([r[k]["q"]["arl0"] for r, _ in pairs]); a1 = np.array([x for x in (r[k]["q"].get("arl1") for r, _ in pairs) if x is not None], float)
        cal = sum(r[k]["calib"]["q_calibrated"] for r, _ in pairs); boot = np.array([r[k]["calib"]["q_arl0_boot"] for r, _ in pairs])
        arl1 = f"{a1.mean():.1f} ({a1.min():.0f}--{a1.max():.0f})" if a1.size else "--"
        cov = f"{np.mean([r['M2']['conformal']['conf_cover'] for r,_ in pairs]):.3f}" if k == "M2" else "--"; size = f"{np.mean([r['M2']['conformal']['conf_size'] for r,_ in pairs]):.2f}" if k == "M2" else "--"
        tex.append(f"{names[reg] if first else ''} & {name} & {cal}/{len(pairs)} & {boot.min():.0f}--{boot.max():.0f} & {np.median(a0):.0f} ({a0.min():.0f}--{a0.max():.0f}) & {arl1} & {cov} & {size} \\\\"); first = False
    for c in ("ewma", "cusum"):
        rl = [r["charts"][c]["run_length"] for r, _ in pairs]; a0 = np.array([x["arl0"] for x in rl]); a1 = np.array([x["arl1"] for x in rl if x.get("arl1") is not None], float)
        arl1 = f"{a1.mean():.1f} ({a1.min():.0f}--{a1.max():.0f})" if a1.size else "--"
        tex.append(f" & {c.upper()} & -- & 24 (nominal) & {np.median(a0):.0f} ({a0.min():.0f}--{a0.max():.0f}) & {arl1} & -- & -- \\\\")
    if reg != "all": tex.append(r"\addlinespace")
tex.append(r"""\bottomrule
\end{tabular}}

Realised ARL$_0$ of the $q$ channels is in group-months (three groups), of EWMA and CUSUM in item-months (100 items), as in Table 6; the stable window is the 24 decision months before the change in the gradual and abrupt cells (72 group-months, 2,400 item-months) and all 82 decision months in the stable cell (246 and 8,200).
""")
# ---------------- Table S29.2
tex.append(r"""\clearpage
\captionof{table}{Table S29.2. Expected cost per item-year in the rare arm (Model A of Section 5.6, units of one alarm-month), by cell and pooled over the fifteen replications, computed as in Table 8: for each policy the lower envelope of its level-wise mean-cost lines (eq. 21), i.e. the mean over replications at the level that is cheapest on average at that $r$, with the standard error in brackets, at $r$ = 2, 5, 10, 20 and 50. The last three columns give, as in Tables S21.1 and S21.2, the band of $r$ in which the policy is cheaper than the better of never and always acting, $\min(rE, 12)$ with $E$ the mean of the subset, its largest saving against that policy, which lies at $r = 12/E$ ("""
           + ", ".join(f"{12 / Ebar[reg]:.1f}" for reg in order) + f"; {12 / Ebar['all']:.1f} pooled" + r"""), and the range of $r$ over which the saving exceeds 10\%. Never act costs $rE$ per item-year ($E$ = """
           + ", ".join(f"{Ebar[reg]:.2f}" for reg in order) + f"; {Ebar['all']:.2f} pooled" + r"""); always act costs 12.}\label{tab:S29.2}
{\scriptsize
\begin{tabular}{llrrrrrccc}
\toprule
Cell & Method & $r=2$ & $r=5$ & $r=10$ & $r=20$ & $r=50$ & Band of $r$ & Largest saving & Saving $>$ 10\% \\
\midrule""")
note_x = None
for reg, pairs in groups:
    first = True
    for name, fn in methods:
        vals = [cost_at(fn, pairs, rr) for rr in RS]; bnd, sv, at, g10, g1 = band(fn, pairs)
        if bnd is None: bstr = "--"
        elif bnd[0] < 1.05: bstr = f"$<$1--{bnd[1]:.0f}$^{{a}}$"; note_x = (names[reg], name, g1)
        else: bstr = f"{bnd[0]:.1f}--{bnd[1]:.0f}"
        gstr = f"{g10[0]:.0f}--{g10[1]:.0f}" if g10 else "--"
        tex.append(f"{names[reg] if first else ''} & {name} & " + " & ".join(f"{m:.2f} ({s:.2f})" for m, s, _ in vals) + f" & {bstr} & {sv:.0f}\\% & {gstr} \\\\"); first = False
    nev = [mse([rr * E for _, E in pairs]) for rr in RS]
    tex.append(" & Never act & " + " & ".join(f"{m:.2f} ({s:.2f})" for m, s in nev) + r" & -- & -- & -- \\")
    tex.append(" & Always act & " + " & ".join("12.00" for _ in RS) + r" & -- & -- & -- \\")
    if reg != "all": tex.append(r"\addlinespace")
tex.append(r"""\bottomrule
\end{tabular}}
""")
if note_x:
    tex.append(r"$^{a}$ " + f"{note_x[1]} in the {note_x[0].lower()} cell: the level-3 alarm is almost never in force, so that the policy costs less than never acting from $r = 1$ by under 1\\%; its saving exceeds 1\\% for $r$ between {note_x[2][0]:.0f} and {note_x[2][1]:.0f}." + "\n")
lvl = {reg: {name: [cost_at(fn, pairs, rr)[2] + 1 for rr in RS] for name, fn in methods[:4]} for reg, pairs in groups}
def lv_s(v): return "/".join(str(x) for x in v)
tex.append(r"Levels used at $r$ = 2/5/10/20/50 (the cheapest on average): " + "; ".join(f"{names[reg].lower()}: M2 {lv_s(lvl[reg]['M2 (proposed)'])}, Poisson M0 {lv_s(lvl[reg]['Poisson M0'])}, XGBoost {lv_s(lvl[reg]['XGBoost'])}, sigma-rule limit {lv_s(lvl[reg]['Sigma-rule (3 limits)'])}" for reg, _ in groups) + ".\n")
# ---------------- Table S29.3 per replication
tex.append(r"""\vspace{8pt}
\captionof{table}{Table S29.3. The rare arm replication by replication: AUC-PR and ECE of the event probabilities for M2, the Poisson model and XGBoost, the level-2 detection share within one month, and the realised ARL$_0$ and ARL$_1$ of the two $q$ channels (group-months; 72 and 246 = no alarm; ARL$_1$ in months after the change). The paired differences quoted in the main text are computed within these rows: AUC-PR of M2 exceeds that of the Poisson model in 1/5, 2/5 and 5/5 replications of the stable, gradual and abrupt cells and that of XGBoost in 15/15; the ECE of M2 is below that of the Poisson model in 3/5, 5/5 and 5/5 and below that of XGBoost in 2/5, 5/5 and 5/5.}\label{tab:S29.3}
{\scriptsize
\begin{tabular}{lrrrrrrrrrrrrr}
\toprule
 & & \multicolumn{3}{c}{AUC-PR} & \multicolumn{3}{c}{ECE} & \multicolumn{3}{c}{Det.\ $\le$1 m, level 2} & \multicolumn{2}{c}{$q$ channel ARL$_0$ / ARL$_1$} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-13}
Cell & Rep & M2 & M0 & XGB & M2 & M0 & XGB & M2 & M0 & XGB & M2 & M0 \\
\midrule""")
for reg in order:
    first = True
    for r, E in cells[reg]:
        def qs(k):
            a1 = r[k]["q"].get("arl1"); return f"{r[k]['q']['arl0']:.0f} / {a1:.0f}" if a1 is not None else f"{r[k]['q']['arl0']:.0f} / --"
        tex.append(f"{names[reg] if first else ''} & {r['rep']} & {r['M2']['prob']['aucpr']:.3f} & {r['M0']['prob']['aucpr']:.3f} & {r['xgb']['prob']['aucpr']:.3f} & {r['M2']['prob']['ece']:.4f} & {r['M0']['prob']['ece']:.4f} & {r['xgb']['prob']['ece']:.4f} & {r['M2']['triage']['level2']['dr_h1']:.2f} & {r['M0']['triage']['level2']['dr_h1']:.2f} & {r['xgb']['triage']['level2']['dr_h1']:.2f} & {qs('M2')} & {qs('M0')} \\\\"); first = False
    if reg != "abrupt": tex.append(r"\addlinespace")
tex.append(r"""\bottomrule
\end{tabular}}
""")
# ---------------- Table S29.4 per-replication costs
tex.append(r"""\clearpage
\captionof{table}{Table S29.4. Expected cost replication by replication in the rare arm (units of one alarm-month per item-year): $E_j$, the shock item-months per item-year of the replication, and the cost (21) of M2, the Poisson model and XGBoost at $r$ = 2, 5, 10, 20 and 50, each at the level that is cheapest on average in its cell at that $r$ (the levels listed under Table S29.2). The paired statements of Section 5.6 are computed within these rows; never acting costs $rE_j$ and always acting 12.}\label{tab:S29.4}
{\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrr*{5}{rrr}}
\toprule
 & & & \multicolumn{3}{c}{$r=2$} & \multicolumn{3}{c}{$r=5$} & \multicolumn{3}{c}{$r=10$} & \multicolumn{3}{c}{$r=20$} & \multicolumn{3}{c}{$r=50$} \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}\cmidrule(lr){13-15}\cmidrule(lr){16-18}
Cell & Rep & $E_j$ & M2 & M0 & XGB & M2 & M0 & XGB & M2 & M0 & XGB & M2 & M0 & XGB & M2 & M0 & XGB \\
\midrule""")
for reg in order:
    pairs = cells[reg]; first = True
    cols = {}
    for rr in RS:
        for key, fn in (("M2", M2f), ("M0", M0f), ("XGB", XGf)):
            c = lines_at(fn, pairs, rr); cols[(rr, key)] = c[int(c.mean(axis=1).argmin())]
    for j, (r, E) in enumerate(pairs):
        tex.append(f"{names[reg] if first else ''} & {r['rep']} & {E:.3f} & " + " & ".join(f"{cols[(rr, key)][j]:.2f}" for rr in RS for key in ("M2", "M0", "XGB")) + r" \\"); first = False
    tex.append(" & mean & " + f"{Ebar[reg]:.3f} & " + " & ".join(f"{cols[(rr, key)].mean():.2f}" for rr in RS for key in ("M2", "M0", "XGB")) + r" \\")
    if reg != "abrupt": tex.append(r"\addlinespace")
tex.append(r"""\bottomrule
\end{tabular}}

""")
open(texpath, "w", encoding="utf8").write("\n".join(tex) + "\n")
print("wrote", texpath)
