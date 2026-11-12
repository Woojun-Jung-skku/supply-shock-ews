"""make_supp_extra.py — Tables S13-S16 of the supplementary material.

Reads the follow-up result files
  FU/B2_alpha_groups.jsonl            (followup/b2_alpha_groups.py)   -> Table S13
  FU/B1_nbcharts_replications_{group,panel}.csv and B1_nbcharts_diag.csv (followup/b1_nbcharts.py, b1_diag.py) -> Table S14
  FU/B5/R/R-sevref__latent.jsonl, R-pareto__latent.jsonl (followup/b5_severity.py) -> Table S15
  FU/B3/R/R-lognormal__latent.jsonl (followup/b3_lognormal.py) with FU/B5/R/R-sevref__latent.jsonl as reference -> Table S16
and writes supplementary_extra.tex (input by supplementary.tex) and the CSV form of each table into tables/.
usage: python make_supp_extra.py FU_DIR [ANALYSIS_DIR]      e.g. FU_DIR = ../results/followup/out, ANALYSIS_DIR = ../results/analysis
Defaults that can be overridden with environment variables: TOOLS (code/tools, for arl.py), REFJSON
(code/followup/ref/reference_numbers.json) and RESA (the Study A reference-cell results used in Table S16).
"""
import csv, json, os, sys
import numpy as np
import pandas as pd

FU = sys.argv[1]
AN = sys.argv[2] if len(sys.argv) > 2 else None
HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "tables")
os.makedirs(TAB, exist_ok=True)
for cand in (os.environ.get("TOOLS"), os.path.join(os.path.dirname(HERE), "code", "tools"),
             os.path.join(os.path.dirname(os.path.dirname(HERE)), "code", "tools")):
    if cand and os.path.isdir(cand):
        sys.path.insert(0, cand)
try:
    import arl as ARL
except ImportError:
    ARL = None
    print("warning: tools/arl.py not found (set TOOLS); pooled ARL0 columns left empty")

out = []


SYM = [("ARL₀", r"ARL$_0$"), ("ARL₁", r"ARL$_1$"), ("γ₀", r"$\gamma_0$"), ("γ₁, γ₂", r"$\gamma_1$, $\gamma_2$"), ("τ_α", r"$\tau_\alpha$"),
       ("m_α", r"$m_\alpha$"), ("α′", r"$\alpha'$"), ("φ", r"$\phi$"), ("β", r"$\beta$"), ("κ_w", r"$\kappa_w$"), ("≤", r"$\le$"), ("≥", r"$\ge$"), ("%", r"\%"), ("&", r"\&")]


def esc(s):
    s = str(s)
    for a, b in SYM:
        s = s.replace(a, b)
    return s


def table(label, caption, header, rows, align=None, size=r"\footnotesize", csvname=None, note=None):
    align = align or "l" + "r" * (len(header) - 1)
    out.append(r"\noindent\begin{minipage}{\linewidth}")
    out.append(r"\captionof{table}{" + caption + r"}\label{tab:" + label + "}")
    out.append(r"\centering" + size)
    out.append(r"\begin{tabular}{" + align + "}")
    out.append(r"\toprule")
    out.append(" & ".join(esc(h) for h in header) + r" \\ \midrule")
    for r in rows:
        if r == "MIDRULE":
            out.append(r"\midrule"); continue
        out.append(" & ".join(esc(c) for c in r) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    if note:
        out.append(r"\par\vspace{3pt}\begin{flushleft}\footnotesize " + note + r"\end{flushleft}")
    out.append(r"\end{minipage}")
    out.append(r"\vspace{22pt}")
    out.append("")
    if csvname:
        with open(os.path.join(TAB, csvname), "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows([r for r in rows if r != "MIDRULE"])


def f(x, d=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


# ------------------------------------------------------------------------------------------ S13: alpha' by group (B2)
p = os.path.join(FU, "B2_alpha_groups.jsonl")
if os.path.exists(p):
    recs = [json.loads(l) for l in open(p)]
    rows = []
    for spread, truth in (("spread2", (1.0, 2.0, 4.0)), ("spread4", (0.5, 2.0, 8.0))):
        for variant, vname in (("hier", "hierarchical (16)"), ("complete", "complete pooling")):
            rr = [x for x in recs if x["variant"] == variant and x["arm"] == spread]
            if not rr:
                continue
            a50 = np.array([np.atleast_1d(x["recovery"]["alpha"]["q50"]) for x in rr]); a05 = np.array([np.atleast_1d(x["recovery"]["alpha"]["q05"]) for x in rr]); a95 = np.array([np.atleast_1d(x["recovery"]["alpha"]["q95"]) for x in rr])
            tr = np.array(truth)
            if a50.shape[1] == 3:
                for g in range(3):
                    cov = ((a05[:, g] <= tr[g]) & (tr[g] <= a95[:, g])).mean()
                    rows.append([f"{vname}, truth ({truth[0]:g}, {truth[1]:g}, {truth[2]:g})" if g == 0 else "", f"group {g + 1}", f"{tr[g]:g}", len(rr), sum(x["failed"] for x in rr),
                                 f(np.exp(np.log(a50[:, g]).mean())), f"{a50[:, g].min():.2f}–{a50[:, g].max():.2f}", f(cov), f(np.mean(np.log(a95[:, g]) - np.log(a05[:, g]))),
                                 f(np.mean([x["recovery"]["tau_alpha"]["q50"] for x in rr]), 3) if g == 0 else "", f(np.mean([x["recovery"]["m_alpha"]["q50"] for x in rr]), 3) if g == 0 else ""])
            else:
                cov = ((a05[:, :1] <= tr[None, :]) & (tr[None, :] <= a95[:, :1])).mean(1)
                rows.append([f"{vname}, truth ({truth[0]:g}, {truth[1]:g}, {truth[2]:g})", "all", f"{np.exp(np.log(tr).mean()):.2f} (geo. mean)", len(rr), sum(x["failed"] for x in rr),
                             f(np.exp(np.log(a50[:, 0]).mean())), f"{a50[:, 0].min():.2f}–{a50[:, 0].max():.2f}", f(cov.mean()), f(np.mean(np.log(a95[:, 0]) - np.log(a05[:, 0]))), "–", "–"])
    table("S13", r"Table S13. Dispersion that differs across groups (Section 5.2 (ii)): the reference cell (gradual, $\bar{\mu}$ = 0.2, I = 30, T = 120) re-drawn with group dispersions $\alpha'_g$ = (1, 2, 4) and (0.5, 2, 8), five replications each, fitted with the hierarchical model and with complete pooling (one $\alpha'$). Posterior median of $\alpha'_g$ (geometric mean over replications and its range), realised coverage of the 90\% interval, mean log-scale width, and the posterior medians of $\tau_\alpha$ and $m_\alpha$ of (16) (hierarchical fits; the truth is sd(log $\alpha'_g$) = 0.57 and 1.13, mean(log $\alpha'_g$) = 0.69). For complete pooling the single $\alpha'$ is compared with each of the three truths.",
          ["Fit, truth", "Group", "Truth", "n", "Failed", "α′ median", "Range", "Cov.", "log width", "τ_α", "m_α"], rows, align="llrrrrrrrrr", size=r"\scriptsize", csvname="S13_alpha_by_group.csv")

# ------------------------------------------------------------------------------------------ S14: NB-designed charts (B1)
rows = []
for pool, pname in (("group", "within group"), ("panel", "across the panel")):
    p = os.path.join(FU, f"B1_nbcharts_replications_{pool}.csv")
    if not os.path.exists(p):
        continue
    d = pd.read_csv(p)
    for key, name in (("nbcusum", "CUSUM, NB design"), ("nbewma", "EWMA, NB design")):
        pr = ARL.pooled(d, f"{key}_arl0", f"{key}_stable_alarm_units", ARL.UNITS["item"]) if ARL else None
        by = {reg: ARL.pooled(d[d.regime == reg], f"{key}_arl0", f"{key}_stable_alarm_units", ARL.UNITS["item"]) for reg in ("stable", "gradual", "abrupt")} if ARL else {}
        rows.append([f"{name}, α̂ pooled {pname}", f(d[f"{key}_dr1"].mean()), f(d[f"{key}_dr3"].mean()), f(d[f"{key}_fa_stable"].mean(), 3),
                     f"{d[f'{key}_lead'].mean():.1f} ({d[f'{key}_lead_q25'].mean():.1f}–{d[f'{key}_lead_q75'].mean():.1f})",
                     f"{pr['est']:.1f} ({pr['lo']:.0f}–{pr['hi']:.0f})" if pr else "–",
                     " / ".join(f"{by[r]['est']:.0f}" for r in ("stable", "gradual", "abrupt")) if by else "–",
                     f(d[f"{key}_arl1"].mean(), 1)])
    if pool == "group":
        for key, name in (("cusum", "Poisson CUSUM (Table 6)"), ("ewma", "Poisson EWMA (Table 6)")):
            pr = ARL.pooled(d, f"{key}_arl0", f"{key}_stable_alarm_units", ARL.UNITS["item"]) if ARL else None
            by = {reg: ARL.pooled(d[d.regime == reg], f"{key}_arl0", f"{key}_stable_alarm_units", ARL.UNITS["item"]) for reg in ("stable", "gradual", "abrupt")} if ARL else {}
            rows.append([name, f(d[f"{key}_dr1"].mean()), f(d[f"{key}_dr3"].mean()), f(d[f"{key}_fa_stable"].mean(), 3),
                         f"{d[f'{key}_lead'].mean():.1f} ({d[f'{key}_lead_q25'].mean():.1f}–{d[f'{key}_lead_q75'].mean():.1f})",
                         f"{pr['est']:.1f} ({pr['lo']:.0f}–{pr['hi']:.0f})" if pr else "–",
                         " / ".join(f"{by[r]['est']:.0f}" for r in ("stable", "gradual", "abrupt")) if by else "–",
                         f(d[f"{key}_arl1"].mean(), 1)])
        dgeo = d.groupby("alpha").agg(a=("alpha_hat_geo", lambda v: float(np.exp(np.log(v).mean()))), p=("alpha_hat_poisson_share", "mean"))
if rows:
    note = (r"Design: phase I = months 13--36 per item, as for the Poisson charts; the dispersion is the pooled moment estimator "
            r"$\hat{\alpha} = \sum_i \hat{\mu}_i^2 / \sum_i (s_i^2 - \hat{\mu}_i)$ over the items of the group (or of the whole panel), Poisson when the pooled excess variance is not positive; "
            r"the CUSUM reference value is the negative-binomial likelihood-ratio value for a 1.5-fold shift, "
            r"$k = \hat{\alpha}\log[(\hat{\alpha} + 1.5\hat{\mu})/(\hat{\alpha} + \hat{\mu})] / \log[1.5(\hat{\alpha} + \hat{\mu})/(\hat{\alpha} + 1.5\hat{\mu})]$; "
            r"the limit is the $(1 - 1/24)$ quantile of the statistic's in-control distribution simulated from the fitted negative binomial (20,000 months), the same nominal run length as the Poisson charts. "
            + (f"Estimated dispersion (geometric mean over items, group pooling): {dgeo.loc[0.5, 'a']:.1f} at $\\alpha'$ = 0.5 and {dgeo.loc[2.0, 'a']:.1f} at $\\alpha'$ = 2, with {100 * dgeo.loc[0.5, 'p']:.0f}\\% and {100 * dgeo.loc[2.0, 'p']:.0f}\\% of the items treated as Poisson." if 'dgeo' in dir() else ""))
    table("S14", r"Table S14. Control charts designed under the negative binomial (Section 5.1, 5.3 (iii)): Study B, pooled over the twelve cells (65 replications), evaluated exactly as the EWMA and CUSUM rows of Table 6. Detection within one and three months, false alarms per item-year in the stable windows, median lead time (mean of the replication quartiles), pooled ARL$_0$ with its 90\% bootstrap interval and by regime (stable / gradual / abrupt cells), and ARL$_1$.",
          ["Chart", "Det. ≤ 1 mo", "Det. ≤ 3 mo", "FA / item-year", "Lead (IQR)", "ARL₀ (90%)", "ARL₀ by regime", "ARL₁"], rows, align="lrrrrrrr", size=r"\scriptsize", csvname="S14_nb_charts.csv", note=note)
    p = os.path.join(FU, "B1_nbcharts_diag.csv")
    if os.path.exists(p):
        dd = pd.read_csv(p)
        rows = [[r["variant"], r["ewma"], r["cusum"]] for _, r in dd.iterrows()]
        table("S14b", r"Table S14 (continued). Why the negative-binomial design does not restore the EWMA's run length: realised ARL$_0$ (90\% interval) of the negative-binomial-designed charts in the four stable cells of Study B (20 replications) when the in-control parameters are estimated from phase I as in Table S14, when the true time-averaged in-control mean of each item is used instead (oracle mean), when the true dispersion is used as well (oracle mean and dispersion), and when the data are generated without indicator effects ($\beta$ = 0) and the parameters estimated as usual.",
              ["In-control model", "EWMA ARL₀", "CUSUM ARL₀"], rows, align="lrr", csvname="S14b_nb_charts_diagnostic.csv")

# ------------------------------------------------------------------------------------------ S15: severity submodel (B5)
rows = []
for cid, name in (("R-sevref", "log-normal (reference)"), ("R-pareto", "Pareto")):
    p = os.path.join(FU, "B5", "R", f"{cid}__latent.jsonl")
    if not os.path.exists(p):
        continue
    rr = [json.loads(l) for l in open(p)]
    sv = [x["M2"]["severity"] for x in rr if x.get("M2") and x["M2"].get("severity")]
    if not sv:
        continue
    g50 = np.array([s["gamma_q50"] for s in sv]); g05 = np.array([s["gamma_q05"] for s in sv]); g95 = np.array([s["gamma_q95"] for s in sv])
    es = np.mean([s["S_mean_obs"] for s in sv])
    g0_truth = np.log(es / (1 - es))
    cov0 = ((g05[:, 0] <= g0_truth) & (g0_truth <= g95[:, 0])).mean()
    cov_rest = ((g05[:, 1:] <= 0) & (0 <= g95[:, 1:])).mean()
    rows.append([name, len(sv), f(es, 3), f"{g50[:, 0].mean():.2f} ({g0_truth:.2f}; {cov0:.2f})",
                 f"{np.abs(g50[:, 1:]).max():.2f} ({cov_rest:.2f})", f(np.mean([s["phi_q50"] for s in sv]), 1),
                 f(np.mean([s["n_losses"] for s in sv]), 0), f(np.mean([s["raw_cover"] for s in sv]), 3), f(np.mean([s["raw_width"] for s in sv]), 2),
                 f(np.mean([s["cqr_cover"] for s in sv]), 3), f(np.mean([s["cqr_width"] for s in sv]), 2),
                 f"{np.mean([x['M2']['severe']['severe_tail'] / max(x['M2']['severe']['level3_months'], 1) for x in rr]):.2f}"])
if rows:
    table("S15", r"Table S15. The severity submodel (S8.2)--(S8.3) of Supplementary Note 8 evaluated (Sections 4.4 and 5.4): the robustness protocol on the reference design with log-normal and with Pareto severity, five replications each. In the data-generating process the loss fraction is drawn independently of the intensity and the indicators, so $\gamma_1 = \gamma_2 = 0$ and logit$^{-1}(\gamma_0)$ = E[S]. Columns: mean realised loss fraction; posterior median of $\gamma_0$ at the last origin against its implied truth and the coverage of its 90\% interval; largest absolute posterior median among $\gamma_1$ and the indicator coefficients $\gamma_2$, with the share of their 90\% intervals covering 0; posterior median of $\phi$; number of losses realised in the decision months; coverage and mean width of the 5--95\% posterior predictive interval of the loss fraction (raw) and after conformalised quantile regression on the earlier losses (CQR, target 0.90); share of level-3 months in which the tail-adjusted expected loss confirms the severe level.",
          ["Severity", "n", "mean S", "γ₀ (truth; cov.)", "max |γ₁, γ₂| (cov. 0)", "φ", "losses", "raw cov.", "width", "CQR cov.", "width", "severe conf."], rows, align="lrrrrrrrrrrr", size=r"\scriptsize", csvname="S15_severity.csv")


# ------------------------------------------------------------------------------------------ S16: log-normal heterogeneity (B3)
def _rob_rows(rr):
    """Recovery and alarm summaries of a set of robustness-protocol records (one per replication)."""
    a50 = np.array([x["recovery"]["alpha"]["q50"] for x in rr]); a05 = np.array([x["recovery"]["alpha"]["q05"] for x in rr]); a95 = np.array([x["recovery"]["alpha"]["q95"] for x in rr])
    b50 = np.array([x["recovery"]["beta"]["q50"] for x in rr]); b05 = np.array([x["recovery"]["beta"]["q05"] for x in rr]); b95 = np.array([x["recovery"]["beta"]["q95"] for x in rr])
    w50 = np.array([x["recovery"]["w"]["q50"] for x in rr]).mean(0); k50 = np.mean([x["recovery"]["kappa_w"]["q50"] for x in rr])
    k05 = np.array([x["recovery"]["kappa_w"]["q05"] for x in rr]); k95 = np.array([x["recovery"]["kappa_w"]["q95"] for x in rr])
    btr = np.array([0.4, 0.3, 0.2])
    d = {}
    d["alpha"] = f"{np.exp(np.log(a50).mean()):.2f} ({np.exp(np.log(a50).mean(1)).min():.2f}–{np.exp(np.log(a50).mean(1)).max():.2f})"
    d["alpha_cov"] = f"{((a05 <= 2.0) & (2.0 <= a95)).mean():.2f}"
    d["alpha_width"] = f"{np.log(a95 / a05).mean():.2f}"
    d["logC"] = f"{np.mean([x['recovery']['logC_summary']['rmse'] for x in rr]):.2f} / {np.mean([x['recovery']['logC_summary']['cover'] for x in rr]):.2f}"
    d["w"] = f"({w50[0]:.2f}, {w50[1]:.2f})"
    d["kappa"] = f"{k50:.2f} ({((k05 <= 0.8) & (0.8 <= k95)).mean():.2f})"
    d["beta"] = f"({b50.mean(0)[0]:.3f}, {b50.mean(0)[1]:.3f}, {b50.mean(0)[2]:.3f})"
    d["beta_cov"] = f"{((b05 <= btr) & (btr <= b95)).mean():.2f}"
    d["failed"] = f"{sum(x['recovery_fit']['failed'] for x in rr)} of {len(rr)}"
    m = [x["M2"] for x in rr]
    def ms(vals, dd=3):
        v = np.array([float(u) for u in vals if u is not None and np.isfinite(float(u))])
        if v.size == 0: return "–"
        return f"{v.mean():.{dd}f}" + (f" ({v.std(ddof=1) / np.sqrt(v.size):.{dd}f})" if v.size > 1 else "")
    d["aucpr"] = ms([u["prob"]["aucpr"] for u in m]); d["brier"] = ms([u["prob"]["brier"] for u in m]); d["ece"] = ms([u["prob"]["ece"] for u in m])
    d["share"] = ms([u["prob"]["base_rate"] for u in m])
    t2 = [u["triage"]["level2"] for u in m]
    d["dr1"] = ms([u["dr_h1"] for u in t2], 2); d["dr3"] = ms([u["dr_h3"] for u in t2], 2); d["fa"] = ms([u["fa_per_itemyear_stable"] for u in t2], 2); d["lead"] = ms([u["lead_med"] for u in t2], 1)
    d["conf"] = f"{np.mean([u['conformal']['conf_cover'] for u in m]):.3f} / {np.mean([u['conformal']['conf_size'] for u in m]):.2f}"
    if ARL is not None:
        df = pd.DataFrame([{"q_arl0": u["q"]["arl0"], "q_units": u["q"].get("stable_alarm_units"), "regime": "gradual"} for u in m])
        pr = ARL.pooled(df, "q_arl0", "q_units", ARL.UNITS["q"])
        d["q_arl0"] = f"{pr['est']:.0f} ({pr['lo']:.0f}–{pr['hi']:.0f})"
    else:
        d["q_arl0"] = "–"
    d["q_arl1"] = ms([u["q"].get("arl1") for u in m], 1)
    d["origins_failed"] = f"{sum(int(u['fit']['n_failed']) for u in m)} of {sum(int(u['fit'].get('n_origins', 14)) if isinstance(u['fit'], dict) else 14 for u in m)}"
    for ch in ("ewma", "cusum"):
        c = [x["charts"][ch] for x in rr]
        if ARL is not None:
            df = pd.DataFrame([{"a": u["run_length"]["arl0"], "u": u["run_length"].get("stable_alarm_units"), "regime": "gradual"} for u in c])
            pr = ARL.pooled(df, "a", "u", ARL.UNITS["item"]); arl0 = f"{pr['est']:.1f}"
        else:
            arl0 = "–"
        d[ch] = f"{np.mean([u['events']['dr_h1'] for u in c]):.2f} / {np.mean([u['events']['fa_per_itemyear_stable'] for u in c]):.2f} / {arl0}"
    return d

p3 = os.path.join(FU, "B3", "R", "R-lognormal__latent.jsonl"); pref = os.path.join(FU, "B5", "R", "R-sevref__latent.jsonl")
if os.path.exists(p3) and os.path.exists(pref):
    r3 = [json.loads(l) for l in open(p3)]; r3 = [x for x in r3 if x.get("M2")]
    rf = [json.loads(l) for l in open(pref)]; rf = [x for x in rf if x.get("M2")]
    L, R = _rob_rows(r3), _rob_rows(rf)
    # archived Study B reference cell (ten replications) and Study A reference-cell fits 0-4, as in Table S8
    A = {k: "–" for k in L}
    REPO = os.path.dirname(HERE)
    refjson = os.environ.get("REFJSON", os.path.join(REPO, "code", "followup", "ref", "reference_numbers.json"))
    resA = os.environ.get("RESA", os.path.join(REPO, "results", "A", "A-a2-mu0.2-gradual-I30-T120__hier.jsonl"))
    if os.path.exists(refjson):
        B = json.load(open(refjson))["B_reference_cell"]
        def mse_(v, dd=3): return f"{v[0]:.{dd}f} ({v[1]:.{dd}f})"
        A.update(aucpr=mse_(B["M2_aucpr"]), brier=mse_(B["M2_brier"]), ece=mse_(B["M2_ece"]), share=mse_(B["M2_base_rate"]),
                 dr1=mse_(B["M2_tri2_dr1"], 2), dr3=mse_(B["M2_tri2_dr3"], 2), fa=mse_(B["M2_tri2_fa_stable"], 2), lead=mse_(B["M2_tri2_lead"], 1),
                 q_arl0=f"{B['M2_q_arl0_pooled'][0]:.0f} ({B['M2_q_arl0_pooled'][1]:.0f}–{B['M2_q_arl0_pooled'][2]:.0f})", q_arl1=mse_(B["M2_q_arl1"], 1),
                 ewma=f"{B['ewma_dr1'][0]:.2f} / {B['ewma_fa_stable'][0]:.2f} / {B['ewma_arl0_pooled'][0]:.1f}",
                 cusum=f"{B['cusum_dr1'][0]:.2f} / {B['cusum_fa_stable'][0]:.2f} / {B['cusum_arl0_pooled'][0]:.1f}")
    if AN and os.path.exists(os.path.join(AN, "B_replications.csv")):
        bb = pd.read_csv(os.path.join(AN, "B_replications.csv")); bb = bb[bb.cell == "B-a2-mu0.2-gradual"]
        if len(bb) and "M2_conf_cover" in bb:
            A["conf"] = f"{bb.M2_conf_cover.mean():.3f} / {bb.M2_conf_size.mean():.2f}"
    if os.path.exists(resA):
        ra = [json.loads(l) for l in open(resA)]; ra = [x for x in ra if x["rep"] < 5]
        a50 = np.array([x["recovery"]["alpha"]["q50"] for x in ra]); a05 = np.array([x["recovery"]["alpha"]["q05"] for x in ra]); a95 = np.array([x["recovery"]["alpha"]["q95"] for x in ra])
        b50 = np.array([x["recovery"]["beta"]["q50"] for x in ra]); b05 = np.array([x["recovery"]["beta"]["q05"] for x in ra]); b95 = np.array([x["recovery"]["beta"]["q95"] for x in ra])
        k05 = np.array([x["recovery"]["kappa_w"]["q05"] for x in ra]); k95 = np.array([x["recovery"]["kappa_w"]["q95"] for x in ra])
        w50 = np.array([x["recovery"]["w"]["q50"] for x in ra]).mean(0); btr = np.array([0.4, 0.3, 0.2])
        A.update(alpha=f"{np.exp(np.log(a50).mean()):.2f} ({np.exp(np.log(a50).mean(1)).min():.2f}–{np.exp(np.log(a50).mean(1)).max():.2f})",
                 alpha_cov=f"{((a05 <= 2.0) & (2.0 <= a95)).mean():.2f}", alpha_width=f"{np.log(a95 / a05).mean():.2f}",
                 logC=f"{np.mean([x['recovery']['logC_summary']['rmse'] for x in ra]):.2f} / {np.mean([x['recovery']['logC_summary']['cover'] for x in ra]):.2f}",
                 w=f"({w50[0]:.2f}, {w50[1]:.2f})", kappa=f"{np.mean([x['recovery']['kappa_w']['q50'] for x in ra]):.2f} ({((k05 <= 0.8) & (0.8 <= k95)).mean():.2f})",
                 beta=f"({b50.mean(0)[0]:.3f}, {b50.mean(0)[1]:.3f}, {b50.mean(0)[2]:.3f})", beta_cov=f"{((b05 <= btr) & (btr <= b95)).mean():.2f}",
                 failed=f"{sum((x.get('fit') or x.get('recovery_fit') or {}).get('failed', 0) for x in ra)} of {len(ra)}")
    rows = [["α′ posterior median, geometric mean over groups and replications (range over replications); truth 2 = the variance-matched value", L["alpha"], R["alpha"], A["alpha"]],
            ["90% intervals of α′ covering 2 (share over groups and replications)", L["alpha_cov"], R["alpha_cov"], A["alpha_cov"]],
            ["Mean width of the 90% interval of log α′", L["alpha_width"], R["alpha_width"], A["alpha_width"]],
            ["log C: RMSE / coverage of the 90% intervals", L["logC"], R["logC"], A["logC"]],
            ["w posterior median (truth (0.5, 0.3))", L["w"], R["w"], A["w"]],
            ["κ_w posterior median (coverage of 0.8)", L["kappa"], R["kappa"], A["kappa"]],
            ["β posterior median (truth (0.4, 0.3, 0.2))", L["beta"], R["beta"], A["beta"]],
            ["90% intervals of β covering the truth", L["beta_cov"], R["beta_cov"], A["beta_cov"]],
            ["Recovery fits failed", L["failed"], R["failed"], A["failed"]],
            "MIDRULE",
            ["AUC-PR", L["aucpr"], R["aucpr"], A["aucpr"]], ["Brier score", L["brier"], R["brier"], A["brier"]], ["ECE", L["ece"], R["ece"], A["ece"]], ["Event share", L["share"], R["share"], A["share"]],
            ["Detected ≤ 1 month (level 2)", L["dr1"], R["dr1"], A["dr1"]], ["Detected ≤ 3 months", L["dr3"], R["dr3"], A["dr3"]],
            ["False alarms per item-year (stable windows)", L["fa"], R["fa"], A["fa"]], ["Lead time (median, months)", L["lead"], R["lead"], A["lead"]],
            ["Conformal coverage / set size", L["conf"], R["conf"], A["conf"]],
            ["ARL₀ of the q channel (pooled, 90% interval)", L["q_arl0"], R["q_arl0"], A["q_arl0"]], ["ARL₁ of the q channel", L["q_arl1"], R["q_arl1"], A["q_arl1"]],
            "MIDRULE",
            ["EWMA: detected ≤ 1 month / false alarms / ARL₀", L["ewma"], R["ewma"], A["ewma"]],
            ["CUSUM: detected ≤ 1 month / false alarms / ARL₀", L["cusum"], R["cusum"], A["cusum"]]]
    table("S16", r"Table S16. Heterogeneity that is not gamma (Sections 3.3 and 5.4): the robustness protocol (recovery fit, M2 alarm pipeline, EWMA and CUSUM) on the reference design with the gamma multiplier of (2) replaced, in the truth only, by a log-normal multiplier of the same mean (1) and variance (1/$\alpha'$ = 0.5), five replications. The fitted model is the gamma--Poisson hierarchy of Section 4 in every column, so the recovery of $\alpha'$ is judged against the variance-matched value 2. Second column: the same protocol with the gamma multiplier, run with the same code and seeds convention (the reference arm of Table S15). Third column: the archived Study B reference cell (ten replications) and, for the recovery rows, the Study A hierarchical fits of replications 0--4 of the reference cell, as in Table S8. Means over replications with Monte Carlo standard errors in brackets.",
          ["Quantity", "Log-normal (5 rep.)", "Gamma, same protocol (5 rep.)", "Study B reference (10 rep.)"], rows, align="p{0.34\\linewidth}lll", size=r"\scriptsize", csvname="S16_lognormal.csv",
          note=r"ARL$_0$ of the $q$ channel: no persistent alarm occurred in the stable windows of the five log-normal replications, so the pooled value is the stable unit-months (360) with the alarm count floored at one, as elsewhere.")

open(os.path.join(HERE, "supplementary_extra.tex"), "w").write("\n".join(out) + "\n")
print("wrote supplementary_extra.tex with", sum(1 for l in out if l.startswith(r"\captionof")), "tables")
