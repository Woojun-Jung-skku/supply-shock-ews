"""cost_curves.py — expected cost of the alarm policies of Study B (Section 5.6; Table 8, Fig. 7, Supplementary
Note 10). No new MCMC: replication-level results of Study B (results/B/*__all.jsonl and the follow-up
baselines) plus the simulated panels regenerated exactly from the archived seeds.

Two cost models, both computed exactly from stored quantities
-------------------------------------------------------------
Per method m, operating level k (triage levels 1-3 for the probability methods; limits 1-3 for the
sigma-rule; one point for EWMA, CUSUM, NB-CUSUM and the Farrington-type threshold), replication j:
  AM_mkj   alarm months per item-year = 12 * alarm_share (share of the 82 x 30 decision item-months,
           t = 36..117, in which level >= k is in force; metrics.event_metrics 'alarm_share')
  ON_mkj   alarm onsets per item-year (n_onsets / 205 item-years); FA_mkj false onsets per item-year
           (onsets with no shock in the next three months, whole window 'fa_per_itemyear'; stable
           windows 'fa_per_itemyear_stable' = the Table 6 quantity)
  DR_mkj   one-month detection rate 'dr_h1': share of realised shock item-months (i, tau), tau = 37..118,
           with the alarm in force in decision month tau - 1 (dr_h3: in some of tau-3..tau-1)
  E_j      realised shock item-months per item-year over tau = 37..118 (30 items x 82 months),
           regenerated from the archived seed (dgp.simulate, dgp.seed_of('B', cell, rep); the
           regeneration is validated in validate_regen.py against the archived base rates and chart
           metrics of all 65 replications)
Model A — posture cost (headline):   EC_A(r) = AM + r (1 - DR) E
   the action behind a level (monitoring, second-source work, safety stock, contingency) is sustained
   while the alarm is on and costs 1 per item-month; a shock that arrives with the alarm in force the
   month before is mitigated, one that arrives without it costs r (the avoidable loss, in units of one
   alarm-month). Reference policies: never act (r E) and always act (12).
Model B — classification cost (supplement; the framing of eq. (19) and of Table 6):
   EC_B(r) = FA + r (1 - DR) E, with the false-alarm cost charged once per false onset. Under this
   model an alarm that never switches off has almost no onsets, so the always-on policy dominates;
   the model is reported to tie the section to Table 6, not as the basis of the conclusions.
Curves over r on a log grid: for a method with several levels the LOWER ENVELOPE of its level-wise
mean-cost lines (the best of its fixed thresholds at that r); one line for a one-point rule.
Statistics: means over the 65 replications (as Table 6: the reference cell has 10), Monte Carlo
standard errors, paired differences and sign tests at the envelope level of each method, savings
1 - EC_M2 / EC_m on the means, value of monitoring 1 - EC_M2 / min(never, always), break-even r*.

usage: python -I cost_curves.py CODE_DIR ANALYSIS_DIR JSONL_EXTRACT_CSV OUT_DIR [FIGURES_DIR]
  FIGURES_DIR  the repository's figures/ folder (style module figstyle.py); if given, the figure of the cost curves is
               written to FIGURES_DIR/out/Fig5_cost_curves.pdf and .png (Fig. 5 of the manuscript; Fig. 7 in the
               numbering of this header, see the README of the repository)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

CODE, AN, JX, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
STYLE = sys.argv[5] if len(sys.argv) > 5 else None
sys.path.insert(0, CODE)
import config as C
C.apply_profile("t4_min")
import dgp

os.makedirs(OUT, exist_ok=True)
MONTHS = np.arange(C.ORIGINS[0], C.DECISION_END + 1)          # decision months 36..117 (0-based)
S0, S1 = MONTHS[0], MONTHS[-1]
I_, T_ = C.REF["I"], C.REF["T"]
ITEM_YEARS = I_ * len(MONTHS) / 12.0                           # 205
R_GRID = np.exp(np.linspace(np.log(1.0), np.log(100.0), 600))
R_REPORT = (2.0, 5.0, 10.0, 20.0, 50.0)

PROB = [("M2", "M2 (proposed)"), ("NBcomplete", "NB complete pooling"), ("NBnone", "NB no pooling"),
        ("M0", "Poisson M0"), ("xgb", "XGBoost"), ("lstm", "LSTM"), ("tft", "TFT")]
ONE = [("ewma", "EWMA"), ("cusum", "CUSUM"), ("nbcusum", "CUSUM, negative-binomial design"),
       ("farr", "Farrington-type threshold")]
LABEL = dict(PROB + ONE + [("sigma", "Sigma-rule (plug-in SRI)"), ("never", "Never act"), ("always", "Always act"),
                           ("head", "M2 headline (triage or q-alarm)")])
METHODS = ["M2", "NBcomplete", "NBnone", "M0", "xgb", "lstm", "tft", "sigma", "ewma", "cusum", "nbcusum", "farr"]
REFS = ["never", "always"]


def levels_of(m):
    return (1, 2, 3) if m in dict(PROB) or m == "sigma" else ((1, 2) if m == "head" else (None,))


def key(m, k):
    if m in dict(PROB):
        return f"{m}_tri{k}"
    if m == "sigma":
        return f"sigma_{k}"
    if m == "head":
        return f"head{k}"
    return m


# ---------------------------------------------------------------- data
J = pd.read_csv(JX)                                            # per-method/level fields from the JSONL


def _find(name):
    """follow-up CSVs live in results/followup/out/ in the repository and next to B_replications.csv in the package."""
    for d in (AN, os.path.join(AN, "..", "followup", "out"), os.path.join(AN, "followup", "out")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


F = pd.read_csv(_find("A3_farrington_replications.csv"))
NB = pd.read_csv(_find("B1_nbcharts_replications_group.csv"))
B = pd.read_csv(os.path.join(AN, "B_replications.csv"))[["cell", "rep", "alpha", "mu", "regime"]]
D = B.merge(J, on=["cell", "rep"]).merge(
    F[["cell", "rep", "farr_dr1", "farr_dr3", "farr_fa", "farr_fa_stable", "farr_alarm_share", "farr_n_onsets"]], on=["cell", "rep"]).merge(
    NB[["cell", "rep", "nbcusum_dr1", "nbcusum_dr3", "nbcusum_fa", "nbcusum_fa_stable", "nbcusum_alarm_share", "nbcusum_n_onsets"]], on=["cell", "rep"])
assert len(D) == 65
# unify field names: <key>_dr1, _dr3, _fa, _fa_stable, _am (alarm months / item-year), _on (onsets / item-year)
for m in METHODS + ["head"]:
    for k in levels_of(m):
        kk = key(m, k)
        if m in ("farr", "nbcusum"):
            D[f"{kk}_am"] = 12.0 * D[f"{kk}_alarm_share"]
            D[f"{kk}_on"] = D[f"{kk}_n_onsets"] / ITEM_YEARS
        else:
            D[f"{kk}_dr1"], D[f"{kk}_dr3"] = D[f"{kk}_dr_h1"], D[f"{kk}_dr_h3"]
            D[f"{kk}_fa"], D[f"{kk}_fa_stable"] = D[f"{kk}_fa_per_itemyear"], D[f"{kk}_fa_per_itemyear_stable"]
            D[f"{kk}_am"] = 12.0 * D[f"{kk}_alarm_share"]
            D[f"{kk}_on"] = D[f"{kk}_n_onsets"] / ITEM_YEARS


def shocks(cell_id, rep, h):
    """(count, per item-year) of realised shock item-months over the detection window of dr_h."""
    parts = cell_id.split("-")
    a, mu, reg = parts[1][1:], parts[2][2:], parts[3]
    I_cell = int(parts[4][1:]) if len(parts) > 4 and parts[4].startswith("I") else I_          # cells with their own I (…-I100)
    s = dgp.simulate(I_cell, T_, float(mu), float(a), reg, seed=dgp.seed_of("B", cell_id, int(rep)))
    N, mk = s["N"], s["Nmask"]
    lo, hi = S0 + h - 1, S1 + 1
    tot = int(((N[:, lo:hi] >= 1) & mk[:, lo:hi]).sum())
    return tot, tot / (I_cell * (hi - lo) / 12.0)


for h in (1, 3):
    v = [shocks(c, r, h) for c, r in zip(D.cell, D.rep)]
    D[f"n_shocks_h{h}"], D[f"E_h{h}"] = [x[0] for x in v], [x[1] for x in v]


# ---------------------------------------------------------------- cost functions
def ec(d, m, k, r, model="A", dr="dr1", fa="fa"):
    E = d["E_h1" if dr == "dr1" else "E_h3"].values
    if m == "never":
        return r * E
    if m == "always":
        return np.full(len(d), 12.0) if model == "A" else np.zeros(len(d))
    kk = key(m, k)
    miss = r * (1.0 - d[f"{kk}_{dr}"].values) * E
    act = d[f"{kk}_am"].values if model == "A" else d[f"{kk}_{fa}"].values
    return act + miss


def curve(d, m, model="A", rgrid=R_GRID, **kw):
    """mean-cost curve = lower envelope over the method's levels; also the level chosen at each r."""
    ks = levels_of(m) if m not in REFS else (None,)
    M = np.array([[ec(d, m, k, r, model, **kw).mean() for r in rgrid] for k in ks])
    pick = np.array(ks)[M.argmin(0)] if ks[0] is not None else None
    return M.min(0), pick, M


def best_level(d, m, r, model="A", **kw):
    ks = levels_of(m) if m not in REFS else (None,)
    vals = [ec(d, m, k, r, model, **kw).mean() for k in ks]
    return ks[int(np.argmin(vals))]


def crossings(a, b, rgrid=R_GRID):
    f = a - b
    out = []
    for i in range(1, len(f)):
        if f[i - 1] * f[i] < 0:
            w = f[i - 1] / (f[i - 1] - f[i])
            out.append(float(np.exp(np.log(rgrid[i - 1]) + w * (np.log(rgrid[i]) - np.log(rgrid[i - 1])))))
    return out


SUBSETS = {"All cells": D, "mu 0.05": D[D.mu == 0.05], "mu 0.2": D[D.mu == 0.2],
           "stable": D[D.regime == "stable"], "gradual": D[D.regime == "gradual"], "abrupt": D[D.regime == "abrupt"],
           "alpha 0.5": D[D.alpha == 0.5], "alpha 2": D[D.alpha == 2]}
CELLS = sorted(D.cell.unique())
for c in CELLS:
    SUBSETS[c] = D[D.cell == c]

out_tables = {}
digest = []
P = digest.append
P("Cost digest — expected cost per item-year, Study B replications (n = 65)")
P(f"shocks per item-year E (tau = 37..118): all {D.E_h1.mean():.3f}; mu 0.05 {D[D.mu == 0.05].E_h1.mean():.3f}; mu 0.2 {D[D.mu == 0.2].E_h1.mean():.3f}; "
  f"by regime stable {D[D.regime == 'stable'].E_h1.mean():.3f}, gradual {D[D.regime == 'gradual'].E_h1.mean():.3f}, abrupt {D[D.regime == 'abrupt'].E_h1.mean():.3f}")
P("alarm months per item-year (12 x alarm share) and false onsets per item-year (whole window / stable), all cells:")
for m in METHODS + ["head"]:
    for k in levels_of(m):
        kk = key(m, k)
        P(f"  {kk:14s} DR1 {D[kk + '_dr1'].mean():.3f}  alarm-months/yr {D[kk + '_am'].mean():5.2f}  onsets/yr {D[kk + '_on'].mean():.3f}  "
          f"false onsets/yr {D[kk + '_fa'].mean():.3f} / {D[kk + '_fa_stable'].mean():.3f}")

for model in ("A", "B"):
    P(f"\n######## Model {model} — {'posture cost per alarm-month' if model == 'A' else 'classification cost per false onset (eq. (19) framing)'} ########")
    rows, sav, be, lev = [], [], [], []
    curves = {}
    for sub, d in SUBSETS.items():
        cur = {m: curve(d, m, model) for m in METHODS + REFS}
        curves[sub] = cur
        triv = np.minimum(cur["never"][0], cur["always"][0])
        for m in METHODS + REFS:
            for r in R_REPORT:
                k = best_level(d, m, r, model)
                mu_ = ec(d, m, k, r, model)
                rows.append(dict(model=model, subset=sub, n=len(d), method=m, label=LABEL[m], r=r, level=k, EC=mu_.mean(),
                                 SE=mu_.std(ddof=1) / np.sqrt(len(mu_)),
                                 **({f"EC_level{kk}": ec(d, m, kk, r, model).mean() for kk in levels_of(m)} if m not in REFS and levels_of(m)[0] else {})))
        for r in R_REPORT:
            k2 = best_level(d, "M2", r, model)
            base = ec(d, "M2", k2, r, model)
            tv = np.minimum(ec(d, "never", None, r, model), ec(d, "always", None, r, model))
            for m in METHODS + REFS:
                if m == "M2":
                    continue
                k = best_level(d, m, r, model)
                other = ec(d, m, k, r, model)
                diff = base - other
                nz = diff[np.abs(diff) > 1e-12]
                wins = int((nz < 0).sum())
                sav.append(dict(model=model, subset=sub, n=len(d), r=r, method=m, label=LABEL[m], M2_level=k2, other_level=k,
                                EC_M2=base.mean(), EC_other=other.mean(), saving=1 - base.mean() / other.mean(),
                                mean_diff=diff.mean(), se_diff=diff.std(ddof=1) / np.sqrt(len(diff)),
                                M2_cheaper_share=wins / max(nz.size, 1), sign_p=(binomtest(wins, nz.size, 0.5).pvalue if nz.size else np.nan)))
            sav.append(dict(model=model, subset=sub, n=len(d), r=r, method="trivial", label="min(never, always)", M2_level=k2, other_level=None,
                            EC_M2=base.mean(), EC_other=tv.mean(), saving=1 - base.mean() / tv.mean(), mean_diff=(base - tv).mean(),
                            se_diff=(base - tv).std(ddof=1) / np.sqrt(len(d)), M2_cheaper_share=float((base < tv).mean()), sign_p=np.nan))
        # break-even ratios: M2 envelope vs each method, and vs the trivial policies
        cm2 = cur["M2"][0]
        for m in METHODS + REFS:
            if m == "M2":
                continue
            xs = crossings(cm2, cur[m][0])
            be.append(dict(model=model, subset=sub, n=len(d), method=m, label=LABEL[m], r_star=";".join(f"{x:.1f}" for x in xs) or "none",
                           M2_cheaper_at_r1=bool(cm2[0] < cur[m][0][0]), M2_cheaper_at_r100=bool(cm2[-1] < cur[m][0][-1])))
        xs = crossings(cm2, triv)
        be.append(dict(model=model, subset=sub, n=len(d), method="trivial", label="min(never, always)", r_star=";".join(f"{x:.1f}" for x in xs) or "none",
                       M2_cheaper_at_r1=bool(cm2[0] < triv[0]), M2_cheaper_at_r100=bool(cm2[-1] < triv[-1])))
        # level switch points of M2 and the value of monitoring
        pick = cur["M2"][1]
        sw = [(float(R_GRID[i]), int(pick[i - 1]), int(pick[i])) for i in range(1, len(pick)) if pick[i] != pick[i - 1]]
        vom = 1 - cm2 / triv
        i_best = int(np.argmax(vom))
        lev.append(dict(model=model, subset=sub, n=len(d), level_at_r1=int(pick[0]), switches=";".join(f"{a:.1f}:{b}->{c}" for a, b, c in sw),
                        level_at_r100=int(pick[-1]), max_value_of_monitoring=float(vom[i_best]), r_at_max=float(R_GRID[i_best]),
                        r_range_vom_above_10pct=(f"{R_GRID[vom > 0.10][0]:.1f}-{R_GRID[vom > 0.10][-1]:.1f}" if (vom > 0.10).any() else "none")))
    out_tables[f"table_{model}"] = pd.DataFrame(rows)
    out_tables[f"savings_{model}"] = pd.DataFrame(sav)
    out_tables[f"breakeven_{model}"] = pd.DataFrame(be)
    out_tables[f"levels_{model}"] = pd.DataFrame(lev)
    out_tables[f"curves_{model}"] = curves

    T1, T2, T3, T4 = (out_tables[f"{t}_{model}"] for t in ("table", "savings", "breakeven", "levels"))
    for sub in ("All cells", "mu 0.05", "mu 0.2", "stable", "gradual", "abrupt"):
        d = SUBSETS[sub]
        P(f"\n== {sub} (n = {len(d)}; E = {d.E_h1.mean():.3f}) ==   EC at the method's best level (level in brackets)")
        P(f"{'method':34s}" + "".join(f"{'r=' + str(int(r)):>16s}" for r in R_REPORT))
        for m in METHODS + REFS:
            t = T1[(T1.subset == sub) & (T1.method == m)].set_index("r")
            P(f"{LABEL[m]:34s}" + "".join(f"{t.loc[r, 'EC']:9.2f} [{t.loc[r, 'level'] if pd.notna(t.loc[r, 'level']) else '-'}]    " for r in R_REPORT))
        P("-- M2 saving vs method (1 - EC_M2/EC_m) | share of replications with M2 cheaper | sign p --")
        for m in METHODS + REFS + ["trivial"]:
            if m == "M2":
                continue
            t = T2[(T2.subset == sub) & (T2.method == m)].set_index("r")
            lab = LABEL.get(m, "min(never, always)")
            P(f"{lab:34s}" + "".join(f"{100 * t.loc[r, 'saving']:7.1f}% {100 * t.loc[r, 'M2_cheaper_share']:3.0f}% " + (f"p={t.loc[r, 'sign_p']:.2f} " if pd.notna(t.loc[r, 'sign_p']) else "       ") for r in R_REPORT))
        P("-- break-even r* (M2 envelope vs method envelope; grid 1..100) --")
        for m in METHODS + REFS + ["trivial"]:
            if m == "M2":
                continue
            t = T3[(T3.subset == sub) & (T3.method == m)].iloc[0]
            P(f"{LABEL.get(m, 'min(never, always)'):34s} r* = {t.r_star:18s} M2 cheaper at r=1: {t.M2_cheaper_at_r1!s:5s} at r=100: {t.M2_cheaper_at_r100}")
        t = T4[T4.subset == sub].iloc[0]
        P(f"-- M2 level by r: {t.level_at_r1} at r=1; switches {t.switches or 'none'}; {t.level_at_r100} at r=100.  "
          f"Value of monitoring 1 - EC_M2/min(never, always): max {100 * t.max_value_of_monitoring:.1f}% at r = {t.r_at_max:.1f}; above 10% for r in {t.r_range_vom_above_10pct}")
    P("\n-- per cell: cheapest method at r = 5, 10, 20 (best level each) and M2 value of monitoring --")
    for c in CELLS:
        d = SUBSETS[c]
        parts = []
        for r in (5.0, 10.0, 20.0):
            vals = {m: ec(d, m, best_level(d, m, r, model), r, model).mean() for m in METHODS}
            bm = min(vals, key=vals.get)
            triv = np.minimum(ec(d, "never", None, r, model), ec(d, "always", None, r, model)).mean()
            parts.append(f"r={int(r)}: {LABEL[bm].split(' (')[0]} {vals[bm]:.2f} (M2 {vals['M2']:.2f}, trivial {triv:.2f})")
        P(f"  {c:24s} E={d.E_h1.mean():.2f}  " + " | ".join(parts))
    # per-cell table to CSV
    pc = []
    for c in CELLS:
        d = SUBSETS[c]
        for r in R_REPORT:
            rec = dict(model=model, cell=c, alpha=d.alpha.iloc[0], mu=d.mu.iloc[0], regime=d.regime.iloc[0], n=len(d), r=r, E=d.E_h1.mean())
            for m in METHODS + REFS:
                k = best_level(d, m, r, model)
                rec[m] = ec(d, m, k, r, model).mean()
                rec[m + "_level"] = k
            pc.append(rec)
    out_tables[f"bycell_{model}"] = pd.DataFrame(pc)

# sensitivity (Model A): dr3; headline levels of M2
P("\n######## Sensitivity (Model A, all cells): detection within 3 months (dr3) and the two-channel headline of M2 ########")
sens = []
for sub in ("All cells", "mu 0.05", "mu 0.2"):
    d = SUBSETS[sub]
    for r in R_REPORT:
        for m in METHODS + REFS:
            k = best_level(d, m, r, "A")
            k3 = best_level(d, m, r, "A", dr="dr3")
            sens.append(dict(subset=sub, r=r, method=m, label=LABEL[m], EC_dr1=ec(d, m, k, r, "A").mean(), level_dr1=k,
                             EC_dr3=ec(d, m, k3, r, "A", dr="dr3").mean(), level_dr3=k3))
        for kk in (1, 2):
            sens.append(dict(subset=sub, r=r, method=f"head{kk}", label=f"M2 headline level {kk}", EC_dr1=ec(d, "head", kk, r, "A").mean(), level_dr1=kk,
                             EC_dr3=ec(d, "head", kk, r, "A", dr="dr3").mean(), level_dr3=kk))
S = pd.DataFrame(sens)
for r in (5.0, 20.0):
    P(f"r = {int(r)}: " + "; ".join(f"{q.label.split(' (')[0]} {q.EC_dr1:.2f}->{q.EC_dr3:.2f}" for _, q in S[(S.subset == 'All cells') & (S.r == r)].iterrows()))
out_tables["sensitivity_A"] = S

# ---------------------------------------------------------------- write
keep = ["cell", "rep", "alpha", "mu", "regime", "t0", "n_shocks_h1", "E_h1", "n_shocks_h3", "E_h3", "n_valid", "n_pos", "base_rate_M2"]
for m in METHODS + ["head"]:
    for k in levels_of(m):
        kk = key(m, k)
        keep += [f"{kk}_dr1", f"{kk}_dr3", f"{kk}_am", f"{kk}_on", f"{kk}_fa", f"{kk}_fa_stable"]
D[keep].to_csv(os.path.join(OUT, "cost_inputs_by_replication.csv"), index=False)
for name, t in out_tables.items():
    if name.startswith("curves"):
        model = name.split("_")[1]
        g = {"r": R_GRID}
        for sub in ("All cells", "mu 0.05", "mu 0.2", "stable", "gradual", "abrupt"):
            for m, (env, pick, M) in t[sub].items():
                g[f"{sub}|{m}"] = env
                if m == "M2":
                    for i, kk in enumerate(levels_of("M2")):
                        g[f"{sub}|M2_level{kk}"] = M[i]
        pd.DataFrame(g).to_csv(os.path.join(OUT, f"cost_curves_grid_{model}.csv"), index=False)
    else:
        t.to_csv(os.path.join(OUT, f"cost_{name}.csv"), index=False)
txt = "\n".join(digest)
print(txt)
open(os.path.join(OUT, "cost_digest.txt"), "w").write(txt + "\n")

# ---------------------------------------------------------------- figure (Model A)
if STYLE:
    sys.path.insert(0, STYLE)
    import figstyle as RF
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    RF.apply()
    cur = out_tables["curves_A"]
    fig, axes = plt.subplots(2, 2, figsize=(RF.W2, 0.80 * RF.W2), constrained_layout=True)
    SHOW = [("M2", "M2 (proposed), best of its three levels"), ("M0", "Poisson M0"), ("xgb", "XGBoost"), ("lstm", "LSTM"),
            ("ewma", "EWMA"), ("cusum", "CUSUM"), ("farr", "Farrington-type threshold")]
    MARK_R = np.array([1, 2, 5, 10, 20, 50, 100], float)
    mk_idx = [int(np.argmin(np.abs(R_GRID - r))) for r in MARK_R]
    for j, sub in enumerate(("mu 0.05", "mu 0.2")):
        cc = cur[sub]
        triv = np.minimum(cc["never"][0], cc["always"][0])
        ax, axr = axes[0, j], axes[1, j]
        env, pick, M = cc["M2"]
        for i, kk in enumerate((1, 2, 3)):                         # the three fixed levels of M2, thin
            ax.plot(R_GRID, M[i], ls=(0, (2.0, 1.6)), color=RF.tint(RF.RED, 0.45), lw=0.7, zorder=2)
            axr.plot(R_GRID, M[i] / triv, ls=(0, (2.0, 1.6)), color=RF.tint(RF.RED, 0.45), lw=0.7, zorder=2)
        for m, lab in SHOW:
            c, mk, filled = RF.METHOD[m]
            sty = dict(ls="-", color=c, lw=1.5 if m == "M2" else 1.0, zorder=4 if m == "M2" else 3, marker=mk, markevery=mk_idx,
                       mfc=c if filled else "white", mec="white" if filled else c, mew=0.5 if filled else 0.9, ms=5.2 if m == "M2" else 4.2)
            ax.plot(R_GRID, cc[m][0], **sty)
            axr.plot(R_GRID, cc[m][0] / triv, **sty)
        ax.plot(R_GRID, cc["never"][0], ls=(0, (1.2, 1.4)), color=RF.MUTED, lw=0.9, zorder=1)
        ax.plot(R_GRID, cc["always"][0], ls=(0, (4, 2)), color=RF.MUTED, lw=0.9, zorder=1)
        axr.axhline(1.0, color=RF.INK, lw=0.7, ls=(0, (4, 2)), zorder=1)
        for a in (ax, axr):
            a.set_xscale("log")
            a.set_xticks([1, 2, 5, 10, 20, 50, 100])
            a.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
            a.minorticks_off()
            a.set_xlim(1, 100)
        ax.set_ylim(0, 16)
        axr.set_ylim(0.6, 1.6)
        axr.set_xlabel("Avoidable loss per shock, $r$ (units of one alarm-month)")
        ax.set_title(r"cells with $\bar{\mu}$ = 0.05 (shocks rare)" if sub == "mu 0.05" else r"cells with $\bar{\mu}$ = 0.2 (shocks frequent)",
                     loc="right", fontsize=8, pad=4)
    axes[0, 0].set_ylabel("Expected cost per item-year\n(alarm-months)")
    axes[1, 0].set_ylabel("Cost relative to the better of\nnever acting and always acting")
    RF.letters(fig, axes.ravel(), groups=((0, 2), (1, 3)))
    hs = [RF.method_handle(m, lab, line=True) for m, lab in SHOW]
    hs.insert(1, Line2D([], [], color=RF.tint(RF.RED, 0.45), lw=0.7, ls=(0, (2.0, 1.6)), label="M2, fixed levels 1–3"))
    hs.append(Line2D([], [], color=RF.MUTED, lw=0.9, ls=(0, (1.2, 1.4)), label="Never act ($rE$)"))
    hs.append(Line2D([], [], color=RF.MUTED, lw=0.9, ls=(0, (4, 2)), label="Always act (12 alarm-months)"))
    fig.legend(handles=hs, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10), frameon=False)
    FIGOUT = os.path.join(STYLE, "out"); os.makedirs(FIGOUT, exist_ok=True)
    RF.save(fig, os.path.join(FIGOUT, "Fig5_cost_curves"), tight=True)
    print("figure written:", os.path.join(FIGOUT, "Fig5_cost_curves.pdf/.png"))
