# -*- coding: utf-8 -*-
"""gen_supp_backtest.py - Supplementary Note 14 and Tables S25-S28 (the backtest on the Korean import panel, Section 5.5)
from the outputs of backtest/backtest_multi.py, plus the main-text Table 7 (table7.md) and the numbers the text quotes
(key_numbers.json).

usage: python supplementary/gen_supp_backtest.py MAIN_OUT PANEL_CSV RESTRICTIONS_CSV ITEMS_CSV SUPP_DIR
                                   [--strict-out DIR] [--top-out DIR] [--total-out DIR] [--dropped CSV]
  MAIN_OUT     backtest_multi.py output folder of the main run (E1: partner share >= 0.25; E2: all measures)
  --strict-out the same fits evaluated on the strict E2 list (new controls only)
  --top-out, --total-out  runs on the E1 sensitivity panels (largest partner; total imports)
Writes SUPP_DIR/supplementary_backtest.tex, SUPP_DIR/tables/S25..S28 csv, MAIN_OUT/table7.md and MAIN_OUT/key_numbers.json.
"""
import argparse, csv, json, os
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("panel"); ap.add_argument("restrictions"); ap.add_argument("items"); ap.add_argument("supp")
ap.add_argument("--strict-out"); ap.add_argument("--top-out"); ap.add_argument("--total-out"); ap.add_argument("--dropped")
a = ap.parse_args()
TAB = os.path.join(a.supp, "tables"); os.makedirs(TAB, exist_ok=True)

LAB = {"M2": "M2 (proposed)", "M0": "Poisson M0", "xgb": "XGBoost (pooled)", "xgbcal": "XGBoost, recalibrated", "sigma": "Sigma-rule (plug-in SRI)", "ewma": "EWMA",
       "cusum": "CUSUM", "farr": "Farrington-type threshold"}
ORDER = ["M2", "M0", "xgb", "xgbcal", "sigma", "ewma", "cusum", "farr"]
GROUP_LAB = {"semiconductor_chemicals": "Semiconductor chemicals, gases and wafers", "battery_materials": "Battery materials",
             "refractory_minor_metals": "Refractory and minor metals", "ferroalloys": "Ferroalloys", "rare_earths_magnets": "Rare earths and magnets"}
TGT_LAB = {"N1": "E1: trade interruptions", "N2": "E2: export-restriction implementations", "N3": "E3: union"}

S = json.load(open(os.path.join(a.out, "summary.json")))
panel = pd.read_csv(a.panel, dtype={"cmd": str, "period": str})
items_meta = {r["hs6"]: r for r in csv.DictReader(open(a.items, encoding="utf-8"))}
restr = pd.read_csv(a.restrictions, dtype=str).fillna("")
out, P = [], None
P = lambda s="": out.append(s)
KEY = {}


def esc(s):
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


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
    pd.DataFrame(rows, columns=header).to_csv(os.path.join(TAB, name), index=False, encoding="utf-8")


def f(x, d=2, dash="--"):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return dash
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return dash


# ------------------------------------------------------------------------------------------------ panel composition (S25)
T = S["T"]; DEC0 = S["origins"][0]
dec = panel[panel.period >= sorted(panel.period.unique())[DEC0]]
comp_rows, comp_csv = [], []
for g in ["semiconductor_chemicals", "battery_materials", "refractory_minor_metals", "ferroalloys", "rare_earths_magnets"]:
    sub = panel[panel.group == g]; subd = dec[dec.group == g]
    n_items = sub.cmd.nunique()
    e1 = int((subd.N1 > 0).sum()); e2 = int((subd.N2 > 0).sum()); e2a = int((subd.N2a > 0).sum())
    im = len(subd)
    comp_rows.append([GROUP_LAB[g], n_items, im, e1, f(100 * e1 / im, 1), e2, e2a])
    comp_csv.append([g, n_items, im, e1, round(100 * e1 / im, 2), e2, e2a])
tot_items = panel.cmd.nunique(); tot_im = len(dec)
tot_e1 = int((dec.N1 > 0).sum()); tot_e2 = int((dec.N2 > 0).sum()); tot_e2a = int((dec.N2a > 0).sum())
comp_rows.append([r"\textbf{All groups}", tot_items, tot_im, tot_e1, f(100 * tot_e1 / tot_im, 1), tot_e2, tot_e2a])
comp_csv.append(["all", tot_items, tot_im, tot_e1, round(100 * tot_e1 / tot_im, 2), tot_e2, tot_e2a])
KEY.update(items=tot_items, groups=5, item_months_decision=tot_im, e1_item_months=tot_e1, e1_rate=round(tot_e1 / tot_im, 4),
           e2_item_months=tot_e2, e2a_item_months=tot_e2a, decision_months=len(dec.period.unique()),
           first_decision_month=dec.period.min(), last_month=dec.period.max(), item_years=S["item_years"])

P(r"\textbf{Supplementary Note 14. The backtest on public trade data (Section 5.5 of the main text).}")
P("")
P(r"\textit{Data.} Monthly imports of the Republic of Korea (UN Comtrade reporter 410) by partner at the six-digit HS level, "
  r"extracted from the UN Comtrade data API on 26 October 2026 with \path{backtest/pull_comtrade_multi.py} for the 65 items of "
  r"Table S25.2 (five groups: semiconductor chemicals, gases and wafers; battery materials; refractory and minor metals; ferroalloys; "
  r"rare earths and magnets), 2013-01 to 2025-12 (Korea's monthly series on Comtrade begin in 2013; HS 810291, unwrought molybdenum, "
  r"is coded 810294 from HS 2007 and returned no records, leaving 64 items). Quantities are net weights in kilograms; the 11\% of "
  r"partner-month records reported without a weight enter the panel as zero quantity and their value enters the unit value---all "
  r"5,440 records of 2013 are among them, so quantities begin in 2014-01, the panel in 2014-02 and complete trailing-twelve-month "
  r"windows in 2015-01, two years before the first decision month. "
  r"Export-restriction measures come from three sources merged by \path{backtest/events_restrictions.py}: the OECD Inventory of "
  r"Export Restrictions on Industrial Raw Materials (complete data set, 2009--2021 at the six-digit level of HS 2007, from the OECD web "
  r"archive of 12 December 2022, the current location of the file), the Global Trade Alert database (the curated data set `Import and "
  r"Export Restrictions: Critical Minerals' and a query on the 65 item codes, 1,593 interventions, 2008--2026, extracted through the "
  r"GTA Data Center on 26 October 2026, CC BY-NC 4.0) and a seed list of eleven documented measures whose announcement and "
  r"implementation dates were checked against the GTA records (nine matched on partner, implementation month and HS scope; the "
  r"October 2025 extension of China's rare-earth controls carries the GTA implementation date of 10 November 2026 after its "
  r"suspension; Zimbabwe's 2022 lithium-ore ban covers no listed item). From the GTA records we keep the five export-type intervention "
  r"types (export ban, export licensing requirement, export quota, export tax, other export-related non-tariff measure), measures whose "
  r"affected jurisdictions include the Republic of Korea (which removes measures aimed at other destinations, such as entity-list "
  r"controls and sanctions packages), measures listing at most 100 HS6 codes (which removes economy-wide frameworks: full export-tariff "
  r"schedules, catch-all rules, the ratification of an export-control law) and measures whose title does not name sanctions, entity "
  r"lists or catch-all controls; from the OECD inventory we keep every introduction, increase, extension or revision of a measure "
  r"(eliminations, decreases and the annual `left unchanged' records are dropped) grouped by country, type, date of introduction and "
  r"legal basis. A measure recorded by more than one source counts once per partner and month. The resulting table holds " + str(len(restr)) +
  r" measures by " + str(restr.partner_iso3.nunique()) + r" implementing jurisdictions, an EU measure counted once per member state "
  r"(Table S27 lists those that became events).")
M = {t: pd.read_csv(os.path.join(a.out, f"metrics_{t}.csv")) for t in ("N1", "N2", "N3")}
_m2 = M["N1"][(M["N1"].method == "M2") & (M["N1"].level == 2)].iloc[0]
KEY["base_rate_3m"] = float(_m2["base_rate"])


def diag_summary(out_dir):
    """convergence record of the fits (fits/diagnostics.csv): repeats by variant and the worst R-hat / ESS of the kept fits."""
    p = os.path.join(out_dir, "fits", "diagnostics.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    rec = {}
    for v in sorted(d.variant.unique()):
        dv = d[d.variant == v]
        kept = dv.sort_values("attempt").groupby("origin").tail(1)                      # the fit in use at each origin
        fails = kept[kept["pass"] == 0]
        esscols = [c for c in kept.columns if c.startswith("ess_")]
        sym = {"beta": r"$\boldsymbol{\beta}$", "alpha": r"$\alpha'$", "logC": r"$\log C$", "m": "the group levels", "d": "the group slopes", "sig": r"$\sigma_\eta$", "tau": r"$\tau_\zeta$"}
        fail_desc = [f"origin {int(r.origin)}, where {sym.get(c[4:], c[4:])} has ESS {int(r[c])}" for _, r in fails.iterrows() for c in [min(esscols, key=lambda c: r[c])]] if len(fails) else []
        rec[v] = dict(n=int(kept.origin.nunique()), repeated=int((dv.attempt == 2).sum()), still_failing=int(len(fails)),
                      max_rhat=float(kept.max_rhat.max()), min_ess=float(kept.min_ess.min()),
                      repeated_origins=sorted(int(x) for x in dv[dv.attempt == 2].origin.unique()), failing=fail_desc)
    return rec


DIAG = diag_summary(a.out)
KEY["convergence"] = DIAG


def diag_sentence(D, variants=("M2", "M0")):
    if not D:
        return ""
    parts = []
    for v in variants:
        if v in D:
            r = D[v]
            lab = "M2" if v == "M2" else "the Poisson model"
            nrep = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(r['repeated'], str(r['repeated']))
            parts.append(f"{nrep} of the {r['n']} fits of {lab}" + (f" (origins {', '.join(str(o) for o in r['repeated_origins'])})" if r['repeated'] else ""))
    worst_r = max(D[v]["max_rhat"] for v in variants if v in D); worst_e = min(D[v]["min_ess"] for v in variants if v in D)
    fails = sum(D[v]["still_failing"] for v in variants if v in D)
    fail_txt = "; ".join(f"{'M2' if v == 'M2' else 'the Poisson model'} at {', '.join(D[v]['failing'])}" for v in variants if v in D and D[v]["failing"])
    return (r" Chains start near the prior medians (the median of 15 prior draws per site); a fit whose rank-normalised split-$\widehat{R}$ "
            r"reached 1.05 or whose bulk effective sample size fell below 100 on any stored quantity ($\boldsymbol{\beta}$, $\alpha'$, $\log C$, "
            r"the group levels and slopes at the origin, $\sigma_\eta$, $\tau_\zeta$) was repeated once with 1,000 warm-up and 1,000 sampling "
            r"iterations and kept either way (" + " and ".join(parts) + r"); the fits in use have $\widehat{R} \leq " + f"{worst_r:.3f}" +
            r"$ and effective sample sizes of at least " + f"{worst_e:.0f}" + (r" at every origin" if fails == 0 else (r" (the repeated fit still below the criterion: " if fails == 1 else r" (the repeated fits still below the criterion: ") + fail_txt + ")") +
            r" (\path{results/backtest/main/fits/diagnostics.csv}).")


P("")
P(r"\textit{Events and indicators.} The panel runs from 2014-02, the first month with a positive trailing-twelve-month quantity (the "
  r"windows are complete from 2015-01), to 2025-12, 143 months, and decisions are made from month 36 (2017-02) to the end, " + str(KEY['decision_months']) + " months. A trade "
  r"interruption (E1) is recorded for item $i$ in month $t$ when a partner that supplied at least a quarter of the item's imports over "
  r"the trailing twelve months ships less than half of its own trailing-twelve-month median; an export-restriction implementation (E2) "
  r"when a partner with a trailing share of at least 5\% implements a measure whose HS scope covers the item (exact six-digit code or "
  r"a two- or four-digit parent); announcement months are recorded separately (E2a) for the lead-time analysis and are never a target. "
  r"E3 is the union. The first version of the backtest plan used a partner-level definition with a lower share threshold (any partner "
  r"with a trailing share of at least 5\%); on the 64-item panel it fired in about 35\% of all item-months over 2014--2025 and in 37\% of the "
  r"decision item-months (4.4 per item-year, a three-month prevalence of 58\%), outside the prevalence range of the simulation design "
  r"(24--55\% over the three-month horizon, Section 5.1) "
  r"and high enough to keep the level-2 posture in force in eleven of twelve months. The share threshold was therefore raised to 25\% "
  r"before the main run and fixed; with it E1 occurs in " + f(100 * KEY['e1_rate'], 1) + r"\% of the decision item-months "
  r"(" + f(12 * KEY['e1_rate'], 1) + r" per item-year; three-month prevalence " + f(100 * KEY['base_rate_3m'], 0) + r"\%). Two alternative definitions---the largest "
  r"partner's shipments halving, and the item's total imports halving---are reported as sensitivity runs in Table S28; items are "
  r"screened on data coverage only (at least 60 months with positive imports), never on the number of events. The indicators and "
  r"exposure measurements are those of Section 5.5 of the main text (partner Herfindahl index, top-partner share, twelve-month change "
  r"in unit value, three-month growth in quantity; top-partner share and unit-value level over the first 36 months as exposure), "
  r"lagged one month. The protocol is that of Section 5.3: refits at origins 36, 42, \ldots, " + str(S["origins"][-1]) + r" (" + str(len(S["origins"])) +
  r" origins, two chains of 500 warm-up and 600 sampling iterations), particle propagation between refits, thresholds $\pi = (0.048, 0.167, 0.5)$ "
  r"with 2-of-2 persistence, the $q$ channel calibrated to ARL$_0 \geq 24$ months in 200 bootstrap paths from the posterior at the "
  r"first origin (threshold " + f(S['q_threshold']['M2'], 2) + ", persistence " + str(S.get('q_persistence', {}).get('M2', '')) +
  r"-of-" + str(S.get('q_persistence', {}).get('M2', '')) + r", implied ARL$_0$ " + f(S.get('arl0_boot', {}).get('M2', float('nan')), 0) +
  r" months), and the baselines of Supplementary Note 2 fitted on the same information set (the LSTM and the temporal fusion "
  r"transformer of Study B are not refitted here; XGBoost, pooled across the items, represents the machine-learning family, and a second "
  r"variant holds out the twelve labelled months before each origin, recalibrates the model's probabilities on them by isotonic regression "
  r"and applies the recalibration until the next origin)."
  + diag_sentence(DIAG) +
  r" A posture dated month $t$ is set on the data through month $t-1$; the probability metrics score $p_{it}(3)$ dated $t$ against an "
  r"event in months $t$ to $t+2$, the window it predicts, and a shock in month $t$ counts as detected within one month when the level "
  r"dated $t$ is in force (within three months, when it is in force in any of the months $t-2$ to $t$).")
P("")
table_env("tab:S25_1", r"Table S25.1. Composition of the Korean import panel by group: items, decision item-months (2017-02 to 2025-12), "
          r"trade interruptions (E1), their share of item-months, export-restriction implementations (E2) and announcements (E2a).",
          ["Group", "Items", "Item-months", "E1", r"E1 share (\%)", "E2", "E2a"], comp_rows, "lrrrrrr")
csv_out("S25_1_panel_composition.csv", ["group", "items", "item_months", "E1", "E1_share_pct", "E2", "E2a"], comp_csv)

# S25.2: the item list with event counts
it_rows, it_csv = [], []
for g in ["semiconductor_chemicals", "battery_materials", "refractory_minor_metals", "ferroalloys", "rare_earths_magnets"]:
    for code, meta in items_meta.items():
        if meta["group"] != g:
            continue
        sub = dec[dec.cmd == code]
        if sub.empty:
            it_rows.append([code, esc(meta["name"]), GROUP_LAB[g], "--", "--", "--", "dropped: no data"]); it_csv.append([code, meta["name"], g, "", "", "", "dropped: no data"]); continue
        pos = int((panel[panel.cmd == code].total_qty_kg > 0).sum())
        it_rows.append([code, esc(meta["name"][:60]), GROUP_LAB[g].split(",")[0], pos, int((sub.N1 > 0).sum()), int((sub.N2 > 0).sum()), ""])
        it_csv.append([code, meta["name"], g, pos, int((sub.N1 > 0).sum()), int((sub.N2 > 0).sum()), ""])
table_env("tab:S25_2", r"Table S25.2. The 65 items: HS 2017 six-digit code, description, group, months with positive imports (2014-02 to 2025-12), "
          r"and E1 and E2 item-months in the decision window.", ["HS6", "Item", "Group", "Months", "E1", "E2", "Note"], it_rows, "llp{3.2cm}rrrl", r"\tiny")
csv_out("S25_2_items.csv", ["hs6", "name", "group", "months_with_imports", "E1", "E2", "note"], it_csv)

# ------------------------------------------------------------------------------------------------ metrics (S26 and Table 7)
CB = {t: pd.read_csv(os.path.join(a.out, f"cost_bands_{t}.csv")) for t in ("N1", "N2", "N3")}


def mrow(t, name, level):
    d = M[t]; sub = d[d.method == name]
    if level is None:
        return sub.iloc[0]
    r = sub[sub.level == level]
    return r.iloc[0] if not r.empty else None


def lead(r):
    return f"{f(r['lead_med'], 1)} ({f(r['lead_q25'], 1)}--{f(r['lead_q75'], 1)})" if r is not None and not np.isnan(r["lead_med"]) else "--"


for t in ("N1", "N2", "N3"):
    rows_s, rows_c = [], []
    for name in ORDER:
        levels = [1, 2, 3] if name in ("M2", "M0", "xgb", "xgbcal", "sigma") else [None]
        for k in levels:
            r = mrow(t, name, k)
            if r is None:
                continue
            lab = LAB[name] + (f", level {k}" if k else "")
            rows_s.append([lab, f(r["aucpr"], 3), f(r["brier"], 3), f(r["ece"], 3), f(r["dr_h1"], 2), f(r["dr_h3"], 2), f(r["fa_per_itemyear"], 2),
                           f(r["alarm_months_per_itemyear"], 1), lead(r)])
            rows_c.append([name, k if k else "", r["aucpr"], r["brier"], r["ece"], r["dr_h1"], r["dr_h3"], r["fa_per_itemyear"], r["alarm_months_per_itemyear"],
                           r["lead_med"], r["lead_q25"], r["lead_q75"], r["n_onsets"]])
        if name in ("M2", "M0"):
            r = mrow(t, name, 2)
            rows_s.append([LAB[name] + ", $q$ channel", "--", "--", "--", f(r["q_dr_h1"], 2), f(r["q_dr_h3"], 2), "--", f(12 * r["q_alarm_share"], 1), f(r["q_lead_med"], 1)])
            rows_s.append([LAB[name] + ", headline (max of level 2 and $q$)", "--", "--", "--", f(r["headline_dr_h1"], 2), "--", f(r["headline_fa_per_itemyear"], 2),
                           f(r["headline_alarm_months_per_itemyear"], 1), "--"])
            rows_c.append([name, "q", "", "", "", r["q_dr_h1"], r["q_dr_h3"], "", 12 * r["q_alarm_share"], r["q_lead_med"], "", "", r["q_n_onsets"]])
            rows_c.append([name, "headline", "", "", "", r["headline_dr_h1"], "", r["headline_fa_per_itemyear"], r["headline_alarm_months_per_itemyear"], "", "", "", ""])
    n = {"N1": "1", "N2": "2", "N3": "3"}[t]
    table_env(f"tab:S26_{n}", f"Table S26.{n}. Alarm performance on the Korean import panel, target {esc(TGT_LAB[t])}: event probabilities for "
              r"`at least one event within three months' and the alarm at each level---events flagged within one and within three months before they occur, "
              r"false alarm onsets per item-year (onsets not followed by an event within three months), alarm months per item-year and lead time in months, "
              r"median (quartiles); the $q$ channel and the headline level of the two models.",
              ["Method", "AUC-PR", "Brier", "ECE", r"Det.\ $\leq$1 m", r"Det.\ $\leq$3 m", "False alarms", "Alarm months", "Lead"], rows_s, "lrrrrrrrl", r"\tiny")
    csv_out(f"S26_{n}_metrics_{t}.csv", ["method", "level", "aucpr", "brier", "ece", "dr_h1", "dr_h3", "fa_per_itemyear", "alarm_months_per_itemyear", "lead_med", "lead_q25", "lead_q75", "n_onsets"], rows_c)

# key numbers for the text
for t in ("N1", "N2", "N3"):
    for name in ORDER:
        for k in ([1, 2, 3] if name in ("M2", "M0", "xgb", "xgbcal", "sigma") else [None]):
            r = mrow(t, name, k)
            if r is None:
                continue
            KEY[f"{t}_{name}_L{k if k else 1}"] = {c: (None if (isinstance(r[c], float) and np.isnan(r[c])) else (float(r[c]) if isinstance(r[c], (float, int, np.floating, np.integer)) else r[c]))
                                                   for c in M[t].columns if c not in ("method", "target", "level")}
KEY["q_threshold"] = S["q_threshold"]; KEY["q_persistence"] = S.get("q_persistence"); KEY["arl0_boot"] = S.get("arl0_boot")
KEY["events_window"] = S["events"]; KEY["E_per_item_year"] = {k: S.get(f"E_{k}") for k in ("N1", "N2", "N3")}
KEY["cost_bands"] = {t: CB[t].to_dict("records") for t in ("N1", "N2", "N3")}

# ------------------------------------------------------------------------------------------------ Table 7 (main text, markdown)
t8 = ["| Method | AUC-PR | Brier | ECE | Det. ≤1 m | Det. ≤3 m | False alarms / item-year | Alarm months / item-year | Lead, median (IQR) | E2: det. ≤1 m | E2: in force at implementation (S27) | E2: alarm months |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
det = pd.read_csv(os.path.join(a.out, "events_N2_detail.csv"))
nE2 = len(det)
inforce = {}
for name in ORDER:
    col = {"M2": "M2_level_at", "M0": "M0_level_at", "xgb": "xgb_level_at", "xgbcal": "xgbcal_level_at", "sigma": "sigma_level_at"}.get(name)
    if col:
        inforce[name] = float((det[col] >= 2).mean())
    else:
        inforce[name] = float(det[f"{name}_alarm_at"].astype(str).str.lower().eq("true").mean())
KEY["E2_in_force_at_implementation_level2"] = inforce; KEY["n_E2_events_detail"] = nE2          # posture dated t = set on data through t-1 (= dr_h1)
KEY["E2_M2_level1_at"] = float((det["M2_level_at"] >= 1).mean()); KEY["E2_M2_q_at"] = float(det["M2_q_alarm_at"].astype(str).str.lower().eq("true").mean())
KEY["E2_M2_months_alert_before_median"] = float(det["M2_months_alert_before"].median())
for name in ORDER:
    lev2 = name in ("M2", "M0", "xgb", "xgbcal", "sigma")
    r1 = mrow("N1", name, 2 if lev2 else None); r2 = mrow("N2", name, 2 if lev2 else None)
    lab = LAB[name] + (", level 2" if lev2 else "")
    t8.append(f"| {lab} | {f(r1['aucpr'], 3)} | {f(r1['brier'], 3)} | {f(r1['ece'], 3)} | {f(r1['dr_h1'], 2)} | {f(r1['dr_h3'], 2)} | {f(r1['fa_per_itemyear'], 2)} | "
              f"{f(r1['alarm_months_per_itemyear'], 1)} | {lead(r1)} | {f(r2['dr_h1'], 2)} | {f(inforce[name], 2)} | {f(r2['alarm_months_per_itemyear'], 1)} |")
r = mrow("N1", "M2", 2); r2 = mrow("N2", "M2", 2)
t8.append(f"| M2, q channel | -- | -- | -- | {f(r['q_dr_h1'], 2)} | {f(r['q_dr_h3'], 2)} | -- | {f(12 * r['q_alarm_share'], 1)} | {f(r['q_lead_med'], 1)} | {f(r2['q_dr_h1'], 2)} | {f(KEY['E2_M2_q_at'], 2)} | {f(12 * r2['q_alarm_share'], 1)} |")
open(os.path.join(a.out, "table7.md"), "w", encoding="utf-8").write("\n".join(t8) + "\n")

# ------------------------------------------------------------------------------------------------ S27: the restriction events
det["month"] = det["month"].astype(str)
d_rows, d_csv = [], []
for _, r in det.sort_values(["month", "item"]).iterrows():
    part, meas = str(r["measure"]).split(":", 1) if ":" in str(r["measure"]) else ("", str(r["measure"]))
    d_rows.append([r["month"][:4] + "-" + r["month"][4:], r["item"], part, esc(meas[:48]), int(r["M2_level_at"]), f(r["M2_p_at"], 2),
                   "yes" if str(r["M2_q_alarm_at"]).lower() == "true" else "no", int(r["M2_months_alert_before"]) if not np.isnan(r["M2_months_alert_before"]) else "--",
                   int(r["xgb_level_at"]), int(r["sigma_level_at"]), "yes" if str(r["ewma_alarm_at"]).lower() == "true" else "no",
                   "yes" if str(r["cusum_alarm_at"]).lower() == "true" else "no", "yes" if str(r["farr_alarm_at"]).lower() == "true" else "no"])
    d_csv.append([r["month"], r["item"], part, meas, r["M2_level_at"], r["M2_p_at"], r["M2_q_alarm_at"], r["M2_months_alert_before"], r["xgb_level_at"],
                  r["sigma_level_at"], r["ewma_alarm_at"], r["cusum_alarm_at"], r["farr_alarm_at"]])
table_env("tab:S27", r"Table S27. The export-restriction implementations (E2) in the decision window, one row per item and implementation month (a month in which two measures fall on one item is one row): implementation month, item, "
          r"implementing partner (Comtrade code), measure, and the posture in force in the implementation month, set on the data through the month before "
          r"(the convention of the detection rates at one month)---the triage level, three-month probability and $q$-alarm of M2 and the months since the "
          r"item first reached the alert level, the triage level of XGBoost and of the sigma-rule, and the EWMA, CUSUM and Farrington-type alarms. "
          r"Several items under one measure are separate rows.",
          ["Month", "HS6", "Partner", "Measure", "M2 lvl", "$p(3)$", "$q$", "Since alert", "XGB lvl", "Sigma lvl", "EWMA", "CUSUM", "Farr."], d_rows, "lllp{4.2cm}rrrrrrrrr", r"\tiny")
csv_out("S27_restriction_events.csv", ["month", "hs6", "partner", "measure", "M2_level_at", "M2_p_at", "M2_q_alarm_at", "M2_months_alert_before", "xgb_level_at",
                                       "sigma_level_at", "ewma_alarm_at", "cusum_alarm_at", "farr_alarm_at"], d_csv)

# ------------------------------------------------------------------------------------------------ S28: sensitivity (strict E2, E1 variants)
sens_rows, sens_csv = [], []
def add_sens(label, outdir, t, panel_path):
    if not outdir or not os.path.exists(os.path.join(outdir, f"metrics_{t}.csv")):
        return
    d = pd.read_csv(os.path.join(outdir, f"metrics_{t}.csv"))
    pv = pd.read_csv(panel_path, dtype={"cmd": str, "period": str}); pv = pv[pv.period >= sorted(pv.period.unique())[DEC0]]
    n_ev = int((pv[t] > 0).sum())                                               # event item-months in the decision window
    for name in ORDER:
        sub = d[d.method == name]; r = sub[sub.level == 2].iloc[0] if (sub.level == 2).any() else sub.iloc[0]
        pm = t == "N1"                                                           # probability metrics are for the E1 label only
        sens_rows.append([esc(label), LAB[name], f(r["aucpr"], 3) if pm else "--", f(r["ece"], 3) if pm else "--", f(r["dr_h1"], 2), f(r["dr_h3"], 2), f(r["fa_per_itemyear"], 2), f(r["alarm_months_per_itemyear"], 1), n_ev])
        sens_csv.append([label, name, r["aucpr"] if pm else "", r["ece"] if pm else "", r["dr_h1"], r["dr_h3"], r["fa_per_itemyear"], r["alarm_months_per_itemyear"], n_ev])
_pv = lambda suffix: a.panel.replace(".csv", f"_{suffix}.csv")
add_sens("E2 strict (new controls only), main fits", a.strict_out, "N2", _pv("strict"))
add_sens("E1 = largest partner halves", a.top_out, "N1", _pv("top"))
add_sens("E1 = total imports halve", a.total_out, "N1", _pv("total"))
if sens_rows:
    table_env("tab:S28", r"Table S28. Sensitivity of the backtest to the event definitions, level 2 of the probability-based methods and the sigma-rule: "
              r"the strict E2 list (new export controls only, annual catalogue and quota renewals excluded; evaluated with the fits of the main run) and the "
              r"two alternative E1 definitions (the largest partner's shipments halving; the item's total imports halving), each a separate run on the same "
              r"64 items. AUC-PR and ECE refer to the E1 probabilities and are not reported for the E2 rows. Last column: event item-months in the decision window.", ["Variant", "Method", "AUC-PR", "ECE", r"Det.\ $\leq$1 m", r"Det.\ $\leq$3 m", "False alarms", "Alarm months", "Events"],
              sens_rows, "llrrrrrrr", r"\tiny")
    csv_out("S28_sensitivity.csv", ["variant", "method", "aucpr", "ece", "dr_h1", "dr_h3", "fa_per_itemyear", "alarm_months_per_itemyear", "events"], sens_csv)

P(r"\textit{Files.} The panel (\path{backtest/panel_410.csv}), the merged measures (\path{backtest/restrictions.csv}), the extracts of the OECD inventory and "
  r"of the GTA database for the 65 items, the scripts and the result files of the runs (\path{results/backtest/}) are in the repository; the Comtrade records "
  r"themselves are not redistributed (UN Comtrade terms of use) and are re-extracted by the script, which needs a free API subscription key.")
open(os.path.join(a.supp, "supplementary_backtest.tex"), "w", encoding="utf-8").write("\n".join(out) + "\n")
json.dump(KEY, open(os.path.join(a.out, "key_numbers.json"), "w"), indent=1, default=str)
print("written", os.path.join(a.supp, "supplementary_backtest.tex"), "tables S25-S28 ->", TAB, "; table7.md and key_numbers.json ->", a.out)
