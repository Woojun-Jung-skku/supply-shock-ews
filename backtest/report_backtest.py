"""report_backtest.py — a markdown summary of a backtest_multi.py output folder: event counts, the Table 6-type metrics
against each event definition, the cost bands and the restriction-event detail (what was in force before each
export-restriction implementation).  usage: python report_backtest.py OUT_DIR [REPORT_MD]"""
import csv, json, os, sys
import pandas as pd

out = sys.argv[1]; rep = sys.argv[2] if len(sys.argv) > 2 else os.path.join(out, "report.md")
S = json.load(open(os.path.join(out, "summary.json")))
L = [f"# Backtest report — {os.path.basename(os.path.abspath(out))}", "",
     f"Items {len(S['items'])}, groups {S['groups']}, months {S['T']}, origins {S['origins'][0]}..{S['origins'][-1]} ({len(S['origins'])}), "
     f"item-years at decision months {S['item_years']}, fit target {S['fit_target']}.", "",
     f"Events in the evaluation window: trade-based N1 = {S['events']['N1']}, restriction N2 = {S['events']['N2']}, union N3 = {S['events']['N3']}; "
     f"restriction events with alarm history = {S.get('n_restriction_events_in_window', 0)}.", "",
     f"q-channel calibration: threshold {S.get('q_threshold')}, persistence {S.get('q_persistence')}, implied in-control run length {S.get('arl0_boot')} months.", ""]
NAME = {"M2": "M2 (proposed)", "M1": "NB single dispersion", "M0": "Poisson M0", "sigma": "Sigma-rule (plug-in SRI)", "ewma": "EWMA", "cusum": "CUSUM",
        "farr": "Farrington-type", "xgb": "XGBoost (pooled)", "xgbcal": "XGBoost, recalibrated"}
for tgt, lab in (("N1", "trade-based interruption events"), ("N2", "export-restriction implementations (independent)"), ("N3", "union")):
    f = os.path.join(out, f"metrics_{tgt}.csv")
    if not os.path.exists(f):
        continue
    m = pd.read_csv(f)
    L += [f"## Target {tgt}: {lab}", "", "| Method | Level | Detected ≤1 m | ≤3 m | False onsets / item-year | Alarm months / item-year | Lead (median) | AUC-PR | Brier | ECE |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in m.iterrows():
        lv = "" if pd.isna(r["level"]) else f"L{int(r['level'])}"
        f2 = lambda x, d=2: "" if pd.isna(x) else f"{x:.{d}f}"
        L.append(f"| {NAME.get(r['method'], r['method'])} | {lv} | {f2(r['dr_h1'])} | {f2(r['dr_h3'])} | {f2(r['fa_per_itemyear'])} | {f2(r['alarm_months_per_itemyear'], 1)} | {f2(r['lead_med'], 1)} | "
                 f"{f2(r.get('aucpr'), 3)} | {f2(r.get('brier'), 3)} | {f2(r.get('ece'), 3)} |")
    q = m[(m.method == "M2") & (m.level == 2)]
    if len(q) and "q_dr_h1" in q.columns and not pd.isna(q.iloc[0]["q_dr_h1"]):
        r = q.iloc[0]
        L += ["", f"M2 change channel (q): detected ≤1 m {r['q_dr_h1']:.2f}, share of months in force {r['q_alarm_share']:.2f}, onsets {int(r['q_n_onsets'])}; "
                  f"headline level (max of triage level 2 and q): detected ≤1 m {r['headline_dr_h1']:.2f}, false onsets {r['headline_fa_per_itemyear']:.2f}, alarm months {r['headline_alarm_months_per_itemyear']:.1f}."]
    b = pd.read_csv(os.path.join(out, f"cost_bands_{tgt}.csv"))
    L += ["", f"Cost (Model A, E = {S.get('E_' + tgt)} events per item-year): band of r in which each policy beats both never and always acting, and its largest saving against the better trivial policy", "",
          "| Method | band of r | max saving | at r |", "|---|---|---|---|"]
    for _, r in b.iterrows():
        band = "none" if pd.isna(r["band_lo"]) else f"{r['band_lo']}–{r['band_hi']}"
        L.append(f"| {NAME.get(r['method'], r['method'])} | {band} | {100 * r['max_saving']:.1f}% | {r['r_at_max']} |")
    L.append("")
f = os.path.join(out, "events_N2_detail.csv")
if os.path.exists(f):
    d = pd.read_csv(f)
    L += ["## Export-restriction implementations: what was in force when each measure took effect", "",
          "The posture dated t is set on the data through t - 1 (the convention of the one-month detection rate); the table also gives the level dated t - 1.", ""]
    if "M2_level_at" in d.columns:
        n = len(d); mb = d["M2_months_alert_before"]
        L += [f"{n} restriction events. M2 triage level ≥ 2 in force at the implementation month: {(d['M2_level_at'] >= 2).mean():.0%}; level ≥ 1: {(d['M2_level_at'] >= 1).mean():.0%}; "
              + (f"q-alarm in force: {d['M2_q_alarm_at'].astype(bool).mean():.0%}; " if "M2_q_alarm_at" in d.columns else "")
              + f"median months since the item first reached the alert level: {mb.median() if mb.notna().any() else 'n/a'} (over {int(mb.notna().sum())} events).", ""]
        for col, nm in (("M0_level_at", "Poisson M0 level ≥ 2"), ("xgb_level_at", "XGBoost level ≥ 2"), ("xgbcal_level_at", "XGBoost (recalibrated) level ≥ 2"), ("sigma_level_at", "sigma-rule level ≥ 2")):
            if col in d.columns:
                L.append(f"{nm} at the implementation month: {(d[col] >= 2).mean():.0%}.")
        for col, nm in (("ewma_alarm_at", "EWMA"), ("cusum_alarm_at", "CUSUM"), ("farr_alarm_at", "Farrington-type")):
            if col in d.columns:
                L.append(f"{nm} alarm at the implementation month: {d[col].astype(bool).mean():.0%}.")
        L += ["", "| Item | Group | Month | Measure | M2 p(3) | M2 level | M2 level the month before | months since first alert | q-alarm |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            mbv = "" if pd.isna(r.get("M2_months_alert_before")) else int(r["M2_months_alert_before"])
            L.append(f"| {r['item']} | {r['group']} | {r['month']} | {str(r['measure'])[:50]} | {r.get('M2_p_at', '')} | {r.get('M2_level_at', '')} | {r.get('M2_level_tm1', '')} | {mbv} | {r.get('M2_q_alarm_at', '')} |")
open(rep, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("written", rep)
