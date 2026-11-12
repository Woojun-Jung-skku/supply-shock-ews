"""Per-replication detail for the rare arm (3 cells x 5 replications): paired M2-M0 / M2-XGB differences in AUC-PR, ECE and
Brier score, the q channel replication by replication, the charts, severity and conformal sets, E per replication, and a
pooled table (Table S29.3 and the paired statements of Section 5.3). Expected cost: rare_cost.py.
usage: python analyze_full.py <repo root> <results_rare dir>"""
import json, glob, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
root, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(root, "code"))
import numpy as np, config as C; C.apply_profile("t4_min"); import dgp
def E_of(cell, rep, reg):
    s = dgp.simulate(100, 120, 0.02, 2.0, reg, seed=dgp.seed_of("B", cell, rep)); return (s["N"][:, 36:118] > 0).sum() / (100 * 82 / 12)
def levs(d): return [(d["triage"][f"level{k}"]["dr_h1"], d["triage"][f"level{k}"]["alarm_share"] * 12) for k in (1, 2, 3)]
cells = {}
for f in sorted(glob.glob(os.path.join(out, "B", "*__all.jsonl"))):
    recs = sorted([json.loads(l) for l in open(f)], key=lambda r: r["rep"]); cells[recs[0]["cell"].split("-")[3]] = recs
order = ["stable", "gradual", "abrupt"]
print("## per-replication metrics")
for reg in order:
    recs = cells[reg]; Es = [E_of(recs[0]["cell"], r["rep"], reg) for r in recs]
    print(f"\n### {reg}: E per rep {np.round(Es,3).tolist()}, prevalence per rep {[round(r['M2']['prob']['base_rate'],3) for r in recs]}")
    print(f"{'rep':>3s} | {'AUC-PR M2':>9s} {'M0':>6s} {'XGB':>6s} | {'ECE M2':>7s} {'M0':>7s} {'XGB':>7s} | {'Brier M2':>8s} {'M0':>7s} {'XGB':>7s} | L2dr M2 M0 XGB | q M2 cal/boot/arl0/arl1 | q M0 cal/boot/arl0/arl1 | refit")
    for r in recs:
        P = {k: r[k]["prob"] for k in ("M2", "M0", "xgb")}
        l2 = [r[k]["triage"]["level2"]["dr_h1"] for k in ("M2", "M0", "xgb")]
        qm, qp = r["M2"], r["M0"]
        def qs(d): return f"{d['calib']['q_calibrated']}/{d['calib']['q_arl0_boot']:.0f}/{d['q']['arl0']:.0f}/{d['q'].get('arl1', float('nan')) if d['q'].get('arl1') is not None else float('nan'):.0f}"
        print(f"{r['rep']:3d} | {P['M2']['aucpr']:9.3f} {P['M0']['aucpr']:6.3f} {P['xgb']['aucpr']:6.3f} | {P['M2']['ece']:7.4f} {P['M0']['ece']:7.4f} {P['xgb']['ece']:7.4f} | {P['M2']['brier']:8.4f} {P['M0']['brier']:7.4f} {P['xgb']['brier']:7.4f} | {l2[0]:.2f} {l2[1]:.2f} {l2[2]:.2f} | {qs(qm):>22s} | {qs(qp):>22s} | {r['M2']['fit']['n_refit']}")
    # paired differences
    for metric in ("aucpr", "ece", "brier"):
        d0 = np.array([r["M2"]["prob"][metric] - r["M0"]["prob"][metric] for r in recs]); dx = np.array([r["M2"]["prob"][metric] - r["xgb"]["prob"][metric] for r in recs])
        print(f"  paired {metric:5s}: M2-M0 mean {d0.mean():+.4f} sd {d0.std(ddof=1):.4f} (M2 better in {(d0 < 0).sum() if metric!='aucpr' else (d0 > 0).sum()}/5) | M2-XGB mean {dx.mean():+.4f} sd {dx.std(ddof=1):.4f} (M2 better in {(dx < 0).sum() if metric!='aucpr' else (dx > 0).sum()}/5)")
    # sd of means
    for k in ("M2", "M0", "xgb"):
        a = np.array([r[k]["prob"]["aucpr"] for r in recs]); e = np.array([r[k]["prob"]["ece"] for r in recs])
        print(f"  {k:3s} AUC-PR mean {a.mean():.3f} sd {a.std(ddof=1):.3f} | ECE mean {e.mean():.4f} sd {e.std(ddof=1):.4f}")
    # charts and sigma rule
    sig = [[(r["M2"]["sigma_rule"][f"level{k}"]["dr_h1"], r["M2"]["sigma_rule"][f"level{k}"]["alarm_share"] * 12) for k in (1, 2, 3)] for r in recs]
    print(f"  sigma L1 dr {np.mean([s[0][0] for s in sig]):.2f} AM {np.mean([s[0][1] for s in sig]):.1f}; L3 dr {np.mean([s[2][0] for s in sig]):.2f} AM {np.mean([s[2][1] for s in sig]):.1f}")
    for c in ("ewma", "cusum"):
        ev = [r["charts"][c]["events"] for r in recs]; rl = [r["charts"][c]["run_length"] for r in recs]
        print(f"  {c:5s} dr1 {np.mean([e['dr_h1'] for e in ev]):.2f} AM {np.mean([e['alarm_share']*12 for e in ev]):.1f} | realised ARL0 {np.mean([x['arl0'] for x in rl]):.0f} (per rep {[round(x['arl0']) for x in rl]}) ARL1 {np.nanmean([x['arl1'] if x.get('arl1') is not None else np.nan for x in rl]):.0f}")
    # q channel detail
    for k in ("M2", "M0"):
        print(f"  q {k}: thr {[r[k]['calib']['q_thr'] for r in recs]} persist {[r[k]['calib']['q_persist'] for r in recs]} boot ARL0 {[round(r[k]['calib']['q_arl0_boot']) for r in recs]} calibrated {[r[k]['calib']['q_calibrated'] for r in recs]} | realised ARL0 {[round(r[k]['q']['arl0']) for r in recs]} stable-alarm units {[r[k]['q']['stable_alarm_units'] for r in recs]} ARL1 {[r[k]['q'].get('arl1') for r in recs]} censored {[r[k]['q'].get('arl1_censored') for r in recs]}")
    # triage L1-L3 M2 vs M0 vs xgb with alarm months and false alarms per item-year
    for k in ("M2", "M0", "xgb"):
        t = [r[k]["triage"] for r in recs]
        print(f"  triage {k:3s}: " + " | ".join(f"L{j} dr {np.mean([x[f'level{j}']['dr_h1'] for x in t]):.2f} AM {np.mean([x[f'level{j}']['alarm_share']*12 for x in t]):.1f} FA/iy {np.mean([x[f'level{j}']['fa_per_itemyear'] for x in t]):.2f} lead {np.nanmean([x[f'level{j}']['lead_med'] if x[f'level{j}']['lead_med'] is not None else np.nan for x in t]):.0f}" for j in (1, 2, 3)))
    cf = [r["M2"]["conformal"] for r in recs]; sv = [r["M2"]["severity"] for r in recs]
    print(f"  conformal cover {np.mean([c['conf_cover'] for c in cf]):.3f} size {np.mean([c['conf_size'] for c in cf]):.2f}; severity CQR cover {np.mean([s['cqr_cover'] for s in sv]):.3f} width {np.mean([s['cqr_width'] for s in sv]):.3f} n_losses {np.mean([s['n_losses'] for s in sv]):.0f}; phi q50 {np.mean([s['phi_q50'] for s in sv]):.2f}")
    print(f"  M2 fits: refit {sum(r['M2']['fit']['n_refit'] for r in recs)}/70 failed {sum(r['M2']['fit']['n_failed'] for r in recs)} rhat_max {max(r['M2']['fit']['rhat_max'] for r in recs)}; M0 failed {sum(r['M0']['fit']['n_failed'] for r in recs)}")

print("\n## pooled over 15 replications (three cells)")
allrecs = [r for reg in order for r in cells[reg]]
for k, name in (("M2", "M2"), ("M0", "Poisson"), ("xgb", "XGBoost")):
    a = np.array([r[k]["prob"]["aucpr"] for r in allrecs]); e = np.array([r[k]["prob"]["ece"] for r in allrecs]); b = np.array([r[k]["prob"]["brier"] for r in allrecs])
    print(f"  {name:8s} AUC-PR {a.mean():.3f} Brier {b.mean():.4f} ECE {e.mean():.4f}")
for metric in ("aucpr", "ece"):
    d0 = np.array([r["M2"]["prob"][metric] - r["M0"]["prob"][metric] for r in allrecs]); dx = np.array([r["M2"]["prob"][metric] - r["xgb"]["prob"][metric] for r in allrecs])
    better0 = (d0 > 0).sum() if metric == "aucpr" else (d0 < 0).sum(); betterx = (dx > 0).sum() if metric == "aucpr" else (dx < 0).sum()
    print(f"  paired {metric}: M2-M0 {d0.mean():+.4f} (M2 better {better0}/15) | M2-XGB {dx.mean():+.4f} (M2 better {betterx}/15)")
# change regimes only
chg = [r for reg in ("gradual", "abrupt") for r in cells[reg]]
for metric in ("ece",):
    d0 = np.array([r["M2"]["prob"][metric] - r["M0"]["prob"][metric] for r in chg]); print(f"  change regimes ECE M2-M0 {d0.mean():+.4f} sd {d0.std(ddof=1):.4f} M2 better {(d0<0).sum()}/10; per rep {np.round(d0,4).tolist()}")
