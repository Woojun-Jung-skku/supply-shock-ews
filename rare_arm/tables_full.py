"""Table-6(a)/(b)-style rows for the rare cells, per cell and pooled over the 15 replications (Table S29.1).
The expected cost (Table S29.2) is computed by rare_cost.py.
usage: python tables_full.py <repo root> <results_rare dir>"""
import json, glob, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
root, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(root, "code"))
import numpy as np, config as C; C.apply_profile("t4_min"); import dgp
def E_of(cell, rep, reg):
    s = dgp.simulate(100, 120, 0.02, 2.0, reg, seed=dgp.seed_of("B", cell, rep)); return (s["N"][:, 36:118] > 0).sum() / (100 * 82 / 12)
cells = {}
for f in sorted(glob.glob(os.path.join(out, "B", "*__all.jsonl"))):
    recs = sorted([json.loads(l) for l in open(f)], key=lambda r: r["rep"]); reg = recs[0]["cell"].split("-")[3]
    cells[reg] = [(r, E_of(r["cell"], r["rep"], reg)) for r in recs]
order = ["stable", "gradual", "abrupt"]
def lv_prob(r, k): return [(r[k]["triage"][f"level{j}"]["dr_h1"], r[k]["triage"][f"level{j}"]["alarm_share"] * 12) for j in (1, 2, 3)]
def lv_sigma(r): return [(r["M2"]["sigma_rule"][f"level{j}"]["dr_h1"], r["M2"]["sigma_rule"][f"level{j}"]["alarm_share"] * 12) for j in (1, 2, 3)]
def lv_chart(r, c): return [(r["charts"][c]["events"]["dr_h1"], r["charts"][c]["events"]["alarm_share"] * 12)]
def table6(pairs, title):
    print(f"\n### Table-6(a) style, level 2: {title} (n = {len(pairs)})")
    print(f"{'Method':24s} {'AUC-PR':>7s} {'Brier':>7s} {'ECE':>7s} {'Det≤1m':>7s} {'Det≤3m':>7s} {'FA/item-yr':>10s} {'Alarm-mo/item-yr':>16s} {'Lead med (IQR)':>18s}")
    for name, k in (("M2 (proposed)", "M2"), ("Poisson M0", "M0"), ("XGBoost", "xgb")):
        P = [r[k]["prob"] for r, _ in pairs]; T = [r[k]["triage"]["level2"] for r, _ in pairs]
        med = np.nanmean([t["lead_med"] if t["lead_med"] is not None else np.nan for t in T]); q25 = np.nanmean([t["lead_q25"] if t["lead_q25"] is not None else np.nan for t in T]); q75 = np.nanmean([t["lead_q75"] if t["lead_q75"] is not None else np.nan for t in T])
        print(f"{name:24s} {np.mean([p['aucpr'] for p in P]):7.3f} {np.mean([p['brier'] for p in P]):7.3f} {np.mean([p['ece'] for p in P]):7.3f} {np.mean([t['dr_h1'] for t in T]):7.2f} {np.mean([t['dr_h3'] for t in T]):7.2f} {np.mean([t['fa_per_itemyear'] for t in T]):10.2f} {np.mean([t['alarm_share']*12 for t in T]):16.1f} {med:8.1f} ({q25:.1f}–{q75:.1f})")
    T = [r["M2"]["sigma_rule"]["level2"] for r, _ in pairs]
    med = np.nanmean([t["lead_med"] if t["lead_med"] is not None else np.nan for t in T]); q25 = np.nanmean([t["lead_q25"] if t["lead_q25"] is not None else np.nan for t in T]); q75 = np.nanmean([t["lead_q75"] if t["lead_q75"] is not None else np.nan for t in T])
    print(f"{'Sigma-rule, level 2':24s} {'–':>7s} {'–':>7s} {'–':>7s} {np.mean([t['dr_h1'] for t in T]):7.2f} {np.mean([t['dr_h3'] for t in T]):7.2f} {np.mean([t['fa_per_itemyear'] for t in T]):10.2f} {np.mean([t['alarm_share']*12 for t in T]):16.1f} {med:8.1f} ({q25:.1f}–{q75:.1f})")
    for c in ("ewma", "cusum"):
        T = [r["charts"][c]["events"] for r, _ in pairs]
        med = np.nanmean([t["lead_med"] if t["lead_med"] is not None else np.nan for t in T]); q25 = np.nanmean([t["lead_q25"] if t["lead_q25"] is not None else np.nan for t in T]); q75 = np.nanmean([t["lead_q75"] if t["lead_q75"] is not None else np.nan for t in T])
        print(f"{c.upper():24s} {'–':>7s} {'–':>7s} {'–':>7s} {np.mean([t['dr_h1'] for t in T]):7.2f} {np.mean([t['dr_h3'] for t in T]):7.2f} {np.mean([t['fa_per_itemyear'] for t in T]):10.2f} {np.mean([t['alarm_share']*12 for t in T]):16.1f} {med:8.1f} ({q25:.1f}–{q75:.1f})")
    print(f"\n### Table-6(b) style: {title}")
    for name, k in (("q channel, M2 (proposed)", "M2"), ("q channel, Poisson M0", "M0")):
        a0 = np.array([r[k]["q"]["arl0"] for r, _ in pairs]); a1 = [r[k]["q"].get("arl1") for r, _ in pairs]; a1 = np.array([x for x in a1 if x is not None], dtype=float)
        cal = sum(r[k]["calib"]["q_calibrated"] for r, _ in pairs); boot = np.array([r[k]["calib"]["q_arl0_boot"] for r, _ in pairs])
        s = f"{name:26s} realised ARL0 median {np.median(a0):5.0f} (range {a0.min():.0f}–{a0.max():.0f}; bootstrap-predicted {boot.min():.0f}–{boot.max():.0f}; calibrated {cal}/{len(pairs)})"
        if a1.size: s += f" | ARL1 mean {a1.mean():.1f} (range {a1.min():.0f}–{a1.max():.0f})"
        if k == "M2": s += f" | set coverage {np.mean([r['M2']['conformal']['conf_cover'] for r,_ in pairs]):.3f} size {np.mean([r['M2']['conformal']['conf_size'] for r,_ in pairs]):.2f}"
        print(s)
    for c in ("ewma", "cusum"):
        rl = [r["charts"][c]["run_length"] for r, _ in pairs]; a0 = np.array([x["arl0"] for x in rl]); a1 = np.array([x["arl1"] for x in rl if x.get("arl1") is not None], dtype=float)
        print(f"{c.upper():26s} realised ARL0 median {np.median(a0):5.0f} (range {a0.min():.0f}–{a0.max():.0f})" + (f" | ARL1 mean {a1.mean():.1f}" if a1.size else ""))
for reg in order: table6(cells[reg], reg)
allp = [p for reg in order for p in cells[reg]]; table6(allp, "pooled over the three rare cells")
print("\nE per cell:", {reg: round(float(np.mean([E for _, E in cells[reg]])), 3) for reg in order}, "pooled", round(float(np.mean([E for _, E in allp])), 3))
print("prevalence per cell:", {reg: round(float(np.mean([r['M2']['prob']['base_rate'] for r, _ in cells[reg]])), 3) for reg in order})
