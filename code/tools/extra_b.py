"""v1.4: numbers behind the §5.3 / §5.5 sentences that Table 5(b) does not show.
usage: python tools/extra_b.py ANALYSIS_DIR RESULTS_DIR"""
import sys, os, json, glob, collections
import numpy as np, pandas as pd

an, res = sys.argv[1], sys.argv[2]
d = pd.read_csv(os.path.join(an, "B_replications.csv"))
pd.set_option("display.width", 200)

print("== per cell, triage level 2: detection within 1 month / false alarms per item-year (stable windows)")
cols = {}
for k in ("M2", "M0", "NBcomplete", "xgb", "lstm", "tft"):
    cols[k + " dr1"] = d.groupby(["alpha", "mu", "regime"])[f"{k}_tri2_dr1"].mean()
    cols[k + " fa"] = d.groupby(["alpha", "mu", "regime"])[f"{k}_tri2_fa_stable"].mean()
t = pd.DataFrame(cols)
print(t.round(3).to_string())
print("\n== cells where a baseline's L2 detection is >= M2's with fewer false alarms")
for k in ("M0", "NBcomplete", "xgb", "lstm", "tft"):
    w = t[(t[k + " dr1"] >= t["M2 dr1"] - 0.005) & (t[k + " fa"] < t["M2 fa"] - 0.01)]
    print(k, len(w), [f"{a}/{m}/{r}" for a, m, r in w.index])

print("\n== AUC-PR difference to M2 by cell (mean over replications)")
for k in ("M0", "NBnone", "NBcomplete", "xgb", "lstm", "tft"):
    x = (d[f"{k}_aucpr"] - d["M2_aucpr"]).groupby([d.alpha, d.mu, d.regime]).mean()
    print(f"{k:11s} min {x.min():+.3f} max {x.max():+.3f}  cells > 0: {int((x > 0).sum())}")

print("\n== conformal coverage before / after t0 (M2, M0, NB): mean over replications")
for k in ("M2", "M0", "NBnone", "NBcomplete"):
    print(k, round(d[f"{k}_conf_cover_pre"].mean(), 3), round(d[f"{k}_conf_cover_post"].mean(), 3),
          "size", round(d[f"{k}_conf_size"].mean(), 2))

print("\n== M2 with / without replications that have a failed origin fit")
f = d["M2_fit_failed_origins"] > 0
print("replications with a failed M2 origin fit:", int(f.sum()), "of", len(d))
print("failed origin fits by cell:", d.loc[f].groupby(["alpha", "mu", "regime"]).size().to_dict())
for c in ("M2_aucpr", "M2_brier", "M2_ece", "M2_tri2_dr1", "M2_tri2_fa_stable", "M2_conf_cover"):
    print(f"{c:22s} all {d[c].mean():.3f}  without {d.loc[~f, c].mean():.3f}")
print("paired differences (baseline - M2), all replications vs without those replications:")
for k in ("M0", "NBnone", "NBcomplete", "xgb", "lstm", "tft"):
    for m in ("aucpr", "ece"):
        x = d[f"{k}_{m}"] - d[f"M2_{m}"]
        print(f"  {k:11s} {m:6s} all {x.mean():+.4f}  without {x[~f].mean():+.4f}")

print("\n== lead time (median months, level 2) and detection within 3 months")
for k in ("M2", "xgb", "lstm", "tft"):
    print(k, round(d[f"{k}_tri2_lead"].median(), 1), round(d[f"{k}_tri2_dr3"].mean(), 3))

# final-failure reasons of the Study A fits (last attempt)
print("\n== Study A: reasons of the fits that failed after the refit (last attempt)")
why = collections.Counter(); n = 0
for p in glob.glob(os.path.join(res, "A", "*.jsonl")):
    for line in open(p):
        r = json.loads(line)
        if not r.get("failed"):
            continue
        n += 1
        a = (r.get("fit") or {}).get("attempts") or r.get("attempts") or []
        last = a[-1] if a else r
        why["rhat"] += int((last.get("n_rhat_over") or 0) > 0 or (last.get("rhat_max") or 0) >= 1.01)
        why["ess"] += int(min(last.get("ess_bulk_min") or 1e9, last.get("ess_tail_min") or 1e9) < 400)
        why["div"] += int((last.get("n_div") or 0) > 0)
print("failed", n, dict(why))
