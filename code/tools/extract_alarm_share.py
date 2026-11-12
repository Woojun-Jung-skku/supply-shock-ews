"""extract_alarm_share.py — pull alarm_share, n_onsets, fa, fa_stable, dr_h1..3 for every method and level
out of the Study B JSONL records (results/B/*__all.jsonl) into one replication-level CSV, and check the
overlapping fields against B_replications.csv.
usage: python -I extract_alarm_share.py RESULTS_B_DIR ANALYSIS_DIR OUT_CSV
"""
import sys, os, glob, json
import numpy as np
import pandas as pd

RB, AN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for path in sorted(glob.glob(os.path.join(RB, "B-*__all.jsonl"))):
    for line in open(path):
        rec = json.loads(line)
        row = dict(cell=rec["cell"], rep=rec["rep"], t0=rec["t0"])
        def put(prefix, ev):
            for k in ("dr_h1", "dr_h2", "dr_h3", "n_onsets", "fa_per_itemyear", "fa_per_itemyear_stable", "alarm_share", "lead_med"):
                row[f"{prefix}_{k}"] = ev.get(k)
        for m in ("M2", "M0", "NBnone", "NBcomplete", "xgb", "lstm", "tft"):
            for k in (1, 2, 3):
                put(f"{m}_tri{k}", rec[m]["triage"][f"level{k}"])
        for k in (1, 2):
            put(f"head{k}", rec["M2"]["headline"][f"level{k}"])
        for k in (1, 2, 3):
            put(f"sigma_{k}", rec["M2"]["sigma_rule"][f"level{k}"])
        for m in ("ewma", "cusum"):
            put(m, rec["charts"][m]["events"])
        rel = rec["M2"]["prob"]["reliability"]
        row["n_valid"] = int(sum(b[1] for b in rel))
        row["n_pos"] = int(round(sum(b[1] * b[3] for b in rel)))
        row["base_rate_M2"] = rec["M2"]["prob"]["base_rate"]
        rows.append(row)
d = pd.DataFrame(rows).sort_values(["cell", "rep"]).reset_index(drop=True)
B = pd.read_csv(os.path.join(AN, "B_replications.csv"))
m = B.merge(d, on=["cell", "rep"], suffixes=("", "_j"))
assert len(m) == 65 == len(d)
chk = [("M2_tri2_dr1", "M2_tri2_dr_h1"), ("M2_tri2_fa_stable", "M2_tri2_fa_per_itemyear_stable"), ("xgb_tri1_dr1", "xgb_tri1_dr_h1"),
       ("lstm_tri3_fa_stable", "lstm_tri3_fa_per_itemyear_stable"), ("ewma_dr1", "ewma_dr_h1"), ("cusum_fa_stable", "cusum_fa_per_itemyear_stable"),
       ("sigma_2_dr1", "sigma_2_dr_h1"), ("head2_fa_stable", "head2_fa_per_itemyear_stable")]
for a, b in chk:
    print(f"{a:28s} vs jsonl: max |diff| = {(m[a] - m[b]).abs().max():.6f}")
print("n_valid unique:", d.n_valid.unique(), " n_pos mean:", d.n_pos.mean())
d.to_csv(OUT, index=False)
print("written", OUT, d.shape)
