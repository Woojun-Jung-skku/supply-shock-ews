"""v1.4: write a plan file (worker slot -> ordered unit list) for run_mac.py --plan.
usage: python tools/make_plan.py plan.json [profile]"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
C.apply_profile(sys.argv[2] if len(sys.argv) > 2 else "t4_min")
import study_a, study_b, study_robust, run
B = {c["id"]: c for c in study_b.cells()}
plan = {"0": [], "1": []}
# worker 0: Study B (reference cell first); worker 1: Study A (30,120) -> (10,60) -> slices -> robustness -> (30,240)
ref_b = next(c["id"] for c in study_b.cells() if c["reference"])
plan["0"] = [["B", ref_b]] + [["B", c] for c in B if c != ref_b]
R = study_robust.cells()
pipes = [["R", c["id"], a] for c in R for a in c["arms"] if C.R_ALARM_HAWKES or "hawkes" not in c["kw"]]
rec = [["R", c["id"], a] for c in R for a in c["arms"] if not (C.R_ALARM_HAWKES or "hawkes" not in c["kw"])]
A = {}
for size, variant in run.A_ORDER:
    for c in study_a.cells(size):
        if variant in c["variants"]:
            A.setdefault((size, variant), []).append(["A", c["id"], variant])
ref_a = next(c["id"] for c in study_a.cells() if c["reference"])
first = [u for u in A[((30, 120), "hier")] if u[1] == ref_a] + [u for u in A[((30, 120), "hier")] if u[1] != ref_a]
plan["1"] = (first + A[((10, 60), "hier")]
             + [u for (sz, v), us in A.items() if v != "hier" and sz != (30, 240) for u in us]
             + pipes + rec + A[((30, 240), "hier")]
             + [u for (sz, v), us in A.items() if v != "hier" and sz == (30, 240) for u in us])
nA = sum(len(v) for v in A.values())
json.dump(plan, open(sys.argv[1], "w"), indent=0)
print({k: len(v) for k, v in plan.items()}, "units;", "A", nA, "R pipelines", len(pipes), "R recovery-only", len(rec))
