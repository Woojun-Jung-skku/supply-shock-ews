"""v1.4: progress of a plan over one or more result folders (done / stage target per study)."""
import sys, os, json, glob, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
prof = os.environ.get("PROFILE", "t4_min"); C.apply_profile(prof)
reps = int(os.environ.get("REPS", 0))
if reps:
    C.PASS1_REPS = dict(A=reps, A_ref=2 * reps, B=reps, B_ref=2 * reps, R=reps)
import study_a, study_b, study_robust, store
dirs = sys.argv[1:]
def n_done(study, cell, var):
    s = set()
    for d in dirs:
        s |= set(store.load(d, study, cell, var).keys())
    return len(s)
tot = collections.defaultdict(lambda: [0, 0])
for c in study_a.cells():
    for v in c["variants"]:
        n = study_a.pass_reps(c, C.PASS1_REPS); t = tot[f"A {c['T']}"]; t[0] += min(n_done("A", c["id"], v), n); t[1] += n
for c in study_b.cells():
    n = min(C.PASS1_REPS["B_ref" if c["reference"] else "B"], c["reps"]); t = tot["B"]; t[0] += min(n_done("B", c["id"], "all"), n); t[1] += n
for c in study_robust.cells():
    for a in c["arms"]:
        n = min(C.PASS1_REPS["R"], c["reps"]); t = tot["R"]; t[0] += min(n_done("R", c["id"], a), n); t[1] += n
print(" | ".join(f"{k}: {v[0]}/{v[1]}" for k, v in sorted(tot.items())))
