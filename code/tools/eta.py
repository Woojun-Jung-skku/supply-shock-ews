"""v1.4: remaining time per worker slot of a plan, from the measured seconds of finished replications.
usage: REPS=3 python tools/eta.py OUT PLAN [slots, e.g. 0,1]"""
import glob, json, os, sys, time
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
C.apply_profile(os.environ.get("PROFILE", "t4_min"))
reps = int(os.environ.get("REPS", 0))
if reps:
    C.PASS1_REPS = dict(A=reps, A_ref=2 * reps, B=reps, B_ref=2 * reps, R=reps)
import study_a, study_b, study_robust, store

out, plan_path = sys.argv[1], sys.argv[2]
slots = sys.argv[3].split(",") if len(sys.argv) > 3 else None
plan = json.load(open(plan_path))
cb = {c["id"]: c for c in study_b.cells()}
ca = {c["id"]: c for c in study_a.cells()}
cr = {c["id"]: c for c in study_robust.cells()}

# measured seconds per replication
sec = defaultdict(list)
for p in glob.glob(os.path.join(out, "A", "*.jsonl")):
    v = os.path.basename(p)[:-6].split("__")[1]
    T = ca[os.path.basename(p).split("__")[0]]["T"] if os.path.basename(p).split("__")[0] in ca else None
    for line in open(p):
        r = json.loads(line)
        sec[("A", T, v)].append(r.get("seconds", np.nan))
for p in glob.glob(os.path.join(out, "B", "*.jsonl")):
    for line in open(p):
        r = json.loads(line)
        sec[("B",)].append(r.get("seconds_batch", np.nan))
for p in glob.glob(os.path.join(out, "R", "*.jsonl")):
    arm = os.path.basename(p)[:-6].split("__")[1]
    for line in open(p):
        r = json.loads(line)
        kind = "rec" if r.get("M2") is None else "pipe"
        sec[("R", kind)].append(r.get("seconds_batch", r.get("recovery_fit", {}).get("seconds", np.nan)))
DEFAULT = {("A", 60): 25.0, ("A", 120): 150.0, ("A", 240): 400.0, ("B",): 1700.0, ("R", "pipe"): 1100.0, ("R", "rec"): 150.0}


def est(key, fallback):
    v = [x for x in sec.get(key, []) if np.isfinite(x)]
    return float(np.mean(v)) if len(v) >= 3 else fallback


def todo_seconds(u):
    if u[0] == "B":
        c = cb[u[1]]
        n = min(C.PASS1_REPS["B_ref" if c["reference"] else "B"], c["reps"])
        left = [r for r in range(n) if r not in store.load(out, "B", c["id"], "all")]
        return len(left), len(left) * est(("B",), DEFAULT[("B",)])
    if u[0] == "A":
        c = ca[u[1]]
        left = [r for r in range(study_a.pass_reps(c, C.PASS1_REPS)) if r not in store.load(out, "A", c["id"], u[2])]
        base = est(("A", c["T"], "hier"), DEFAULT[("A", c["T"])])
        return len(left), len(left) * est(("A", c["T"], u[2]), base * (1.3 if u[2] != "hier" else 1.0))
    c = cr[u[1]]
    left = [r for r in range(min(C.PASS1_REPS["R"], c["reps"])) if r not in store.load(out, "R", c["id"], u[2])]
    kind = "pipe" if (C.R_ALARM_HAWKES or "hawkes" not in c["kw"]) else "rec"
    return len(left), len(left) * est(("R", kind), DEFAULT[("R", kind)])


now = time.time()
for k, units in plan.items():
    if slots and k not in slots:
        continue
    n = s = 0
    by = defaultdict(lambda: [0, 0.0])
    for u in units:
        a, b = todo_seconds(u)
        n += a; s += b
        key = u[0] + (f"{ca[u[1]]['T']}" if u[0] == "A" else "")
        by[key][0] += a; by[key][1] += b
    parts = ", ".join(f"{kk} {v[0]} ({v[1] / 3600:.1f} h)" for kk, v in by.items() if v[0])
    print(f"slot {k}: {n} replications, {s / 3600:.1f} h -> {time.strftime('%m-%d %H:%M', time.localtime(now + s))} | {parts}")
print("per-replication seconds used:", {"/".join(map(str, k)): round(est(k, DEFAULT.get(k, np.nan))) for k in sorted(sec, key=str)})
