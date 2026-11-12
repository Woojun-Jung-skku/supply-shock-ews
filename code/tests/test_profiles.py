"""Profile bookkeeping (no fitting): cell counts, variants, per-cell pass replications, origins.
Run: python tests/test_profiles.py"""
import os, sys, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import config as C
import study_a, study_b, study_robust, alarm

EQ = {(10, 60): 0.3, (30, 120): 1.0, (30, 240): 4.0}     # rough cost of one fit relative to (30, 120)


def summary(profile):
    C.apply_profile(profile)
    importlib.reload(study_a)
    cells = study_a.cells()
    assert len(cells) == 108
    fits = eqs = 0
    by = {}
    for c in cells:
        for v in c["variants"]:
            n = study_a.pass_reps(c, C.PASS1_REPS)
            fits += n
            eqs += n * EQ[(c["I"], c["T"])] * (1.15 if v == "horseshoe" else 1.0)
            by[(c["I"], c["T"], v)] = by.get((c["I"], c["T"], v), 0) + n
    nb = sum(C.PASS1_REPS["B_ref" if c["reference"] else "B"] for c in study_b.cells())
    nr = sum(C.PASS1_REPS["R"] * len(c["arms"]) for c in study_robust.cells())
    return dict(fits=fits, eq=round(eqs), by=by, B=nb, R=nr,
                m2=len(alarm.origins_of("M2")), base=len(alarm.origins_of("M0")))


s = summary("oneday")
assert s["fits"] == 108 * 30 * 3 + 36 * 30 + 30 * 3 == 10890, s["fits"]   # hier/none/complete everywhere, horseshoe in abrupt, reference cell +30 x 3
assert s["m2"] == 14 and s["base"] == 7 and s["B"] == 12 * 30 + 30 and s["R"] == 15 * 30

s = summary("colab_min")
assert s["fits"] == 1380 and s["B"] == 130 and s["R"] == 150 and s["m2"] == 14 and s["base"] == 4, s
assert s["by"][(30, 240, "none")] == 30 and s["by"][(30, 240, "horseshoe")] == 30

s = summary("t4_min")
assert s["m2"] == 14 and s["base"] == 4 and s["B"] == 11 * 5 + 10 and s["R"] == 15 * 5, s
assert s["by"][(30, 240, "hier")] == 36 * 3, s["by"]
assert (30, 240, "none") not in s["by"] and (30, 240, "horseshoe") not in s["by"]
assert s["by"][(30, 120, "none")] == 2 * 10 + 20 and s["by"][(30, 120, "complete")] == 2 * 10 + 20   # slice 10, reference 20
assert s["by"][(10, 60, "none")] == 3 * 10 and s["by"][(10, 60, "horseshoe")] == 3 * 10
assert s["by"][(30, 120, "hier")] == 30 * 5 + 5 * 10 + 20 and s["by"][(10, 60, "hier")] == 30 * 5 + 6 * 10
assert s["fits"] == 738, s["fits"]
ref = next(c for c in study_a.cells() if c["reference"])
assert study_a.pass_reps(ref, C.PASS1_REPS) == 20 and ref["slice"] and not ref["long"]
top = dict(A=35, A_ref=50, B=35, B_ref=40, R=35)                       # a top-up round keeps the (30, 240) cap
assert study_a.pass_reps(ref, top) == 50
long = next(c for c in study_a.cells() if c["long"])
assert study_a.pass_reps(long, top) == 3 and study_a.pass_reps(long, C.PASS1_REPS) == 3
sl = next(c for c in study_a.cells() if c["slice"] and not c["reference"])
assert study_a.pass_reps(sl, top) == 35 and study_a.pass_reps(sl, C.PASS1_REPS) == 10
assert study_a.pass_reps(sl, 2) == 10 and study_a.pass_reps(long, 200) == 3
C.B_M2_STRIDE = 2
assert alarm.origins_of("M2") == C.ORIGINS[::2] and len(alarm.origins_of("M2")) == 7
C.B_M2_STRIDE = 1
C.apply_profile("oneday")
print("profiles OK:", {k: v for k, v in s.items() if k != "by"})
