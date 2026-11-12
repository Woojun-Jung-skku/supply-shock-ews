"""check_cost_definition.py — the cost code of the rare arm (rare_cost.py) applied to the archived Study B records
must reproduce Table 8 of the manuscript (eight policies whose operating points are in results/B/*.jsonl) and the
'All cells' row of Table S21.1. This ties Tables S29.2 and S29.4 to the definitions of Section 5.6.

usage:  python rare_arm/check_cost_definition.py <repo root> <results_rare dir>        (exit code 1 on a mismatch)"""
import glob, importlib.util, json, os, sys
import numpy as np

root = sys.argv[1]
spec = importlib.util.spec_from_file_location("rare_cost", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rare_cost.py"))
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)       # reads <repo root> and <results_rare dir> from argv
import dgp  # noqa: E402  (on the path after rare_cost is loaded)

recs = []
for f in sorted(glob.glob(os.path.join(root, "results", "B", "*.jsonl"))):
    recs += [json.loads(l) for l in open(f) if l.strip()]


def parse(cell):
    p = cell.split("-"); return float(p[1][1:]), float(p[2][2:]), p[3]


pairs = []
for r in recs:
    a, mu, reg = parse(r["cell"])
    N = dgp.simulate(30, 120, mu, a, reg, seed=dgp.seed_of("B", r["cell"], r["rep"]))["N"]
    pairs.append((r, (N[:, 36:118] > 0).sum() / (30 * 82 / 12)))

TABLE8 = {"M2": [5.18, 8.03, 10.47, 12.02, 12.91], "M0": [5.42, 8.15, 10.53, 12.01, 12.91], "xgb": [4.91, 8.76, 10.79, 12.19, 12.95],
          "lstm": [5.05, 8.78, 10.95, 12.18, 13.07], "tft": [4.98, 8.97, 11.07, 12.19, 13.18]}
OTHER = (("sigma-rule", rc.lv_sigma, [5.05, 10.06, 16.96, 30.77, 72.21]), ("EWMA", lambda r: rc.lv_chart(r, "ewma"), [5.56, 9.85, 16.99, 31.27, 74.13]),
         ("CUSUM", lambda r: rc.lv_chart(r, "cusum"), [5.18, 9.47, 16.62, 30.92, 73.83]))
ok = True
print(f"Study B: {len(pairs)} replications, mean E = {np.mean([e for _, e in pairs]):.2f}")
for k, ref in TABLE8.items():
    got = [round(rc.env_mean(pairs, (lambda r, k=k: rc.lv_prob(r, k)), rr)[0], 2) for rr in rc.RS]
    print(f"{k:11s} {got}  {'= Table 8' if got == ref else '!= ' + str(ref)}"); ok &= got == ref
for name, fn, ref in OTHER:
    got = [round(rc.env_mean(pairs, fn, rr)[0], 2) for rr in rc.RS]
    print(f"{name:11s} {got}  {'= Table 8' if got == ref else '!= ' + str(ref)}"); ok &= got == ref
E = np.array([e for _, e in pairs])
s = rc.summary(rc.curve(pairs, lambda r: rc.lv_prob(r, "M2")), np.minimum(rc.GRID * E.mean(), 12.0))
print("M2, all cells:", rc.fmt(s), "| Table S21.1: 30% at r = 5.4; above 10% for 3.1-11.8; break-even ratios 2.6 and 19.4")
ok &= abs(s["band"][0] - 2.6) < 0.06 and abs(s["band"][1] - 19.4) < 0.06 and abs(s["max"] - 29.8) < 0.2
print("reproduced" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
