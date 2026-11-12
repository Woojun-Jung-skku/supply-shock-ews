"""rare_cost.py — expected cost of the alarm policies in the rare arm, with the definitions of Section 5.6
(Table 8, Tables S21.1 and S21.2) of the manuscript:

  EC_m(r)  = min over levels k of  mean_j [ AM_mkj + r (1 - DR_mkj) E_j ]      (lower envelope of the level-wise mean-cost lines)
  trivial  = min(r * mean_j E_j, 12)                                            (never act / always act, at the subset mean E)
  value    = 1 - EC_m(r) / trivial ;  band = {r : EC_m(r) < trivial}

for each cell (five replications) and for the three cells pooled (fifteen). Paired statements compare, replication by
replication, the cost of each method at the level that is cheapest on average at that r (as in Table S20).

usage:  python rare_cost.py <repo root> <results_rare dir> [--json out.json]
E_j (shock item-months per item-year over the decision months 37-118) is recomputed from the seeds with code/dgp.py.
For comparison the script also prints the figures under the other convention (per-replication lower envelope and
per-replication trivial policy), which is NOT the one used in the manuscript."""
import json, glob, os, sys
import numpy as np

root, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(root, "code"))
import dgp  # noqa: E402

RS = (2, 5, 10, 20, 50)
GRID = np.exp(np.linspace(0.0, np.log(300.0), 60001))      # r = 1 ... 300
ORDER = ("stable", "gradual", "abrupt")


def E_of(cell, rep, reg):
    s = dgp.simulate(100, 120, 0.02, 2.0, reg, seed=dgp.seed_of("B", cell, rep))
    return (s["N"][:, 36:118] > 0).sum() / (100 * 82 / 12)


def lv_prob(r, k):
    return [(r[k]["triage"][f"level{j}"]["alarm_share"] * 12, r[k]["triage"][f"level{j}"]["dr_h1"]) for j in (1, 2, 3)]


def lv_sigma(r):
    return [(r["M2"]["sigma_rule"][f"level{j}"]["alarm_share"] * 12, r["M2"]["sigma_rule"][f"level{j}"]["dr_h1"]) for j in (1, 2, 3)]


def lv_chart(r, c):
    return [(r["charts"][c]["events"]["alarm_share"] * 12, r["charts"][c]["events"]["dr_h1"])]


METHODS = (("M2 (proposed)", lambda r: lv_prob(r, "M2")), ("Poisson M0", lambda r: lv_prob(r, "M0")),
           ("XGBoost", lambda r: lv_prob(r, "xgb")), ("Sigma-rule (plug-in SRI)", lv_sigma),
           ("EWMA", lambda r: lv_chart(r, "ewma")), ("CUSUM", lambda r: lv_chart(r, "cusum")))


def load():
    cells = {}
    for f in sorted(glob.glob(os.path.join(out, "B", "*__all.jsonl"))):
        recs = sorted((json.loads(l) for l in open(f) if l.strip()), key=lambda r: r["rep"])
        reg = recs[0]["cell"].split("-")[3]
        cells[reg] = [(r, E_of(r["cell"], r["rep"], reg)) for r in recs]
    return cells


def arrays(pairs, fn):
    """AM[k, j], DR[k, j] for the levels k of a method and the replications j of a subset."""
    lv = [fn(r) for r, _ in pairs]
    AM = np.array([[l[k][0] for l in lv] for k in range(len(lv[0]))])
    DR = np.array([[l[k][1] for l in lv] for k in range(len(lv[0]))])
    return AM, DR


def lines(pairs, fn, r):
    """cost[k, j] at ratio r."""
    AM, DR = arrays(pairs, fn)
    E = np.array([e for _, e in pairs])
    return AM + r * (1.0 - DR) * E[None, :]


def env_mean(pairs, fn, r):          # the manuscript's definition
    c = lines(pairs, fn, r).mean(axis=1)
    return float(c.min()), int(c.argmin())


def env_rep(pairs, fn, r):           # per-replication envelope (not used in the manuscript)
    return float(lines(pairs, fn, r).min(axis=0).mean())


def curve(pairs, fn, per_rep=False):
    AM, DR = arrays(pairs, fn)
    E = np.array([e for _, e in pairs])
    c = AM[:, :, None] + GRID[None, None, :] * ((1.0 - DR) * E[None, :])[:, :, None]      # level, rep, r
    return c.min(axis=0).mean(axis=0) if per_rep else c.mean(axis=1).min(axis=0)


def summary(ec, triv):
    val = 1.0 - ec / triv
    pays = val > 0
    if not pays.any():
        return dict(band=None, max=float(val.max() * 100), at=float(GRID[val.argmax()]), over10=None)
    g10 = GRID[val > 0.10]
    return dict(band=(float(GRID[pays].min()), float(GRID[pays].max())), max=float(val.max() * 100), at=float(GRID[val.argmax()]),
                over10=(float(g10.min()), float(g10.max())) if g10.size else None)


def fmt(s):
    if s["band"] is None:
        return f"band none; max {s['max']:.1f}% at r = {s['at']:.1f}"
    o = f"{s['over10'][0]:.1f}-{s['over10'][1]:.1f}" if s["over10"] else "none"
    return f"band {s['band'][0]:.1f}-{s['band'][1]:.1f}; max {s['max']:.1f}% at r = {s['at']:.1f}; >10% for r in {o}"


def block(pairs, title, store):
    E = np.array([e for _, e in pairs]); Eb = float(E.mean())
    print(f"\n=== {title}: n = {len(pairs)}, mean E = {Eb:.3f} shock item-months per item-year, 12/E = {12 / Eb:.1f}")
    print(f"{'Method':26s} " + " ".join(f"{'r = ' + str(r):>8s}" for r in RS) + "   level at each r | band, largest saving against min(rE, 12)")
    trivA = np.minimum(GRID * Eb, 12.0)
    trivB = np.array([np.minimum(g * E, 12.0).mean() for g in GRID])
    st = {"E": Eb, "methods": {}}
    for name, fn in METHODS:
        vals = [env_mean(pairs, fn, r) for r in RS]
        nl = arrays(pairs, fn)[0].shape[0]
        lev = "".join(str(nl - k if nl == 3 else 1) for _, k in vals) if nl == 3 else "-----"
        # levels are stored level1, level2, level3 -> index 0, 1, 2
        lev = "/".join(f"L{k + 1}" for _, k in vals) if nl == 3 else "one level"
        sA = summary(curve(pairs, fn), trivA)
        sB = summary(curve(pairs, fn, per_rep=True), trivB)
        print(f"{name:26s} " + " ".join(f"{v:8.2f}" for v, _ in vals) + f"   {lev:22s} | {fmt(sA)}")
        st["methods"][name] = dict(cost={r: v for r, (v, _) in zip(RS, vals)}, level={r: k + 1 for r, (_, k) in zip(RS, vals)},
                                   paper=sA, per_rep=sB, cost_per_rep_env={r: env_rep(pairs, fn, r) for r in RS})
    print(f"{'Never act':26s} " + " ".join(f"{r * Eb:8.2f}" for r in RS))
    print(f"{'Always act':26s} " + " ".join(f"{12.0:8.2f}" for _ in RS))
    # level switching of M2 along the grid
    AM, DR = arrays(pairs, METHODS[0][1])
    best = (AM[:, :, None] + GRID[None, None, :] * ((1 - DR) * E[None, :])[:, :, None]).mean(axis=1).argmin(axis=0)
    sw = [(int(best[i - 1]) + 1, int(best[i]) + 1, float(GRID[i])) for i in range(1, GRID.size) if best[i] != best[i - 1]]
    print("M2 level switches: " + "; ".join(f"L{a} -> L{b} at r = {r:.1f}" for a, b, r in sw))
    st["switch"] = sw
    # paired comparisons at the level that is cheapest on average
    for other in ("Poisson M0", "XGBoost"):
        fo = dict(METHODS)[other]
        row = []
        for r in RS:
            cm, km = env_mean(pairs, METHODS[0][1], r); co, ko = env_mean(pairs, fo, r)
            a = lines(pairs, METHODS[0][1], r)[km]; b = lines(pairs, fo, r)[ko]
            row.append((r, 100 * (co - cm) / co, int((a < b).sum()), len(a)))
        print(f"M2 against {other:11s}: " + "; ".join(f"r = {r}: {d:+.1f}% ({w}/{n})" for r, d, w, n in row) + "   [+ = M2 cheaper; (replications in which M2 is cheaper)]")
        st.setdefault("paired", {})[other] = row
    print("-- other convention (per-replication envelope and per-replication trivial policy; not the manuscript's):")
    for name, fn in METHODS:
        print(f"   {name:26s} " + " ".join(f"{env_rep(pairs, fn, r):8.2f}" for r in RS) + f"   | {fmt(st['methods'][name]['per_rep'])}")
    store[title] = st


def main():
    cells = load()
    store = {}
    for reg in ORDER:
        block(cells[reg], reg, store)
    block([p for reg in ORDER for p in cells[reg]], "pooled", store)
    print("\nper-replication E_j:", {reg: [round(float(e), 3) for _, e in cells[reg]] for reg in ORDER})
    if "--json" in sys.argv:
        json.dump(store, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
