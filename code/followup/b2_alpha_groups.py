"""B2 — dispersion that differs across groups (manuscript Section 5.2 (ii), Table S13).

Reference cell (gradual, mu-bar = 0.2, I = 30, T = 120, three groups of ten items), replications 0..4, with the
group dispersions alpha'_g set to (1, 2, 4) ('spread2') and (0.5, 2, 8) ('spread4') instead of a common value;
everything else as in Table 4 (dgp.simulate with a per-group alpha). Fits: the hierarchical model (group-level
alpha'_g with the hyperprior (20)) and, for comparison, complete pooling (one alpha', u = 0); sampler, convergence
rules and refit as in Study A. Writes <out>/B2_alpha_groups.jsonl (one line per fit, metrics.recovery_q, truth
alpha'_g) and B2_alpha_groups_summary.txt (per arm and group: posterior median of alpha'_g, 90% coverage, log width;
tau_alpha and m_alpha posteriors; beta, w, log C and trend recovery).
usage: python followup/b2_alpha_groups.py OUT_DIR [arms]   arms e.g. hier:spread2 complete:spread2 hier:spread4 complete:spread4
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as C
C.apply_profile("t4_min")
import dgp
import models as Mo
import nuts_gpu as NG
import metrics as MT
import study_a

REF = [c for c in study_a.cells() if c["reference"]][0]
REPS = (0, 1, 2, 3, 4)
SPREADS = {"spread2": (1.0, 2.0, 4.0), "spread4": (0.5, 2.0, 8.0)}


def simulate(spread, r):
    s = dgp.simulate(REF["I"], REF["T"], float(REF["mu"]), SPREADS[spread], REF["regime"], seed=dgp.seed_of("B2", spread, r))
    s["uid"] = f"B2-{spread}:{r}"
    return s


def main():
    out = sys.argv[1]
    arms = sys.argv[2:] or ["hier:spread2", "complete:spread2", "hier:spread4", "complete:spread4"]
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "B2_alpha_groups.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            rec = json.loads(line); done.add((rec["variant"], rec["arm"], rec["rep"]))
    log = NG.LOG[0]
    I, T = REF["I"], REF["T"]
    cap = Mo.mcap(I, T, C.K, C.MISS_DEFAULT)
    for arm in arms:
        variant, spread = arm.split(":")
        todo = [r for r in REPS if (variant, spread, r) not in done]
        if not todo:
            continue
        sims = [simulate(spread, r) for r in todo]
        ds = [Mo.prep(s, T, "latent", cap) for s in sims]
        fitter = NG.BatchNUTS(Mo.ews_model, study_a.VARIANT_SPEC[variant], ds[0], post_fn=study_a.post_fn,
                              name=f"B2-{variant}-{spread}-I{I}-T{T}")
        seeds = [dgp.seed_of("B2-fit", spread, variant, r) % 2 ** 31 for r in todo]
        t0 = time.time()
        res, info, _ = fitter.fit(ds, seeds, batch=len(ds))
        host = {k: (np.asarray(v) if not isinstance(v, dict) else {kk: np.asarray(vv) for kk, vv in v.items()})
                for k, v in res.items()}
        with open(path, "a") as f:
            for j, r in enumerate(todo):
                q = {k: {kk: vv[j] for kk, vv in v.items()} for k, v in host.items() if k != "cp"}
                tr = sims[j]["truth"]
                rec = dict(rep=r, cell=REF["id"], variant=variant, arm=spread, alpha_g=list(SPREADS[spread]), design=C.DESIGN,
                           **{k: v for k, v in info[j].items() if k != "attempts"})
                rec["truth"] = dict(alpha=[float(a) for a in np.atleast_1d(tr["alpha"])], kappa_w=tr["kappa_w"])
                rec["recovery"] = MT.recovery_q(q, tr, variant)
                f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)) + "\n")
        log(f"B2 {arm}: {len(todo)} fits in {time.time() - t0:.0f}s, failed {sum(i['failed'] for i in info)}, refit {sum(i['n_refit'] for i in info)}")
        NG.clear_done()
    summarise(path, out)


def summarise(path, out):
    recs = [json.loads(l) for l in open(path)]
    lines = ["B2 dispersion by group, reference cell (gradual, mu-bar 0.2, I 30, T 120), replications 0-4; truth beta (0.4, 0.3, 0.2), w (0.5, 0.3), kappa_w 0.8"]
    for spread, truth in SPREADS.items():
        for variant in ("hier", "complete"):
            rr = [x for x in recs if x["variant"] == variant and x["arm"] == spread]
            if not rr:
                continue
            a50 = np.array([np.atleast_1d(x["recovery"]["alpha"]["q50"]) for x in rr])
            a05 = np.array([np.atleast_1d(x["recovery"]["alpha"]["q05"]) for x in rr])
            a95 = np.array([np.atleast_1d(x["recovery"]["alpha"]["q95"]) for x in rr])
            tr = np.array(truth)
            lines.append(f"\n== {variant} | alpha'_g truth {truth} | n = {len(rr)}, failed {sum(x['failed'] for x in rr)}, refit {sum(x['n_refit'] for x in rr)}")
            if a50.shape[1] == len(tr):
                for g in range(len(tr)):
                    cov = ((a05[:, g] <= tr[g]) & (tr[g] <= a95[:, g])).mean()
                    lines.append(f"  group {g}: truth {tr[g]:g}  median (geo mean over reps) {np.exp(np.log(a50[:, g]).mean()):.2f}  range {a50[:, g].min():.2f}-{a50[:, g].max():.2f}  90% coverage {cov:.2f}  log width {np.mean(np.log(a95[:, g]) - np.log(a05[:, g])):.2f}")
                order = np.mean([np.all(np.diff(a) > 0) for a in a50])
                lines.append(f"  ordering alpha'_0 < alpha'_1 < alpha'_2 recovered in {order:.0%} of replications")
            else:
                cov = ((a05[:, :1] <= tr[None, :]) & (tr[None, :] <= a95[:, :1])).mean(1)
                lines.append(f"  single alpha': median (geo mean over reps) {np.exp(np.log(a50[:, 0]).mean()):.2f} range {a50[:, 0].min():.2f}-{a50[:, 0].max():.2f}; covers the three truths in {cov.mean():.2f} of cases; "
                             f"geometric mean of the truths {np.exp(np.log(tr).mean()):.2f}, harmonic {len(tr)/np.sum(1/tr):.2f}")
            for site in ("m_alpha", "tau_alpha"):
                if site in rr[0]["recovery"]:
                    m50 = np.mean([x["recovery"][site]["q50"] for x in rr]); m05 = np.mean([x["recovery"][site]["q05"] for x in rr]); m95 = np.mean([x["recovery"][site]["q95"] for x in rr])
                    lines.append(f"  {site}: posterior median {m50:.3f} (mean 5% {m05:.3f}, 95% {m95:.3f})" + (f"; truth sd of log alpha'_g {np.std(np.log(tr)):.3f}, mean {np.mean(np.log(tr)):.3f}" if site == "tau_alpha" else ""))
            bt, wt, kt = np.array(C.BETA), np.array(C.W_TRUE), C.KAPPA_TRUE
            b50 = np.array([x["recovery"]["beta"]["q50"] for x in rr]); b05 = np.array([x["recovery"]["beta"]["q05"] for x in rr]); b95 = np.array([x["recovery"]["beta"]["q95"] for x in rr])
            w50 = np.array([x["recovery"]["w"]["q50"] for x in rr]); w05 = np.array([x["recovery"]["w"]["q05"] for x in rr]); w95 = np.array([x["recovery"]["w"]["q95"] for x in rr])
            k05 = np.array([x["recovery"]["kappa_w"]["q05"] for x in rr]); k95 = np.array([x["recovery"]["kappa_w"]["q95"] for x in rr]); k50 = np.array([x["recovery"]["kappa_w"]["q50"] for x in rr])
            lines.append(f"  beta bias {(b50 - bt).mean():+.3f} cover {((b05 <= bt) & (bt <= b95)).mean():.2f}; w median ({', '.join(f'{v:.2f}' for v in w50.mean(0))}) cover {((w05 <= wt) & (wt <= w95)).mean():.2f}; kappa_w median {k50.mean():.2f} cover {((k05 <= kt) & (kt <= k95)).mean():.2f}")
            lines.append(f"  log C rmse {np.mean([x['recovery']['logC_summary']['rmse'] for x in rr]):.3f} cover {np.mean([x['recovery']['logC_summary']['cover'] for x in rr]):.2f}; m_t cover {np.mean([x['recovery']['m_t']['cover'] for x in rr]):.2f} rmse {np.mean([x['recovery']['m_t']['rmse'] for x in rr]):.3f}")
    txt = "\n".join(lines); print(txt)
    open(os.path.join(out, "B2_alpha_groups_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
