"""A1 — prior sensitivity of the hierarchical model (manuscript Section 5.4, Table S6).

Reference cell (gradual, alpha' = 2, mu-bar = 0.2, I = 30, T = 120), replications 0..4 with the Study A seeds
(the same data as the existing hierarchical fits, which are the baseline arm), refitted under five
alternative priors:
  m_alpha_log1   dispersion hyperprior centred away from the truth: m_alpha ~ N(log 1, 1)   (eq. 20)
  m_alpha_log5                                                     m_alpha ~ N(log 5, 1)
  innov_x2       trend-innovation scales doubled: sigma_eta, tau_zeta ~ half-N(0, 0.2)       (eq. 21)
  innov_x5       ... times five: half-N(0, 0.5)
  ahp_reversed   AHP prior pointing the wrong way: w_AHP = (0.375, 0.625) instead of (0.625, 0.375) (eq. 19)
Everything else (data, sampler, convergence rules, refit) as in Study A. Writes <out>/A1_priors.jsonl
(one line per fit, metrics.recovery_q) and A1_priors_summary.txt (per arm: alpha', beta, w, kappa_w,
log C and trend-path recovery against the baseline fits of the same replications).
usage: python followup/a1_priors.py OUT_DIR BASELINE_JSONL [arms]
"""
import json
import math
import os
import sys
import time

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

CELL = [c for c in study_a.cells() if c["reference"]][0]
REPS = (0, 1, 2, 3, 4)
ARMS = {
    "m_alpha_log1": dict(prior=dict(m_alpha_mean=math.log(1.0))),
    "m_alpha_log5": dict(prior=dict(m_alpha_mean=math.log(5.0))),
    "innov_x2": dict(prior=dict(sigma_eta_sd=0.2, tau_zeta_sd=0.2)),
    "innov_x5": dict(prior=dict(sigma_eta_sd=0.5, tau_zeta_sd=0.5)),
    "ahp_reversed": dict(wahp=(0.375, 0.625)),
}
BASE_PRIOR = dict(C.PRIOR)
BASE_WAHP = tuple(C.W_AHP)
BASE_INIT = dict(C.INIT_UNCONSTRAINED)


def apply_arm(arm):
    C.PRIOR.clear(); C.PRIOR.update(BASE_PRIOR); C.PRIOR.update(ARMS[arm].get("prior", {}))
    C.W_AHP = ARMS[arm].get("wahp", BASE_WAHP)
    C.INIT_UNCONSTRAINED.update(BASE_INIT)
    C.INIT_UNCONSTRAINED["m_alpha"] = C.PRIOR["m_alpha_mean"]          # chains start at the prior median (v1.4)
    C.INIT_UNCONSTRAINED["alpha1"] = C.PRIOR["m_alpha_mean"]
    C.INIT_UNCONSTRAINED["alpha_i"] = C.PRIOR["m_alpha_mean"]


def main():
    out, base_path = sys.argv[1], sys.argv[2]
    arms = sys.argv[3:] or list(ARMS)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "A1_priors.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            rec = json.loads(line)
            done.add((rec["arm"], rec["rep"]))
    log = NG.LOG[0]
    I, T = CELL["I"], CELL["T"]
    cap = Mo.mcap(I, T, C.K, C.MISS_DEFAULT)
    for arm in arms:
        todo = [r for r in REPS if (arm, r) not in done]
        if not todo:
            continue
        apply_arm(arm)
        sims = [study_a.simulate(CELL, r) for r in todo]                 # data unchanged (W_AHP enters the model only)
        ds = [Mo.prep(s, T, "latent", cap) for s in sims]
        fitter = NG.BatchNUTS(Mo.ews_model, study_a.VARIANT_SPEC["hier"], ds[0], post_fn=study_a.post_fn,
                              name=f"A1-{arm}-I{I}-T{T}")
        seeds = [dgp.seed_of("A-fit", CELL["id"], "hier", r) % 2 ** 31 for r in todo]
        t0 = time.time()
        res, info, _ = fitter.fit(ds, seeds, batch=len(ds))
        host = {k: (np.asarray(v) if not isinstance(v, dict) else {kk: np.asarray(vv) for kk, vv in v.items()})
                for k, v in res.items()}
        with open(path, "a") as f:
            for j, r in enumerate(todo):
                q = {k: {kk: vv[j] for kk, vv in v.items()} for k, v in host.items() if k != "cp"}
                tr = sims[j]["truth"]
                rec = dict(rep=r, cell=CELL["id"], variant="hier", arm=arm, settings=ARMS[arm], design=C.DESIGN, **info[j])
                rec["truth"] = dict(alpha=tr["alpha"], kappa_w=tr["kappa_w"])
                rec["recovery"] = MT.recovery_q(q, tr, "hier")
                f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)) + "\n")
        log(f"A1 {arm}: {len(todo)} fits in {time.time() - t0:.0f}s, failed {sum(i['failed'] for i in info)}, "
            f"refit {sum(i['n_refit'] for i in info)}")
        NG.clear_done()
    summarise(path, base_path, out)


def _arm_stats(rr):
    a50 = np.array([x["recovery"]["alpha"]["q50"] for x in rr])          # (n, G)
    a05 = np.array([x["recovery"]["alpha"]["q05"] for x in rr]); a95 = np.array([x["recovery"]["alpha"]["q95"] for x in rr])
    b50 = np.array([x["recovery"]["beta"]["q50"] for x in rr]); b05 = np.array([x["recovery"]["beta"]["q05"] for x in rr]); b95 = np.array([x["recovery"]["beta"]["q95"] for x in rr])
    w50 = np.array([x["recovery"]["w"]["q50"] for x in rr]); w05 = np.array([x["recovery"]["w"]["q05"] for x in rr]); w95 = np.array([x["recovery"]["w"]["q95"] for x in rr])
    k50 = np.array([x["recovery"]["kappa_w"]["q50"] for x in rr]); k05 = np.array([x["recovery"]["kappa_w"]["q05"] for x in rr]); k95 = np.array([x["recovery"]["kappa_w"]["q95"] for x in rr])
    bt, wt, kt = np.array(C.BETA), np.array(C.W_TRUE), C.KAPPA_TRUE
    return dict(n=len(rr), failed=sum(x["failed"] for x in rr), refit=sum(x["n_refit"] for x in rr),
                alpha_geo=float(np.exp(np.log(a50).mean())), alpha_cover=float(((a05 <= 2) & (2 <= a95)).mean()),
                alpha_logwidth=float(np.mean(np.log(a95) - np.log(a05))),
                beta_bias=float((b50 - bt).mean()), beta_cover=float(((b05 <= bt) & (bt <= b95)).mean()),
                w_bias=float((w50 - wt).mean()), w_cover=float(((w05 <= wt) & (wt <= w95)).mean()),
                kappa_med=float(k50.mean()), kappa_cover=float(((k05 <= kt) & (kt <= k95)).mean()),
                logC_rmse=float(np.mean([x["recovery"]["logC_summary"]["rmse"] for x in rr])),
                logC_cover=float(np.mean([x["recovery"]["logC_summary"]["cover"] for x in rr])),
                m_cover=float(np.mean([x["recovery"]["m_t"]["cover"] for x in rr])),
                m_rmse=float(np.mean([x["recovery"]["m_t"]["rmse"] for x in rr])),
                d_cover=float(np.mean([x["recovery"]["delta_t"]["cover"] for x in rr])),
                d_rmse=float(np.mean([x["recovery"]["delta_t"]["rmse"] for x in rr])),
                sig_eta=float(np.mean([x["recovery"]["sigma_eta"]["q50"] for x in rr])),
                tau_zeta=float(np.mean([x["recovery"]["tau_zeta"]["q50"] for x in rr])))


def summarise(path, base_path, out):
    recs = [json.loads(l) for l in open(path)]
    base = [json.loads(l) for l in open(base_path)]
    base = [x for x in base if x["rep"] in REPS]
    rows = {"baseline (manuscript priors)": _arm_stats(base)}
    for arm in ARMS:
        rr = [x for x in recs if x["arm"] == arm]
        if rr:
            rows[arm] = _arm_stats(rr)
    cols = ["n", "failed", "alpha_geo", "alpha_cover", "alpha_logwidth", "beta_bias", "beta_cover", "w_bias", "w_cover",
            "kappa_med", "kappa_cover", "logC_rmse", "logC_cover", "m_cover", "m_rmse", "d_cover", "d_rmse", "sig_eta", "tau_zeta"]
    lines = ["A1 prior sensitivity, reference cell, replications 0-4 (truth: alpha' 2, beta (0.4, 0.3, 0.2), w (0.5, 0.3), kappa_w 0.8)",
             "arm | " + " | ".join(cols)]
    for k, v in rows.items():
        lines.append(f"{k} | " + " | ".join(f"{v[c]:.3f}" if isinstance(v[c], float) else str(v[c]) for c in cols))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(out, "A1_priors_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
