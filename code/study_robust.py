"""study_robust.py — robustness (§5.4): one factor at a time around the reference cell
(alpha' = 2, mu = 0.2, gradual, I = 30, T = 120); 100 replications per cell as the final target,
filled in passes.

  Hawkes truth, n {0, 0.3, 0.6} x kappa*Delta {0.3, 1, 4}      9 cells
  measurement error in z, sd 0.5                               1
  15% missing indicators: latent AR(1) (model) and LOCF arms   1 (two arms on the same data)
  Pareto severity                                              1
  20% cold-start items                                         1
  indicator lag misspecified by one period                     1
Each replication: a recovery fit on the full series (M2, as in Study A) and the alarm pipeline
(M2, rolling origins as in Study B) with the EWMA / CUSUM charts alongside.
v1.3: all latent-AR(1) arms are run together (same shapes), the LOCF arm separately.
Results: <out>/R/<cell>__<arm>.jsonl.
"""
import hashlib
import time
import numpy as np
import config as C
import dgp
import models as Mo
import nuts_gpu as NG
import metrics as MT
import alarm
import evaluate as EV
import store
import study_a
import study_b


def cells():
    base = dict(alpha=C.REF["alpha"], mu=C.REF["mu"], regime=C.REF["regime"], I=C.REF["I"], T=C.REF["T"],
                reps=C.R_REPS)
    out = []
    for n in C.HAWKES_N:
        for kd in C.HAWKES_KD:
            out.append(dict(base, id=f"R-hawkes-n{n}-k{kd}", kw=dict(hawkes=(n, kd)), arms=["latent"]))
    out.append(dict(base, id=f"R-measerr{C.MEAS_ERR}", kw=dict(meas_err=C.MEAS_ERR), arms=["latent"]))
    out.append(dict(base, id=f"R-miss{int(C.MISS_HIGH * 100)}", kw=dict(miss=C.MISS_HIGH), arms=["latent", "locf"]))
    out.append(dict(base, id="R-pareto", kw=dict(severity="pareto"), arms=["latent"]))
    out.append(dict(base, id=f"R-cold{int(C.COLD_FRAC * 100)}", kw=dict(cold=C.COLD_FRAC), arms=["latent"]))
    out.append(dict(base, id=f"R-lag{C.LAG_TRUE_MISSPEC}", kw=dict(x_lag=C.LAG_TRUE_MISSPEC), arms=["latent"]))
    return out


def robust_cap(I, T):
    """Latent-slot cap shared by all robustness cells (largest missingness, no cold-start removal),
    so that every chunk of every arm has the same array shapes."""
    return max(Mo.mcap(I, T, C.K, c["kw"].get("miss", C.MISS_DEFAULT), 0.0) for c in cells())


def _tag(items, arm):
    h = hashlib.sha1("|".join(f"{c['id']}:{r}" for c, r in items).encode()).hexdigest()[:10]
    return f"R-{arm}-{len(items)}reps-{h}"


_FITTERS = {}


def run_items(items, arm, out, batch, study="R", log=None):
    """Recovery fit + M2 alarm pipeline for the (cell, replication) pairs `items` with indicator
    mode `arm`, in GPU batches of `batch`."""
    log = log or NG.LOG[0]
    tmp = store.tmp_dir(out, study, "batches")
    spec = ("M2", "hier", "gauss", arm, C.G)
    for b0 in range(0, len(items), batch):
        chunk = items[b0:b0 + batch]
        tag = _tag(chunk, arm)
        sims = [study_b.simulate(c, r, study, **c["kw"]) for c, r in chunk]
        I, T = sims[0]["N"].shape
        t0 = time.time()
        cap = robust_cap(I, T)                              # one cap for every robustness cell: one compiled fitter
        ds = [Mo.prep(s, T, arm, cap) for s in sims]
        key = (arm, I, T, cap)
        if key not in _FITTERS:
            _FITTERS[key] = NG.BatchNUTS(Mo.ews_model, spec, ds[0], post_fn=study_a.post_fn, name=f"R-recovery-{arm}")
        fitter = _FITTERS[key]
        res, info, _ = fitter.fit(ds, [dgp.seed_of("R-fit", c["id"], arm, r) % 2 ** 31 for c, r in chunk], batch=len(chunk))
        host = {k: (np.asarray(v) if not isinstance(v, dict) else {kk: np.asarray(vv) for kk, vv in v.items()})
                for k, v in res.items()}
        log(f"{tag} recovery fits: {time.time() - t0:.0f}s, failed {sum(i['failed'] for i in info)}/{len(chunk)}")
        with_alarm = [j for j, (c, _) in enumerate(chunk) if C.R_ALARM_HAWKES or "hawkes" not in c["kw"]]
        pipe = [None] * len(chunk)
        if with_alarm:
            sub, _ = alarm.run([sims[j] for j in with_alarm], "M2", tag, tmp, (study,), len(with_alarm), xmode=arm, log=log, cap=cap)
            for j, r in zip(with_alarm, sub):
                pipe[j] = r
            log(f"{tag} alarm pipeline ({len(with_alarm)} replications): {time.time() - t0:.0f}s")
        months = pipe[with_alarm[0]]["months"] if with_alarm else None
        for j, (c, r) in enumerate(chunk):
            s = sims[j]
            tr = s["truth"]
            q = {k: {kk: vv[j] for kk, vv in v.items()} for k, v in host.items() if k != "cp"}
            rec = dict(rep=r, cell=c["id"], arm=arm, design=C.DESIGN, recovery_fit=info[j],
                       truth=dict(alpha=tr["alpha"], kappa_w=tr["kappa_w"]),
                       recovery=MT.recovery_q(q, tr, "hier"))
            if pipe[j] is None:                                          # recovery-only arm (reduced profile)
                rec["M2"] = None
                store.append(out, study, c["id"], arm, rec)
                continue
            rec["M2"] = EV.model_metrics(pipe[j], s, c["regime"], with_extras=True)
            rec["charts"] = EV.chart_metrics(s, months, c["regime"], dgp.seed_of("chart", c["id"], r))
            if c["kw"].get("cold"):                                      # alarms for cold-start items only
                cold = tr["cold_items"]
                sub = dict(s, N=s["N"][cold], Nmask=s["Nmask"][cold], group=s["group"][cold])
                from ml import labels
                y = labels(sub["N"])[months]
                val = EV._valid(sub, months)
                rec["M2_cold_items"] = MT.prob_metrics(pipe[j]["p"][:, cold][val], y[val])
            rec["seconds_batch"] = round(time.time() - t0, 1)
            store.append(out, study, c["id"], arm, rec)
        NG.clear_done()
        log(f"{tag}: saved {len(chunk)} replications, {time.time() - t0:.0f}s")
    return len(items)


def todo_items(arm, out, reps, study="R"):
    items = []
    for c in cells():
        if arm not in c["arms"]:
            continue
        n = min(reps if isinstance(reps, int) else reps["R"], c["reps"])
        done = store.load(out, study, c["id"], arm)
        items += [(c, r) for r in range(n) if r not in done]
    return items


def run_pass(out, batch, reps, study="R", log=None):
    n = 0
    for arm in ("latent", "locf"):
        n += run_items(todo_items(arm, out, reps, study), arm, out, batch, study, log)
    return n
