"""run.py — entry points used by the Colab notebook (v1.3).

  pilot(out, batch)        timing probe: Study A reference cell (hierarchical) at two batch sizes ->
                           seconds per NUTS iteration-step as a function of the number of chains,
                           pass rate, R-hat exceedances; written to pilot_report.json
  estimate(out)            projected GPU hours of pass 1 and of the full design from the pilot
  one_day(out, batch, hours)   the one-day schedule: pass 1 of every study in priority order
                           (B -> A -> robustness), then top-ups towards the manuscript's replication
                           counts while time remains; every unit is resumable
  study_a_pass / study_b_pass / robust_pass(out, batch, reps)   one pass of one study
  progress(out)            replications done / target per study, in-progress batches, warnings
"""
import collections
import glob
import json
import os
import time
import numpy as np
import config as C
import alarm
import nuts_gpu as NG
import store
import study_a
import study_b
import study_robust


# ---------------------------------------------------------------- logging
def make_logger(out):
    path = os.path.join(out, "run.log")

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(path, "a") as f:
            f.write(line + "\n")
    NG.LOG[0] = log
    return log


def warn(out, msg):
    with open(os.path.join(out, "warnings.log"), "a") as f:
        f.write(f"[{time.strftime('%F %T')}] {msg}\n")
    NG.LOG[0]("WARNING " + msg)


def _fail_report(recs):
    sites = collections.Counter(r["attempts"][-1]["rhat_site"] for r in recs if r["failed"]
                                and r["attempts"][-1]["rhat_max"] >= C.RHAT_MAX)
    other = collections.Counter("ESS" if r["ess_bulk_min"] <= C.ESS_MIN or r["ess_tail_min"] <= C.ESS_MIN
                                else "divergence" for r in recs if r["failed"] and r["rhat_max"] < C.RHAT_MAX)
    return dict(rhat_sites=dict(sites), other=dict(other))


def check_failures(out, study, cell_id, variant):
    recs = list(store.load(out, study, cell_id, variant).values())
    key = "recovery_fit" if study == "R" else None
    fits = [r[key] if key else r for r in recs]
    if len(fits) >= 20:
        rate = sum(f["failed"] for f in fits) / len(fits)
        if rate > C.FAIL_WARN:
            warn(out, f"{study} {cell_id} {variant}: failure rate {rate:.0%} > {C.FAIL_WARN:.0%} — {_fail_report(fits)}")


# ---------------------------------------------------------------- pilot and estimate
def pilot(out, batch, sizes=(8, 64), log=None):
    """Fit the reference cell (hierarchical M2, full settings) with `sizes` replications in one call
    each, on the pilot study 'pilot_A' (kept apart from the results). Reports seconds per
    iteration-step per call, from which time = a + b * chains is fitted."""
    log = log or make_logger(out)
    ref = next(c for c in study_a.cells() if c["reference"])
    rows = []
    r0 = 0
    for n in sizes:
        items = [(ref, r) for r in range(r0, r0 + n)]
        r0 += n
        t0 = time.time()
        study_a.run_items(items, "hier", out, n, study="pilot_A", log=log)
        wall = time.time() - t0
        recs = [store.load(out, "pilot_A", ref["id"], "hier")[r] for _, r in items]
        first = [r["attempts"][0] for r in recs]
        steps = np.mean([a["mean_steps"] for a in first])
        iters = C.CHAINS * (C.WARMUP + C.SAMPLES)
        fit_s = np.mean([a["seconds"] for a in first]) * n           # seconds of the design fit call
        rows.append(dict(reps=n, chains=n * C.CHAINS, wall_s=round(wall), fit_s=round(fit_s, 1),
                         compile_s=round(NG.COMPILE_SECONDS[0]), mean_steps=round(float(steps), 1),
                         us_per_chain_step=round(1e6 * fit_s / (iters * steps * n), 3),
                         passed_first=int(sum(a["ok"] for a in first)), passed_after_refit=int(sum(1 - r["failed"] for r in recs)),
                         rhat_over_mean=round(float(np.mean([a["n_rhat_over"] for a in first])), 2),
                         n_rhat=first[0]["n_rhat"], ess_bulk_min=round(float(min(a["ess_bulk_min"] for a in first))),
                         failures=_fail_report(recs)))
        log(f"pilot {n} reps: {rows[-1]}")
    # time per fit call = a + b * chains (seconds), from the two sizes; per-step version for scaling
    if len(rows) >= 2:
        (x1, y1), (x2, y2) = [(r["chains"], r["fit_s"]) for r in rows[:2]]
        b = max((y2 - y1) / (x2 - x1), 0.0)
        a = max(y1 - b * x1, 0.0)
    else:
        a, b = 0.0, rows[0]["fit_s"] / rows[0]["chains"]
    n_all = sum(r["reps"] for r in rows)
    pass_first = sum(r["passed_first"] for r in rows) / max(n_all, 1)
    rep = dict(rows=rows, model=dict(a=round(a, 1), b=round(b, 4), iters=C.CHAINS * (C.WARMUP + C.SAMPLES),
                                     steps=rows[-1]["mean_steps"], pass_first=round(pass_first, 3)),
               time=time.strftime("%F %T"), design=C.DESIGN)
    json.dump(rep, open(os.path.join(out, "pilot_report.json"), "w"), indent=1)
    log(f"pilot: fit seconds ~ {a:.0f} + {b:.3f} x chains (design fit, {rep['model']['iters']} iterations)")
    return rep


# leapfrog steps per iteration at the design settings (0.99 / depth 10), reference-like cells, CPU
# measurements of 2026-09-30: the trajectory length grows with the series length (local linear trend)
STEPS_BY_T = {60: 127, 120: 511, 240: 1023}
STEPS_ORIGIN = (36, 127)                      # Study B origin 36 (data up to month 36): ~127 steps (Mac run)


def _steps_at(T):
    if T in STEPS_BY_T:
        return STEPS_BY_T[T]
    t0, s0 = STEPS_ORIGIN
    return float(np.clip(s0 + (STEPS_BY_T[120] - s0) * (T - t0) / (120 - t0), s0, 1023))


def _fit_seconds(model, chains, iters=None, obs_scale=1.0, steps_scale=1.0):
    """Seconds of one fit call (design fit) plus the refit call that almost every batch needs: with
    first-pass rate p per replication, a batch of R replications triggers the refit (config.REFIT,
    warm-up + samples iterations, on the failed lanes only) with probability 1 - p^R. obs_scale: I*T relative to the pilot cell; steps_scale: leapfrog steps per
    iteration relative to the pilot's."""
    it = model["iters"] if iters is None else iters
    base = (model["a"] + model["b"] * chains) * (it / model["iters"]) * obs_scale * steps_scale
    p = model.get("pass_first", 0.7)
    R = max(1, chains // C.CHAINS)
    refit_iters = C.CHAINS * (C.REFIT[0] + C.REFIT[3])
    lanes = chains * max(1.0 - p, 0.05)                              # only the failed replications are refitted
    refit = (model["a"] + model["b"] * lanes) * (refit_iters / model["iters"]) * obs_scale * steps_scale
    return base + (1.0 - p ** R) * refit


def estimate(out, reps=None):
    """Projected GPU hours from pilot_report.json: pass 1 (config.PASS1_REPS) and the full design.
    Fits scale with the number of chains (pilot model) and with I*T relative to the reference cell;
    the particle filter, bootstrap, ML baselines and charts are added as fractions measured in the
    CPU validation (rough; the run log gives the real numbers)."""
    rep = json.load(open(os.path.join(out, "pilot_report.json")))
    m = rep["model"]
    reps = reps or C.PASS1_REPS
    full = dict(A=C.A_REPS, A_ref=C.A_REPS_REF, B=C.B_REPS, B_ref=C.B_REPS_REF, R=C.R_REPS)

    def hours(rp):
        # Study A: one call per (size, variant) with all its cells
        ha = 0.0
        s_ref = m.get("steps", STEPS_BY_T[120])
        for I, T in C.SIZES:
            cs = study_a.cells((I, T))
            scale = I * T / (C.REF["I"] * C.REF["T"])
            ss = _steps_at(T) / s_ref
            for v in ("hier", "none", "complete", "horseshoe"):
                n = sum(study_a.pass_reps(c, rp) for c in cs if v in c["variants"])
                if n:
                    ha += _fit_seconds(m, n * C.CHAINS, obs_scale=scale, steps_scale=ss) * (1.15 if v == "horseshoe" else 1.0)
        # Study B: M2 at its origins (first full, then warm-started), baselines at theirs; the steps
        # per iteration grow with the origin (data up to month t)
        nb = sum(rp["B_ref" if c["reference"] else "B"] for c in study_b.cells())
        ch = nb * C.CHAINS
        warm_iters = C.CHAINS * (C.WARM_START_WARMUP + C.SAMPLES)

        def pipeline(chains, name="M2"):
            org = alarm.origins_of(name)
            return sum(_fit_seconds(m, chains, None if i == 0 else warm_iters, steps_scale=_steps_at(t) / s_ref)
                       for i, t in enumerate(org))
        hb = sum(pipeline(ch, name) for name in C.B_MODELS)
        hb *= 1.35                                                     # filter, bootstrap, severity, charts, ML
        # robustness: recovery fit for every arm, M2 pipeline for the non-Hawkes arms (config.R_ALARM_HAWKES)
        nr = sum(rp["R"] * len(c["arms"]) for c in study_robust.cells())
        na = sum(rp["R"] * len(c["arms"]) for c in study_robust.cells() if C.R_ALARM_HAWKES or "hawkes" not in c["kw"])
        hr = _fit_seconds(m, nr * C.CHAINS) + pipeline(na * C.CHAINS)
        hr *= 1.3
        return dict(study_A_h=round(ha / 3600, 1), study_B_h=round(hb / 3600, 1), robust_h=round(hr / 3600, 1),
                    total_h=round((ha + hb + hr) / 3600, 1))
    return dict(pass1=hours(reps), full_design=hours(full), pilot_model=m)


# ---------------------------------------------------------------- passes
def is_oom(e):
    m = str(e)
    return any(k in m for k in ("RESOURCE_EXHAUSTED", "out of memory", "OUT_OF_MEMORY", "MemoryAllocation", "OOM"))


def with_oom_retry(fn, batch, log=None, n_items=None, **kw):
    """Run fn(batch=...) and halve the batch on a GPU out-of-memory error (results are kept:
    finished replications and finished pipeline stages are cached, so the retry resumes)."""
    log = log or NG.LOG[0]
    if n_items:
        batch = min(batch, n_items)
    while True:
        try:
            return fn(batch=batch, **kw)
        except Exception as e:                                       # XlaRuntimeError: RESOURCE_EXHAUSTED
            if not is_oom(e) or batch <= 1:
                raise
            NG._CACHE.clear()
            batch = max(1, batch // 2)
            log(f"GPU out of memory — retrying with batch {batch}")


def _batch_cap(batch):
    return min(batch, C.MAX_CHAINS // C.CHAINS)


DEADLINE = [None]


def past_deadline():
    return DEADLINE[0] is not None and time.time() > DEADLINE[0]


def study_b_pass(out, batch, reps=None, log=None):
    log = log or NG.LOG[0]
    reps = reps or C.PASS1_REPS
    items = study_b.todo_items(out, reps)
    if not items or past_deadline():
        return 0
    n = with_oom_retry(lambda batch: study_b.run_items(items, out, batch, log=log), _batch_cap(batch), log, len(items))
    for c in study_b.cells():
        recs = list(store.load(out, "B", c["id"], "all").values())
        if len(recs) >= 20:
            rate = np.mean([r["any_failed_fit"] for r in recs])
            if rate > C.FAIL_WARN:
                warn(out, f"B {c['id']}: {rate:.0%} of replications have a failed origin fit")
    return n


A_ORDER = (((30, 120), "hier"), ((10, 60), "hier"), ((30, 120), "none"), ((30, 120), "complete"),
           ((10, 60), "none"), ((10, 60), "complete"), ((30, 240), "hier"),
           ((30, 120), "horseshoe"), ((10, 60), "horseshoe"), ((30, 240), "horseshoe"),
           ((30, 240), "none"), ((30, 240), "complete"))


def study_a_pass(out, batch, reps=None, log=None, order=A_ORDER):
    log = log or NG.LOG[0]
    reps = reps or C.PASS1_REPS
    n = 0
    for size, variant in order:
        if past_deadline():
            log(f"[deadline] stopping Study A pass before {size} {variant}")
            break
        items = study_a.todo_items(size, variant, out, reps)
        if not items:
            continue
        n += with_oom_retry(lambda batch: study_a.run_items(items, variant, out, batch, log=log), _batch_cap(batch), log, len(items))
        for c in study_a.cells(size):
            if variant in c["variants"]:
                check_failures(out, "A", c["id"], variant)
    return n


def robust_pass(out, batch, reps=None, log=None):
    log = log or NG.LOG[0]
    reps = reps or C.PASS1_REPS
    n = 0
    for arm in ("latent", "locf"):
        if past_deadline():
            log(f"[deadline] stopping robustness pass before arm {arm}")
            break
        items = study_robust.todo_items(arm, out, reps)
        if not items:
            continue
        n += with_oom_retry(lambda batch: study_robust.run_items(items, arm, out, batch, log=log), _batch_cap(batch), log, len(items))
    for c in study_robust.cells():
        for arm in c["arms"]:
            check_failures(out, "R", c["id"], arm)
    return n


def one_day(out, batch, hours=11.0, log=None, topup_step=30):
    """Pass 1 of every study (config.PASS1_REPS), then top-ups of `topup_step` replications per cell
    per round towards the manuscript's counts, until `hours` have elapsed. Units already complete are
    skipped, so re-running after a disconnect continues where it stopped."""
    import alarm
    log = log or make_logger(out)
    t_start = time.time()
    deadline = t_start + hours * 3600
    DEADLINE[0] = deadline
    alarm.DEADLINE[0] = deadline

    def left():
        return (deadline - time.time()) / 3600

    def stage(name, fn):
        if time.time() > deadline:
            log(f"[one_day] {left():.1f} h left — skipping {name}")
            return False
        log(f"[one_day] === {name} ({left():.1f} h left) ===")
        t0 = time.time()
        try:
            n = fn()
        except alarm.Deadline as e:
            log(f"[one_day] {e}")
            return False
        except Exception as e:
            import traceback
            log(f"[one_day] {name} FAILED: {e!r}\n{traceback.format_exc()[-1500:]}")
            warn(out, f"{name} failed: {e!r}")
            return True                                              # go on with the next unit
        log(f"[one_day] {name}: {n} replication-fits in {(time.time() - t0) / 3600:.2f} h")
        return time.time() < deadline

    stage("Study B pass 1", lambda: study_b_pass(out, batch, C.PASS1_REPS, log))
    stage("Study A pass 1", lambda: study_a_pass(out, batch, C.PASS1_REPS, log))
    stage("robustness pass 1", lambda: robust_pass(out, batch, C.PASS1_REPS, log))
    full = dict(A=C.A_REPS, A_ref=C.A_REPS_REF, B=C.B_REPS, B_ref=C.B_REPS_REF, R=C.R_REPS)
    prev, k = dict(C.PASS1_REPS), 1
    while time.time() < deadline:
        rp = {key: min(C.PASS1_REPS[key] + k * topup_step, full[key]) for key in full}
        if rp == prev:
            log("[one_day] full design reached")
            break
        done = stage(f"top-up round {k} (to {rp})", lambda: (study_b_pass(out, batch, rp, log)
                                                            + study_a_pass(out, batch, rp, log)
                                                            + robust_pass(out, batch, rp, log)))
        if not done:
            break
        prev, k = rp, k + 1
    DEADLINE[0] = alarm.DEADLINE[0] = None
    log(f"[one_day] finished after {(time.time() - t_start) / 3600:.2f} h\n" + progress(out))


# ---------------------------------------------------------------- progress
def progress(out, reps=None):
    reps = reps or C.PASS1_REPS
    rows = []
    tot = collections.defaultdict(lambda: [0, 0, 0, 0])            # done, pass-1 target, full target, failed
    for c in study_a.cells():
        for v in c["variants"]:
            recs = store.load(out, "A", c["id"], v)
            t = tot["A"]; t[0] += len(recs); t[1] += study_a.pass_reps(c, reps); t[2] += c["reps"]
            t[3] += sum(r["failed"] for r in recs.values())
    for c in study_b.cells():
        recs = store.load(out, "B", c["id"], "all")
        t = tot["B"]; t[0] += len(recs); t[1] += min(reps["B_ref" if c["reference"] else "B"], c["reps"]); t[2] += c["reps"]
        t[3] += sum(r["any_failed_fit"] for r in recs.values())
    for c in study_robust.cells():
        for arm in c["arms"]:
            recs = store.load(out, "R", c["id"], arm)
            t = tot["R"]; t[0] += len(recs); t[1] += min(reps["R"], c["reps"]); t[2] += c["reps"]
            t[3] += sum(r["recovery_fit"]["failed"] for r in recs.values())
    names = dict(A="Study A", B="Study B", R="Robustness")
    for k in ("B", "A", "R"):
        d, n1, n2, f = tot[k]
        rows.append(f"{names[k]}: {d} replication-fits done — pass 1 target {n1} ({100 * d / max(n1, 1):.0f}%), "
                    f"full design {n2} ({100 * d / max(n2, 1):.0f}%), failed {f}")
    for p in sorted(glob.glob(os.path.join(out, "_tmp", "*", "batches", "*.pkl.json"))):
        try:
            j = json.load(open(p))
            if j["done"] < j["n_origins"] and time.time() - os.path.getmtime(p) < 2 * 86400:
                rows.append(f"in progress: {os.path.basename(p)[:-9]} origins {j['done']}/{j['n_origins']} ({j['seconds'] / 60:.0f} min so far)")
        except Exception:
            pass
    wp = os.path.join(out, "warnings.log")
    if os.path.exists(wp):
        lines = open(wp).read().strip().splitlines()
        rows.append(f"warnings.log: {len(lines)} entries" + (f" — last: {lines[-1]}" if lines else ""))
    return "\n".join(rows)
