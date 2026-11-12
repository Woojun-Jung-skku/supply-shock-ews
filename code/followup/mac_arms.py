"""followup/mac_arms.py — follow-up arms A2, A5 and A6 on a multi-core Mac (v1.4 code, Terminal).

  python followup/mac_arms.py a2 --out ~/ews_followup --workers 5
  python followup/mac_arms.py a5 --out ~/ews_followup --workers 6
  python followup/mac_arms.py a6 --out ~/ews_followup --workers 4
  python followup/mac_arms.py summary --out ~/ews_followup            # tables + followup_summary.txt
  python followup/mac_arms.py smoke --out ~/ews_smoke                 # mechanics only (tiny settings), ~5-10 min

A2  beta = 0 (the indicators carry no information). The robustness protocol of §5.4 — recovery fit (M2,
    full series) + the M2 rolling-origin alarm pipeline + the EWMA / CUSUM charts — on the reference design
    (alpha' = 2, mu-bar = 0.2, gradual, I = 30, T = 120) with C.BETA = (0, 0, 0) in the truth; 5 replications,
    seeds of study 'R', cell id 'R-beta0'. Results <out>/R/R-beta0__latent.jsonl (study_robust record).
A5  M2 refitted every 24 months (origins 36, 60, 84, 108; C.B_M2_STRIDE = 4) on the 65 Study B replications
    (the same simulated data and the same origin-36 seeds as Table 6; followup/b_items.json); M2 only, no
    ML baselines. Results <out>/B24/<cell>__M2.jsonl (the Study B 'M2' block).
A6  'horseshoe2' — the regularised horseshoe of §4.1 with its half-Cauchy local scale written as z sqrt(nu),
    z ~ half-N(0, 1), nu ~ InverseGamma(1/2, 1/2) (same prior, friendlier geometry; models.py) — sampled with
    warm-up 1,000 / 1,000 draws per chain, target acceptance 0.99, tree depth 10, the refit at the same
    settings, in the two strong-data abrupt cells A-a2-mu0.2-abrupt-I30-T120 and A-a2-mu0.5-abrupt-I30-T120,
    replications 0-9 (the data of the Study A horseshoe fits). Results <out>/A6/<cell>__horseshoe2.jsonl
    (Study A record incl. the changepoint statistics of §5.2 (iii)).
Every arm is resumable (finished replications are skipped), runs one replication per unit in worker processes
with single-thread JAX (one core each), and logs to <out>/logs/<arm>_w<k>.log and <out>/run.log. Nothing in a
results folder is ever overwritten; the arms write to their own folders (R, B24, A6) under --out.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

A6_CELLS = ("A-a2-mu0.2-abrupt-I30-T120", "A-a2-mu0.5-abrupt-I30-T120")
A6_REPS = 10
A6_SAMPLER = dict(warmup=1000, samples=1000, target_accept=0.99, depth=10, refit=(1000, 0.99, 10, 1000))
A2_REPS = 5


def _worker_env():
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("PYTHONHASHSEED", "0")           # identical traced programs -> compilation cache hits


# ---------------------------------------------------------------------------------------------- work lists
def items_a2(C, reps):
    cell = dict(alpha=C.REF["alpha"], mu=C.REF["mu"], regime=C.REF["regime"], I=C.REF["I"], T=C.REF["T"],
                reps=reps, id="R-beta0", kw={}, arms=["latent"])
    return [(cell, r) for r in range(reps)]


def items_a5(study_b):
    cells = {c["id"]: c for c in study_b.cells()}
    pairs = json.load(open(os.path.join(HERE, "b_items.json")))
    return [(cells[cid], int(r)) for cid, r in pairs]


def items_a6(study_a, reps):
    cells = {c["id"]: c for c in study_a.cells((30, 120))}
    return [(cells[cid], r) for cid in A6_CELLS for r in range(reps)]


# ---------------------------------------------------------------------------------------------- one worker
def worker(args):
    _worker_env()
    sys.path.insert(0, ROOT)
    import jax
    jax.config.update("jax_enable_x64", False)
    cache = args.xla_cache or os.path.join(args.out, "_tmp", "jax_cache")
    os.makedirs(cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)
    import numpy as np
    import config as C
    C.apply_profile("t4_min")
    C.XGB_WORKERS = 1
    if args.tiny:                                             # mechanics smoke test only (not the design)
        C.CHAINS, C.WARMUP, C.SAMPLES, C.WARM_START_WARMUP, C.REFIT = 2, 30, 30, 10, (30, 0.99, 10, 60)
        C.ORIGINS, C.DECISION_END, C.BOOT_PATHS, C.BOOT_CHUNK = (36, 42, 48, 54), 59, 8, 2
        C.DIAG_CHUNK = 2
        C.RHAT_MAX, C.ESS_MIN = 1.5, 5
        C.BOOT_MONTHS = C.DECISION_END - C.ORIGINS[0]
    import dgp, models as Mo, nuts_gpu as NG, metrics as MT, store, study_a, study_b, alarm, evaluate as EV
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    wlog = os.path.join(args.out, "logs", f"{args.arm}_w{args.worker}.log")
    rlog = os.path.join(args.out, "run.log")

    def log(msg, main=False):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] [{args.arm} w{args.worker}] {msg}"
        print(line, flush=True)
        with open(wlog, "a") as f:
            f.write(line + "\n")
        if main:
            with open(rlog, "a") as f:
                f.write(line + "\n")
    NG.LOG[0] = log
    k, N = args.worker, args.nslots
    t_start = time.time()
    n_done = 0
    if args.segment_seconds:                                  # v1.4 segmented fits (Mac VM: commands end after ~3 min):
        C.CHUNK_DIR = os.path.join(args.out, "_tmp", f"ckpt_{args.arm}_w{k}")   # checkpoint, TimeUp, resume on the next call
        C.DEADLINE_EPOCH = t_start + args.segment_seconds
        if args.boot_seconds:
            C.BOOT_SECONDS_MAX = args.boot_seconds

    def loop(items, done, run_one):
        """One replication after another; a TimeUp (segmented run) ends this call with the work checkpointed."""
        nonlocal n_done
        for c, r in items[k::N]:
            if done(c, r):
                continue
            try:
                run_one(c, r)
            except NG.TimeUp as e:
                log(f"[timeup] {e}")
                return
            n_done += 1

    if args.arm == "a2":
        C.BETA = (0.0, 0.0, 0.0)
        C.DESIGN = C.DESIGN + "+A2-beta0"
        import study_robust
        if args.tiny:
            study_robust.cells = lambda _c=study_robust.cells: [dict(c, I=10, T=60) for c in _c()]
        items = items_a2(C, args.reps or A2_REPS)
        if args.tiny:
            items = items[:1]
            for c, _ in items:
                c["I"], c["T"] = 10, 60
        def run_a2(c, r):
            t0 = time.time()
            study_robust.run_items([(c, r)], "latent", args.out, 1, "R", log)
            log(f"A2 {c['id']} rep {r} done in {(time.time() - t0) / 60:.1f} min", main=True)
        loop(items, lambda c, r: r in store.load(args.out, "R", c["id"], "latent"), run_a2)

    elif args.arm == "a5":
        C.B_M2_STRIDE = 4                                     # M2 origins 36, 60, 84, 108 (24 months)
        C.DESIGN = C.DESIGN + "+A5-M2-24m"
        items = items_a5(study_b)
        if args.tiny:
            items = items[:1]
            study_b.I_, study_b.T_ = 10, 60
            for c, _ in items:
                c["I"], c["T"] = 10, 60
        assert alarm.origins_of("M2") == C.ORIGINS[::4], alarm.origins_of("M2")
        tmp = store.tmp_dir(args.out, "B24", "batches")
        def run_a5(c, r):
            t0 = time.time()
            s = study_b.simulate(c, r, "B")                   # the Table 6 data (seeds of study 'B')
            tag = f"B24-{c['id']}-r{r}"
            res, _ = alarm.run([s], "M2", tag, tmp, ("B",), 1, log=log)    # same origin-36 seed as Study B
            rec = dict(rep=r, cell=c["id"], design=C.DESIGN, t0=int(s["truth"]["t0"]),
                       M2=EV.model_metrics(res[0], s, c["regime"], with_extras=True))
            rec["M2"]["origins"] = res[0]["origins"]
            rec["seconds_batch"] = round(time.time() - t0, 1)
            store.append(args.out, "B24", c["id"], "M2", rec)
            NG.clear_done()
            f = rec["M2"]["fit"]
            log(f"A5 {c['id']} rep {r}: origins {rec['M2']['origins']}, failed origins {f['n_failed']}, refits {f['n_refit']}, "
                f"{(time.time() - t0) / 60:.1f} min", main=True)
        loop(items, lambda c, r: r in store.load(args.out, "B24", c["id"], "M2"), run_a5)

    elif args.arm == "a6":
        C.DESIGN = C.DESIGN + "+A6-horseshoe2"
        spec = ("M2", "hier", "horseshoe2", "latent", C.G)
        items = items_a6(study_a, args.reps or A6_REPS)
        if args.tiny:
            items = items[:1]
            for c, _ in items:
                c["I"], c["T"] = 10, 60
        sampler = dict(A6_SAMPLER)
        if args.tiny:
            sampler = dict(warmup=30, samples=30, target_accept=0.99, depth=10, refit=(30, 0.99, 10, 60))
        fitter = [None]

        def run_a6(c, r):
            t0 = time.time()
            I, T = c["I"], c["T"]
            cap = Mo.mcap(I, T, C.K, C.MISS_DEFAULT)
            s = study_a.simulate(c, r)                        # the Study A data of this (cell, replication)
            d = Mo.prep(s, T, "latent", cap)
            if fitter[0] is None:
                fitter[0] = NG.BatchNUTS(Mo.ews_model, spec, d, post_fn=study_a.post_fn, name=f"A6-horseshoe2-I{I}-T{T}",
                                         warmup=sampler["warmup"], samples=sampler["samples"],
                                         target_accept=sampler["target_accept"], depth=sampler["depth"], refit=sampler["refit"])
            seed = dgp.seed_of("A-fit", c["id"], "horseshoe2", r) % 2 ** 31
            res, info, _ = fitter[0].fit([d], [seed], batch=1)
            host = {kk: (np.asarray(v) if not isinstance(v, dict) else {k2: np.asarray(v2) for k2, v2 in v.items()})
                    for kk, v in res.items()}
            q = {kk: {k2: v2[0] for k2, v2 in v.items()} for kk, v in host.items() if kk != "cp"}
            tr = s["truth"]
            rec = dict(rep=r, cell=c["id"], variant="horseshoe2", design=C.DESIGN,
                       sampler=dict(warmup=sampler["warmup"], samples=sampler["samples"], target_accept=sampler["target_accept"],
                                    depth=sampler["depth"], refit=list(sampler["refit"]), chains=C.CHAINS), **info[0])
            rec["truth"] = dict(alpha=tr["alpha"], kappa_w=tr["kappa_w"])
            rec["recovery"] = MT.recovery_q(q, tr, "hier")
            cp = {kk: v[0] for kk, v in host["cp"].items()}
            rec["changepoint"] = MT.changepoint_q(cp, tr["t0"])
            store.append(args.out, "A6", c["id"], "horseshoe2", rec)
            NG.clear_done()
            a = info[0]["attempts"][-1]
            log(f"A6 {c['id']} rep {r}: failed {info[0]['failed']} (R-hat max {info[0]['rhat_max']:.4f} on {info[0]['rhat_site']}, "
                f"ESS min {min(info[0]['ess_bulk_min'], info[0]['ess_tail_min']):.0f}, div {info[0]['n_div']}), refit {info[0]['n_refit']}, "
                f"steps {a['mean_steps']:.0f}, {(time.time() - t0) / 60:.1f} min", main=True)
        loop(items, lambda c, r: r in store.load(args.out, "A6", c["id"], "horseshoe2"), run_a6)
    log(f"worker done: {n_done} replications in {(time.time() - t_start) / 60:.1f} min", main=True)


# ---------------------------------------------------------------------------------------------- progress
def progress(args):
    sys.path.insert(0, ROOT)
    import config as C
    C.apply_profile("t4_min")
    import store, study_b, study_a
    out = args.out
    lines = []
    for arm in ("a2", "a5", "a6"):
        if arm == "a2":
            total, done = A2_REPS, len(store.load(out, "R", "R-beta0", "latent"))
        elif arm == "a5":
            its = items_a5(study_b)
            done = sum(1 for c, r in its if r in store.load(out, "B24", c["id"], "M2"))
            total = len(its)
        else:
            its = items_a6(study_a, A6_REPS)
            done = sum(1 for c, r in its if r in store.load(out, "A6", c["id"], "horseshoe2"))
            total = len(its)
        lines.append(f"{arm}: {done}/{total} replications done")
    lp = os.path.join(out, "logs")
    if os.path.isdir(lp):
        for f in sorted(os.listdir(lp)):
            ls = open(os.path.join(lp, f)).read().strip().splitlines()
            if ls:
                lines.append(f"{f}: {ls[-1][:150]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- summary
def _ms(v):
    import numpy as np
    v = np.asarray([x for x in v if x is not None], float)
    v = v[np.isfinite(v)]
    if not v.size:
        return (float("nan"), float("nan"), 0)
    return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan"), int(v.size))


def _f(x, d=3):
    try:
        return f"{x:.{d}f}"
    except (TypeError, ValueError):
        return "-"


def _mse(t, d=3):
    return f"{_f(t[0], d)} ({_f(t[1], d)}; n={t[2]})"


def summary(args):
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import numpy as np
    import pandas as pd
    import config as C
    C.apply_profile("t4_min")
    import store, study_a, study_b
    import analyze as AN
    import arl as ARL
    out = args.out
    ref = json.load(open(os.path.join(HERE, "ref", "reference_numbers.json")))
    L = [f"Follow-up arms A2 / A5 / A6 — summary of {out} ({time.strftime('%Y-%m-%d %H:%M')})", ""]

    # ---- A2
    recs = list(store.load(out, "R", "R-beta0", "latent").values())
    L.append(f"A2  beta = 0, reference design, n = {len(recs)} replications (robustness protocol; study_robust record)")
    if recs:
        b50 = np.array([r["recovery"]["beta"]["q50"] for r in recs]); b05 = np.array([r["recovery"]["beta"]["q05"] for r in recs]); b95 = np.array([r["recovery"]["beta"]["q95"] for r in recs])
        L.append(f"  recovery fits: failed {sum(r['recovery_fit']['failed'] for r in recs)}/{len(recs)}, refits {sum(r['recovery_fit']['n_refit'] for r in recs)}")
        L.append(f"  beta posterior median, mean over replications (truth 0): {np.round(b50.mean(0), 3).tolist()}; 90% intervals cover 0: {((b05 <= 0) & (0 <= b95)).mean():.2f}; mean width {np.round((b95 - b05).mean(0), 3).tolist()}")
        w50 = np.array([r["recovery"]["w"]["q50"] for r in recs]); k50 = np.array([r["recovery"]["kappa_w"]["q50"] for r in recs])
        L.append(f"  w posterior median {np.round(w50.mean(0), 3).tolist()} (truth {list(C.W_TRUE)}), kappa_w {k50.mean():.3f} (truth {C.KAPPA_TRUE}); log C RMSE {np.mean([r['recovery']['logC_summary']['rmse'] for r in recs]):.3f}, cover {np.mean([r['recovery']['logC_summary']['cover'] for r in recs]):.2f}")
        rows = []
        for r in recs:
            if r.get("M2") is None:
                continue
            o = AN.model_row("M2", r["M2"])
            o["regime"] = C.REF["regime"]                     # tools/arl.py: stable-window length by regime
            for ch in ("ewma", "cusum"):
                c = r["charts"][ch]
                o.update({f"{ch}_dr1": c["events"]["dr_h1"], f"{ch}_dr3": c["events"]["dr_h3"], f"{ch}_fa_stable": c["events"]["fa_per_itemyear_stable"],
                          f"{ch}_lead": c["events"]["lead_med"], f"{ch}_arl0": c["run_length"]["arl0"], f"{ch}_arl1": c["run_length"].get("arl1"),
                          f"{ch}_stable_alarm_units": c["run_length"].get("stable_alarm_units")})
            sr = r["M2"].get("sigma_rule")
            if sr:
                o.update(sigma_2_dr1=sr["level2"]["dr_h1"], sigma_2_fa_stable=sr["level2"]["fa_per_itemyear_stable"], sigma_2_lead=sr["level2"]["lead_med"])
            rows.append(o)
        d = pd.DataFrame(rows)
        rc = ref["B_reference_cell"]
        if len(d):
            pr = ARL.pooled(d, "M2_q_arl0", "M2_q_stable_alarm_units", ARL.UNITS["q"])
            L.append("  M2 alarm pipeline (triage level 2) — beta = 0 arm | Study B reference cell (beta = (0.4, 0.3, 0.2), n = %d):" % rc["n"])
            for k, lab in (("M2_aucpr", "AUC-PR"), ("M2_brier", "Brier"), ("M2_ece", "ECE"), ("M2_base_rate", "base rate"), ("M2_tri2_dr1", "detected <= 1 mo"),
                           ("M2_tri2_dr3", "detected <= 3 mo"), ("M2_tri2_fa_stable", "false alarms / item-year"), ("M2_tri2_lead", "lead (months)"), ("M2_q_arl1", "ARL1 (q)")):
                L.append(f"    {lab:26s} {_mse(_ms(d[k]))} | {_mse(tuple(rc[k]))}")
            L.append(f"    {'ARL0 (q, pooled)':26s} {pr['est']:.1f} (90% {pr['lo']:.1f}-{pr['hi']:.1f}) | {rc['M2_q_arl0_pooled'][0]:.1f} (90% {rc['M2_q_arl0_pooled'][1]:.1f}-{rc['M2_q_arl0_pooled'][2]:.1f})")
            for ch in ("ewma", "cusum"):
                pr = ARL.pooled(d, f"{ch}_arl0", f"{ch}_stable_alarm_units", ARL.UNITS["item"])
                L.append(f"    {ch.upper():26s} dr1 {_mse(_ms(d[ch + '_dr1']))} | {_mse(tuple(rc[ch + '_dr1']))};  fa {_mse(_ms(d[ch + '_fa_stable']))} | {_mse(tuple(rc[ch + '_fa_stable']))};  ARL0 {pr['est']:.1f} | {rc[ch + '_arl0_pooled'][0]:.1f}")
            if "sigma_2_dr1" in d:
                L.append(f"    {'sigma-rule level 2':26s} dr1 {_mse(_ms(d['sigma_2_dr1']))}, fa {_mse(_ms(d['sigma_2_fa_stable']))}")
            L.append(f"    reference-cell ML AUC-PR for scale: XGBoost {rc['xgb_aucpr'][0]:.3f}, LSTM {rc['lstm_aucpr'][0]:.3f}, TFT {rc['tft_aucpr'][0]:.3f}")
    L.append("")

    # ---- A5
    its = items_a5(study_b)
    recs = []
    for c, r in its:
        rr = store.load(out, "B24", c["id"], "M2")
        if r in rr:
            x = rr[r]; x["_regime"] = c["regime"]; x["_alpha"] = c["alpha"]; x["_mu"] = c["mu"]
            recs.append(x)
    L.append(f"A5  M2 refitted every 24 months (origins 36, 60, 84, 108), n = {len(recs)}/{len(its)} Study B replications")
    if recs:
        rows = []
        for x in recs:
            o = AN.model_row("M2", x["M2"]); o.update(regime=x["_regime"], alpha=x["_alpha"], mu=x["_mu"], cell=x["cell"], rep=x["rep"])
            rows.append(o)
        d = pd.DataFrame(rows)
        rf = ref["B_M2_stride6_pooled"]
        pr = ARL.pooled(d, "M2_q_arl0", "M2_q_stable_alarm_units", ARL.UNITS["q"])
        L.append("  Table 6 M2 row — 24-month refit | 6-month refit (Table 6, n = 65):")
        for k, lab in (("aucpr", "AUC-PR"), ("brier", "Brier"), ("ece", "ECE"), ("tri2_dr1", "detected <= 1 mo"), ("tri2_dr3", "detected <= 3 mo"),
                       ("tri2_fa_stable", "false alarms / item-year"), ("tri2_lead", "lead (months)"), ("q_arl1", "ARL1 (q)"), ("conf_cover", "conformal cover"), ("conf_size", "conformal size")):
            L.append(f"    {lab:26s} {_mse(_ms(d['M2_' + k]))} | {_mse(tuple(rf[k]))}")
        L.append(f"    {'ARL0 (q, pooled)':26s} {pr['est']:.1f} (90% {pr['lo']:.1f}-{pr['hi']:.1f}; no-alarm share {pr['no_alarm_share']:.2f}) | {rf['q_arl0_pooled'][0]:.1f} (90% {rf['q_arl0_pooled'][3]:.1f}-{rf['q_arl0_pooled'][4]:.1f})")
        L.append(f"    {'failed origin fits / rep':26s} {_mse(_ms(d['M2_fit_failed_origins']))} of 4 | {_mse(tuple(rf['fit_failed_origins']))} of 14;  fit seconds / rep {_mse(_ms(d['M2_fit_seconds']), 0)} | {_mse(tuple(rf['fit_seconds']), 0)}")
        L.append("  by regime (24-month | 6-month): AUC-PR, detected <= 1 mo, false alarms, ARL0, ARL1")
        for reg in ("stable", "gradual", "abrupt"):
            dd = d[d.regime == reg]
            if not len(dd):
                continue
            pr = ARL.pooled(dd, "M2_q_arl0", "M2_q_stable_alarm_units", ARL.UNITS["q"])
            rr = ref["B_M2_stride6_by_regime"][reg]
            L.append(f"    {reg:8s} n={len(dd)}/{rr['n']}: AUC-PR {_f(dd.M2_aucpr.mean())} | {_f(rr['aucpr'][0])};  dr1 {_f(dd.M2_tri2_dr1.mean())} | {_f(rr['dr1'][0])};  "
                     f"fa {_f(dd.M2_tri2_fa_stable.mean())} | {_f(rr['fa_stable'][0])};  ARL0 {pr['est']:.1f} | {rr['arl0'][0]:.1f};  ARL1 {_f(dd.M2_q_arl1.mean(), 1)} | {_f(rr['arl1'][0], 1)}")
        d.to_csv(os.path.join(out, "A5_M2_24m_replications.csv"), index=False)
    L.append("")

    # ---- A6
    L.append(f"A6  horseshoe2 (reparameterised regularised horseshoe; warm-up 1,000 / 1,000 draws, target 0.99, depth 10), replications 0-{A6_REPS - 1}")
    L.append("  cell | variant | n | failed | R-hat fail | ESS fail | div fail | R-hat max (median) | ESS min (median) | change-date |error| mean / median | 90% cover | P(true month) mean [range] | m_t RMSE | m_t cover | s / fit (median) | steps")
    for cid in A6_CELLS:
        rr = list(store.load(out, "A6", cid, "horseshoe2").values())
        rows = {}
        if rr:
            cp = [x["changepoint"] for x in rr]
            rows["horseshoe2 (A6)"] = dict(n=len(rr), failed=sum(x["failed"] for x in rr), rhat_fail=sum(x["rhat_max"] >= C.RHAT_MAX for x in rr),
                                           ess_fail=sum(min(x["ess_bulk_min"], x["ess_tail_min"]) <= C.ESS_MIN for x in rr), div_fail=sum(x["n_div"] > 0 for x in rr),
                                           rhat_max_med=float(np.median([x["rhat_max"] for x in rr])), ess_min_med=float(np.median([min(x["ess_bulk_min"], x["ess_tail_min"]) for x in rr])),
                                           cp_abs_err=float(np.mean([c["abs_err"] for c in cp])), cp_abs_err_med=float(np.median([c["abs_err"] for c in cp])),
                                           cp_cover=float(np.mean([c["cover"] for c in cp])), cp_max_prob=float(np.mean([c["max_prob"] for c in cp])),
                                           cp_max_prob_range=[min(c["max_prob"] for c in cp), max(c["max_prob"] for c in cp)],
                                           m_rmse=float(np.mean([x["recovery"]["m_t"]["rmse"] for x in rr])), m_cover=float(np.mean([x["recovery"]["m_t"]["cover"] for x in rr])),
                                           seconds_med=float(np.median([x["seconds"] for x in rr])), steps_med=float(np.median([x["attempts"][-1]["mean_steps"] for x in rr])))
        for v in ("horseshoe", "hier"):
            rows[f"{v} (v1.4)"] = ref["A6_v14_reference"][f"{cid}__{v}"]
        for lab, s in rows.items():
            L.append(f"  {cid} | {lab:16s} | {s['n']} | {s['failed']} | {s['rhat_fail']} | {s['ess_fail']} | {s['div_fail']} | {s['rhat_max_med']:.4f} | {s['ess_min_med']:.0f} | "
                     f"{s['cp_abs_err']:.2f} / {s['cp_abs_err_med']:.1f} | {s['cp_cover']:.2f} | {s['cp_max_prob']:.2f} [{s['cp_max_prob_range'][0]:.2f}-{s['cp_max_prob_range'][1]:.2f}] | "
                     f"{s['m_rmse']:.3f} | {s['m_cover']:.2f} | {s['seconds_med']:.0f} | {s['steps_med']:.0f}")
        if rr:
            sites = {}
            for x in rr:
                if x["failed"]:
                    sites[x["rhat_site"]] = sites.get(x["rhat_site"], 0) + 1
            if sites:
                L.append(f"    failed fits' worst R-hat site: {sites}")
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(out, "followup_summary.txt"), "w").write(txt + "\n")
    return txt


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", choices=("a2", "a5", "a6", "summary", "progress", "smoke"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reps", type=int, default=0, help="override replications (a2: default 5, a6: default 10 per cell)")
    ap.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--nslots", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--quiet", action="store_true", help="nice 10 (slower, quieter fans)")
    ap.add_argument("--segment-seconds", type=float, default=0.0, help="v1.4 segmented fits: checkpoint and stop after this many seconds; re-run to continue (Mac VM)")
    ap.add_argument("--boot-seconds", type=float, default=0.0, help="segmented runs: seconds one bootstrap block needs")
    ap.add_argument("--xla-cache", default="", help="JAX compilation cache directory (default <out>/_tmp/jax_cache)")
    args = ap.parse_args()
    args.out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(args.out, exist_ok=True)
    if args.arm == "summary":
        _worker_env()
        summary(args)
        return
    if args.arm == "progress":
        _worker_env()
        print(progress(args))
        return
    if args.arm == "smoke":
        # tiny settings, one worker per arm, mechanics only: ~5-10 min; results under <out> are NOT design results
        _worker_env()
        for arm in ("a2", "a5", "a6"):
            cmd = [sys.executable, os.path.abspath(__file__), arm, "--out", args.out, "--worker", "0", "--nslots", "1", "--tiny"]
            if args.xla_cache:
                cmd += ["--xla-cache", args.xla_cache]
            t0 = time.time()
            rc = subprocess.call(cmd)
            print(f"smoke {arm}: exit {rc} in {(time.time() - t0) / 60:.1f} min", flush=True)
            if rc:
                sys.exit(rc)
        print("SMOKE OK — mechanics of a2 / a5 / a6 run end to end (tiny settings; delete this folder before the real run)")
        return
    if args.worker is not None:
        worker(args)
        return
    _worker_env()
    N = args.workers
    cmd = [sys.executable, os.path.abspath(__file__), args.arm, "--out", args.out, "--nslots", str(N)]
    if args.reps:
        cmd += ["--reps", str(args.reps)]
    if args.segment_seconds:
        cmd += ["--segment-seconds", str(args.segment_seconds)]
    if args.boot_seconds:
        cmd += ["--boot-seconds", str(args.boot_seconds)]
    if args.xla_cache:
        cmd += ["--xla-cache", args.xla_cache]
    if args.quiet:
        cmd = ["nice", "-n", "10"] + cmd
    procs = [subprocess.Popen(cmd + ["--worker", str(k)]) for k in range(N)]
    print(f"started {N} workers for {args.arm}; logs in {args.out}/logs, one line per replication in {args.out}/run.log", flush=True)
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        print("stopped; re-run the same command to resume (finished replications are kept)")
        return
    print(progress(args))


if __name__ == "__main__":
    main()
