"""B5 — severity submodel (8)-(9) evaluated (manuscript Sections 4.3, 5.4 and 8, Table S15).

Re-runs the robustness protocol of Section 5.4 (recovery fit + M2 alarm pipeline + EWMA / CUSUM) on the reference
design with log-normal severity (cell 'R-sevref', the Table 4 reference) and with Pareto severity (cell 'R-pareto',
the arm of Section 5.4), five replications each, with the v1.24 pipeline that also records, at every origin, the
posterior of (gamma, phi) of the severity submodel and the posterior predictive 5-95% interval of the loss fraction,
and evaluates that interval on the losses realised in the decision months, raw and after conformalised quantile
regression (evaluate.severity_metrics). In the data-generating process the loss fraction is drawn independently of
the intensity and the indicators, so the truth is gamma_1 = gamma_2 = 0 and logit^{-1}(gamma_0) = E[S].
Results <out>/R/<cell>__latent.jsonl (study_robust records with an 'M2' -> 'severity' block).
usage: python followup/b5_severity.py OUT_DIR [--tiny] [cells...]
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", False)
import config as C
C.apply_profile("t4_min")
C.XGB_WORKERS = 1

def main():
    out = sys.argv[1]
    tiny = "--tiny" in sys.argv
    which = [a for a in sys.argv[2:] if not a.startswith("--")] or ["R-pareto", "R-sevref"]
    reps = 1 if tiny else 5
    if tiny:
        C.CHAINS, C.WARMUP, C.SAMPLES, C.WARM_START_WARMUP, C.REFIT = 2, 30, 30, 10, (30, 0.99, 10, 60)
        C.ORIGINS, C.DECISION_END, C.BOOT_PATHS, C.BOOT_CHUNK = (36, 42, 48, 54), 59, 8, 2
        C.DIAG_CHUNK = 2
        C.RHAT_MAX, C.ESS_MIN = 1.5, 5
        C.BOOT_MONTHS = C.DECISION_END - C.ORIGINS[0]
        C.CONF_MIN_CALIB = 5
    import nuts_gpu as NG, store, study_robust
    os.makedirs(out, exist_ok=True)
    def log(msg, main=False):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] [B5] {msg}"
        print(line, flush=True)
        with open(os.path.join(out, "run.log"), "a") as f: f.write(line + "\n")
    NG.LOG[0] = log
    base = dict(alpha=C.REF["alpha"], mu=C.REF["mu"], regime=C.REF["regime"], I=C.REF["I"], T=C.REF["T"], reps=reps)
    cells = {"R-pareto": dict(base, id="R-pareto", kw=dict(severity="pareto"), arms=["latent"]),
             "R-sevref": dict(base, id="R-sevref", kw={}, arms=["latent"])}
    if tiny:
        for c in cells.values(): c["I"], c["T"] = 10, 60
    for cid in which:
        c = cells[cid]
        done = store.load(out, "R", c["id"], "latent")
        for r in range(reps):
            if r in done:
                continue
            t0 = time.time()
            study_robust.run_items([(c, r)], "latent", out, 1, "R", log)
            log(f"{cid} rep {r} done in {(time.time() - t0) / 60:.1f} min", main=True)

if __name__ == "__main__":
    main()
