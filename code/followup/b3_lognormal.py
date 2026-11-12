"""B3 — heterogeneity that is not gamma (manuscript Sections 3.5, 5.4 and 8, Table S16).

Runs the robustness protocol of Section 5.4 (recovery fit + M2 alarm pipeline + EWMA / CUSUM) on the reference
design with the gamma multiplier of eq. (2) replaced, in the truth only, by a log-normal multiplier with the same
mean (1) and variance (1/alpha' = 0.5): cell 'R-lognormal' (dgp.simulate(..., frailty="lognormal")). The model
fitted is unchanged (gamma-Poisson, hierarchical), so the arm asks what a gamma fit does to log-normal
heterogeneity at the reference base rate. Section 3.5 argues that at these base rates the shape of the
heterogeneity is not separately identified from the counts, so the recovery of alpha' is judged against the
variance-matched value 2 and the alarms against the reference arm 'R-sevref' (results/followup/out/B5/R).
Results <out>/R/R-lognormal__latent.jsonl (study_robust records, five replications).
usage: python followup/b3_lognormal.py OUT_DIR [--tiny]
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
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] [B3] {msg}"
        print(line, flush=True)
        with open(os.path.join(out, "run.log"), "a") as f: f.write(line + "\n")
    NG.LOG[0] = log
    base = dict(alpha=C.REF["alpha"], mu=C.REF["mu"], regime=C.REF["regime"], I=C.REF["I"], T=C.REF["T"], reps=reps)
    c = dict(base, id="R-lognormal", kw=dict(frailty="lognormal"), arms=["latent"])
    if tiny:
        c["I"], c["T"] = 10, 60
    done = store.load(out, "R", c["id"], "latent")
    for r in range(reps):
        if r in done:
            continue
        t0 = time.time()
        study_robust.run_items([(c, r)], "latent", out, 1, "R", log)
        log(f"{c['id']} rep {r} done in {(time.time() - t0) / 60:.1f} min", main=True)

if __name__ == "__main__":
    main()
