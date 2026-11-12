"""run_rare_full.py — the rare arm of the simulation study (Sections 5.1, 5.3 and 5.6; Supplementary Note 17).

Three cells at a base rate of 0.02 shocks per item-month (I = 100, T = 120, alpha' = 2; stable, gradual and abrupt
regimes), run through the Study B pipeline of code/study_b.py (M2, Poisson M0, XGBoost, sigma-rule, EWMA, CUSUM and the
bootstrap-calibrated q channel) with the t4_min design constants. Place this file in the repository root, next to code/.

  python run_rare_full.py --out results_rare --reps 5 --no-nets            # the configuration reported in the paper
  python run_rare_full.py --out results_rare --reps 5 --regimes gradual    # one regime; LSTM and TFT included
  python run_rare_full.py --out OUT --reps 1 --smoke                       # two-origin smoke test

Resumable: rerun the same command; finished origins and replications are cached under <out>/_tmp, and one line per
finished replication is appended to <out>/B/<cell>__all.jsonl. --hours stops a process cleanly after that many hours;
--rep-start lets two processes share the replications of one cell. Seeds are dgp.seed_of("B", cell, rep), so the
records do not depend on the machine. (Log lines carry the working name "pilot" of this driver.)
"""
import argparse, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--hours", type=float, default=1.9)
ap.add_argument("--smoke", action="store_true"); ap.add_argument("--models", default="M2,M0")
ap.add_argument("--no-nets", action="store_true", help="skip LSTM/TFT (pilot behaviour)")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--regimes", default="stable,gradual,abrupt"); ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--rep-start", type=int, default=0, help="first replication index to run (lets two processes split one cell's reps)")
a = ap.parse_args()
import jax
jax.config.update("jax_enable_x64", False)
cache = os.path.join(a.out, "_tmp", "jax_cache"); os.makedirs(cache, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", cache)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)
import numpy as np
import config as C
C.apply_profile("t4_min")
C.XGB_WORKERS = 1
C.NN_CHUNK = 1
C.DIAG_CHUNK = 1
I, T = 100, 120
if a.smoke:
    C.CHAINS, C.WARMUP, C.SAMPLES, C.WARM_START_WARMUP, C.REFIT = 2, 30, 30, 10, (30, 0.99, 10, 60)
    C.ORIGINS, C.DECISION_END, C.BOOT_PATHS, C.BOOT_CHUNK = (36, 42), 47, 8, 2
    C.RHAT_MAX, C.ESS_MIN = 1.5, 5
    C.BOOT_MONTHS = C.DECISION_END - C.ORIGINS[0]
    I, T = 20, 60
import study_b, alarm, run, nuts_gpu as NG, store
os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
logf = open(os.path.join(a.out, "logs", "pilot.log"), "a")
def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
NG.LOG[0] = log
def _no_nets(name, Ns, Xs, Zs, ys, t, pm, seed, log_=None):
    return [np.zeros((len(pm), Ns[r].shape[0]), np.float32) for r in range(len(Ns))], [{} for _ in Ns]
if a.no_nets:
    study_b._nn_chunks = _no_nets
regs = a.regimes.split(",")
cells = [dict(id=f"B-a2-mu0.02-{reg}-I{I}", alpha="2", mu="0.02", regime=reg, I=I, T=T, reps=a.reps, reference=False) for reg in regs]
items = [(c, r) for c in cells for r in range(a.rep_start, a.reps) if r not in store.load(a.out, "B", c["id"], "all")]
log(f"pilot start: {len(items)} items to do, models {a.models}, deadline {a.hours} h, design {C.DESIGN}, I={I} T={T}")
alarm.DEADLINE[0] = run.DEADLINE[0] = time.time() + a.hours * 3600
C.DEADLINE_EPOCH = None
t0 = time.time()
try:
    n = study_b.run_items(items, a.out, 1, "B", models=tuple(a.models.split(",")), log=log)
    log(f"pilot finished {n} items in {(time.time() - t0) / 60:.1f} min")
except (alarm.Deadline, NG.TimeUp) as e:
    log(f"[deadline] {e} — rerun to resume")
