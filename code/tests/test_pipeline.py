"""End-to-end smoke test on CPU with tiny MCMC settings (not the design). v1.3: cross-cell batches.
usage: python tests/test_pipeline.py [A|B|R|all] [out_dir]"""
import sys, os, time, shutil, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
C.CHAINS, C.WARMUP, C.SAMPLES = 2, 50, 50
C.WARM_START_WARMUP = 20
C.REFIT = (50, 0.99, 10, 100)
C.ORIGINS = (36, 42, 48)
C.DECISION_END = 53
C.BOOT_PATHS = 8
C.BOOT_CHUNK = 3
C.NN_STEPS, C.NN_EVAL_EVERY = 40, 20
C.NN_HIDDEN = (8,)
C.NN_CHUNK = 3
C.DIAG_CHUNK = 2
C.XGB_GRID = C.XGB_GRID[:2]
C.RHAT_MAX = 1.5                       # tiny chains: do not trigger refits everywhere
C.ESS_MIN = 5
C.SIZES = ((10, 60),)
C.REF = dict(alpha="2", mu="0.2", regime="gradual", I=10, T=60)
C.BOOT_MONTHS = C.DECISION_END - C.ORIGINS[0]
which = sys.argv[1] if len(sys.argv) > 1 else "all"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.environ.get("SP", "/tmp"), "test_out")
import study_a, study_b, study_robust, store, run
log = run.make_logger(OUT) if os.path.isdir(OUT) or not os.makedirs(OUT, exist_ok=True) else run.make_logger(OUT)
if which in ("A", "all"):
    shutil.rmtree(os.path.join(OUT, "A"), ignore_errors=True)
    # two cells of the (10, 60) size, 2 reps each, hier + none, then resume (nothing to do)
    cells = [c for c in study_a.cells((10, 60)) if c["id"] in ("A-a2-mu0.5-abrupt-I10-T60", "A-a0.5-mu0.05-stable-I10-T60")]
    items = [(c, r) for c in cells for r in range(2)]
    for v in ("hier", "none"):
        t = time.time(); n = study_a.run_items(items, v, OUT, batch=4); print(v, n, round(time.time() - t))
    for c in cells:
        r = store.load(OUT, "A", c["id"], "hier"); print(c["id"], len(r), json.dumps(r[0])[:700])
    assert study_a.run_items([(c, r) for c in cells for r in range(2) if r not in store.load(OUT, "A", c["id"], "hier")], "hier", OUT, 3) == 0
    print("A todo after run:", len(study_a.todo_items((10, 60), "hier", OUT, 2)), "(should be 34*2 - 4 = 68)")
if which in ("B", "all"):
    shutil.rmtree(os.path.join(OUT, "B"), ignore_errors=True); shutil.rmtree(os.path.join(OUT, "_tmp", "B"), ignore_errors=True)
    study_b.I_, study_b.T_ = 10, 60
    cells = [c for c in study_b.cells() if c["id"] in ("B-a2-mu0.2-gradual", "B-a0.5-mu0.05-stable")]
    for c in cells: c["I"], c["T"] = 10, 60
    items = [(c, r) for c in cells for r in range(2)]
    t = time.time(); study_b.run_items(items, OUT, batch=4, models=("M2", "M0")); print("B", round(time.time() - t))
    r = store.load(OUT, "B", cells[0]["id"], "all")
    print(json.dumps(r[0], indent=0)[:3000])
    print("origins M2", r[0]["M2"]["origins"], "M0", r[0]["M0"]["origins"])
if which in ("R", "all"):
    shutil.rmtree(os.path.join(OUT, "R"), ignore_errors=True); shutil.rmtree(os.path.join(OUT, "_tmp", "R"), ignore_errors=True)
    cells = {c["id"]: c for c in study_robust.cells()}
    for c in cells.values(): c["I"], c["T"] = 10, 60
    items = [(cells["R-cold20"], 0), (cells["R-miss15"], 0), (cells["R-hawkes-n0.3-k1.0"], 0)]
    t = time.time(); study_robust.run_items(items, "latent", OUT, batch=3); print("R latent", round(time.time() - t))
    t = time.time(); study_robust.run_items([(cells["R-miss15"], 0)], "locf", OUT, batch=1); print("R locf", round(time.time() - t))
    r = store.load(OUT, "R", "R-cold20", "latent"); print(json.dumps(r[0])[:1500])
    r = store.load(OUT, "R", "R-miss15", "locf"); print(json.dumps(r[0])[:600])
print(run.progress(OUT))
