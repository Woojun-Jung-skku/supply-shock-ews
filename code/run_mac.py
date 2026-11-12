"""run_mac.py — run the simulation on a multi-core CPU (MacBook) with several worker processes.

  python run_mac.py --out ~/ews_v1_results --workers 8 --profile mac_min          # run (or resume)
  python run_mac.py --out ~/ews_v1_results --progress                             # progress only
  python run_mac.py --out ~/ews_v1_results --workers 8 --profile mac_min --hours 8  # stop after 8 h

Each worker process owns a disjoint set of cells (round-robin over the cell lists), so no two
processes ever write the same result file; the priority inside a worker is Study B -> Study A
(sizes (30,120), (10,60), (30,240)) -> robustness, as in run.one_day. Re-running the same command
resumes: finished replications are skipped, finished pipeline stages are cached per replication, and
Study B batches continue at origin granularity. Logs: <out>/logs/worker_<k>.log and <out>/run.log
(one line per unit). JAX runs single-threaded in every worker (one core each).
"""
import argparse
import os
import subprocess
import sys
import time


def _worker_env():
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


def worker(args):
    _worker_env()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import jax
    jax.config.update("jax_enable_x64", False)
    cache = args.xla_cache or os.path.join(args.out, "_tmp", "jax_cache")
    os.makedirs(cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)
    import config as C
    C.apply_profile(args.profile)
    C.XGB_WORKERS = 1                                         # v1.4: one core per worker process
    if args.tiny:                                             # mechanics smoke test only (not the design)
        C.CHAINS, C.WARMUP, C.SAMPLES, C.WARM_START_WARMUP, C.REFIT = 2, 30, 30, 10, (30, 0.99, 10, 60)
        C.ORIGINS, C.DECISION_END, C.BOOT_PATHS, C.BOOT_CHUNK = (36, 42), 47, 8, 2
        C.NN_STEPS, C.NN_EVAL_EVERY, C.NN_CHUNK, C.DIAG_CHUNK = 40, 20, 2, 2
        C.RHAT_MAX, C.ESS_MIN = 1.5, 5
        C.SIZES, C.REF = ((10, 60),), dict(alpha="2", mu="0.2", regime="gradual", I=10, T=60)
        C.BOOT_MONTHS = C.DECISION_END - C.ORIGINS[0]
        C.PASS1_REPS = dict(A=1, A_ref=1, B=1, B_ref=1, R=1)
    if args.reps:
        C.PASS1_REPS = dict(A=args.reps, A_ref=2 * args.reps, B=args.reps, B_ref=2 * args.reps, R=args.reps)
    import run, study_a, study_b, study_robust, alarm, nuts_gpu as NG
    if args.tiny:
        study_b.I_, study_b.T_ = 10, 60
        _cells_b, _cells_r = study_b.cells, study_robust.cells
        def _tiny_b():
            cs = _cells_b()[:3]
            for c in cs: c["I"], c["T"] = 10, 60
            return cs
        def _tiny_r():
            cs = [c for c in _cells_r() if c["id"] in ("R-hawkes-n0.3-k1.0", "R-cold20", "R-miss15")]
            for c in cs: c["I"], c["T"] = 10, 60
            return cs
        study_b.cells, study_robust.cells = _tiny_b, _tiny_r
        _cells_a = study_a.cells
        study_a.cells = lambda size=None: _cells_a(size)[:3]
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    wlog_path = os.path.join(args.out, "logs", f"worker_{args.worker}.log")
    main_log = run.make_logger(args.out)

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] [w{args.worker}] {msg}"
        print(line, flush=True)
        with open(wlog_path, "a") as f:
            f.write(line + "\n")
    NG.LOG[0] = log
    k, N = args.worker, (args.nslots or args.workers)
    if args.hours:
        run.DEADLINE[0] = alarm.DEADLINE[0] = time.time() + args.hours * 3600
    batch = run._batch_cap(args.batch)
    studies = args.studies.split(",")
    t_start = time.time()

    def rss_mb():
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except OSError:
            pass
        import resource                                           # macOS: peak RSS in bytes
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2

    def recycle(reason):
        """v1.4: leave with exit code 75 so that the wrapper loop (tools/worker_loop.sh) starts a fresh
        process — compiled programs accumulate (a worker reached 3.3 GB and the JIT then failed to link
        every new program: 'Failed to materialize symbols'). Finished work is on disk; nothing is lost."""
        log(f"[recycle] {reason}; RSS {rss_mb():.0f} MB — exiting for a fresh process")
        sys.stdout.flush()
        os._exit(75)

    if args.max_rss_mb:                                       # hard limit inside a replication, after checkpointed origins
        hard = int(args.max_rss_mb * 1.45)

        def _after_origin():
            if rss_mb() > hard:
                recycle(f"RSS above {hard} MB inside a replication (after a checkpointed origin)")
        NG.AFTER_ORIGIN[0] = _after_origin

    def unit(name, fn, n_items=None):
        if args.dry:
            log(f"[dry] {name}: {n_items} replication-fits to do")
            return
        if run.past_deadline():
            log(f"[deadline] skipping {name}")
            return
        t0 = time.time()
        try:
            n = fn()
        except NG.TimeUp as e:                                    # segmented run: checkpointed, stop this worker
            log(f"[timeup] {e}")
            raise
        except alarm.Deadline as e:
            log(f"[deadline] {e}")
            return
        except Exception as e:
            import traceback
            log(f"{name} FAILED: {e!r}\n{traceback.format_exc()[-1500:]}")
            run.warn(args.out, f"worker {k}: {name} failed: {e!r}")
            if args.max_rss_mb and ("materialize symbols" in repr(e) or isinstance(e, MemoryError)):
                recycle(f"{name}: broken JIT / out of memory")
            return False
        if n:
            main_log(f"[w{k}] {name}: {n} replication-fits in {(time.time() - t0) / 60:.1f} min")
        if args.max_rss_mb:
            r = rss_mb()
            log(f"[rss] {r:.0f} MB after {name}")
            if r > args.max_rss_mb:
                recycle(f"RSS above {args.max_rss_mb} MB")
        return True

    if args.segment_seconds:                                  # v1.4: Mac VM (3-minute commands) — checkpointed fits
        C.CHUNK_DIR = os.path.join(args.out, "_tmp", f"ckpt_w{k}")
        C.DEADLINE_EPOCH = t_start + args.segment_seconds
        if args.boot_seconds:                                 # time one bootstrap block / one ML origin needs
            C.BOOT_SECONDS_MAX = args.boot_seconds
        if args.ml_seconds:
            C.ML_SECONDS_MAX = args.ml_seconds
    reps = C.PASS1_REPS
    if args.plan:                                             # v1.4: explicit unit list per worker (plan file)
        import json as _json
        cb = {c["id"]: c for c in study_b.cells()}
        ca = {c["id"]: c for c in study_a.cells()}
        cr = {c["id"]: c for c in study_robust.cells()}

        def todo(u):
            # v1.4: optional 4th element of a unit = first replication index to run (replications below it
            # are done elsewhere, e.g. on the other machine); default --rep-min
            lo = int(u[3]) if len(u) > 3 else args.rep_min
            hi = int(u[4]) if len(u) > 4 else 10 ** 9            # optional 5th element: first index not to run
            if u[0] == "B":
                c = cb[u[1]]
                return [(c, r) for r in range(lo, min(hi, reps["B_ref" if c["reference"] else "B"], c["reps"]))
                        if r not in study_b.store.load(args.out, "B", c["id"], "all")]
            if u[0] == "A":
                c, variant = ca[u[1]], u[2]
                done = study_a.store.load(args.out, "A", c["id"], variant)
                return [(c, r) for r in range(lo, min(hi, study_a.pass_reps(c, reps))) if r not in done]
            c, arm = cr[u[1]], u[2]
            done = study_robust.store.load(args.out, "R", c["id"], arm)
            return [(c, r) for r in range(lo, min(hi, reps["R"], c["reps"])) if r not in done]
        skip, tried = set(), set()
        while not run.past_deadline():
            # the plan file is re-read before every unit: units can be moved between workers while they run
            # (only units a worker has not started yet should be moved)
            plan = _json.load(open(args.plan)).get(str(k), [])
            nxt = next(((u, it) for u in plan if tuple(u) not in skip for it in [todo(u)] if it), None)
            if nxt is None:
                break
            u, items = nxt
            if args.max_rss_mb:
                # recycling mode: one replication per call, so the memory check (and a restart) falls between
                # replications; a unit is skipped in this process only after a failure or a repeated replication
                items = items[:1]
                rk = (tuple(u), items[0][1])
                if rk in tried:
                    skip.add(tuple(u))
                    continue
                tried.add(rk)
            else:
                skip.add(tuple(u))                            # one attempt per unit and process (a failure is logged)
            if u[0] == "B":
                ok = unit(f"Study B {u[1]}", lambda items=items: study_b.run_items(items, args.out, batch, log=log), len(items))
            elif u[0] == "A":
                ok = unit(f"Study A {u[1]} {u[2]}", lambda items=items, v=u[2]: study_a.run_items(items, v, args.out, batch, log=log), len(items))
            else:
                ok = unit(f"robustness {u[1]} {u[2]}", lambda items=items, a=u[2]: study_robust.run_items(items, a, args.out, batch, log=log), len(items))
            if ok is False:
                skip.add(tuple(u))
        log(f"worker plan done in {(time.time() - t_start) / 3600:.2f} h")
        return
    # every unit below is one cell (or one (size, variant) of one cell): batches never span cells, so
    # their progress files stay valid when the number of workers changes
    if "B" in studies:
        for c in study_b.cells()[k::N]:
            items = [(c, r) for r in range(min(reps["B_ref" if c["reference"] else "B"], c["reps"]))
                     if r not in study_b.store.load(args.out, "B", c["id"], "all")]
            if items:
                unit(f"Study B {c['id']}", lambda items=items: study_b.run_items(items, args.out, batch, log=log), len(items))
    if "A" in studies:
        for size, variant in run.A_ORDER:
            for c in study_a.cells(size)[k::N]:
                if variant not in c["variants"]:
                    continue
                n = study_a.pass_reps(c, reps)
                done = study_a.store.load(args.out, "A", c["id"], variant)
                items = [(c, r) for r in range(n) if r not in done]
                if items:
                    unit(f"Study A {c['id']} {variant}", lambda items=items, variant=variant: study_a.run_items(items, variant, args.out, batch, log=log), len(items))
    if "R" in studies:
        cs = study_robust.cells()
        mine = cs[k::N]
        for arm in ("latent", "locf"):
            for c in mine:
                if arm not in c["arms"]:
                    continue
                done = study_robust.store.load(args.out, "R", c["id"], arm)
                items = [(c, r) for r in range(min(reps["R"], c["reps"])) if r not in done]
                if items:
                    unit(f"robustness {c['id']} {arm}", lambda items=items, arm=arm: study_robust.run_items(items, arm, args.out, batch, log=log), len(items))
    log(f"worker done in {(time.time() - t_start) / 3600:.2f} h")


def worker_main(args):
    import nuts_gpu as NG
    try:
        worker(args)
    except NG.TimeUp:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--profile", default="t4_min", choices=("mac_min", "t4_min", "colab_min", "oneday"))
    ap.add_argument("--studies", default="B,A,R", help="comma-separated subset of B,A,R in priority order")
    ap.add_argument("--hours", type=float, default=0.0, help="stop starting new units after this many hours")
    ap.add_argument("--batch", type=int, default=1, help="replications per fit call inside a worker (v1.4: 1 — chains and replications run one after another on a CPU)")
    ap.add_argument("--slots", default="", help="v1.4: comma-separated worker slots to run on this machine (default 0..workers-1)")
    ap.add_argument("--nslots", type=int, default=0, help="v1.4: total worker slots over all machines (default = workers); slot k owns cells k, k+nslots, ...")
    ap.add_argument("--reps", type=int, default=0, help="override replications per cell (reference cell: 2x)")
    ap.add_argument("--quiet", action="store_true", help="nice 10 (slower, quieter fans)")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--dry", action="store_true", help="list the units each worker would run, without fitting")
    ap.add_argument("--plan", default="", help="v1.4: JSON {slot: [[study, cell, variant/arm], ...]}; a worker runs exactly its list")
    ap.add_argument("--boot-seconds", type=float, default=0.0, help="v1.4 segmented runs: seconds one bootstrap block needs (default config)")
    ap.add_argument("--ml-seconds", type=float, default=0.0, help="v1.4 segmented runs: seconds one ML origin needs (default config)")
    ap.add_argument("--rep-min", type=int, default=0, help="v1.4: plan mode, run only replications >= this index (default 0)")
    ap.add_argument("--max-rss-mb", type=int, default=0, help="v1.4: after a unit, exit with code 75 when the process RSS exceeds this (use with tools/worker_loop.sh)")
    ap.add_argument("--segment-seconds", type=float, default=0.0, help="v1.4: stop (checkpointed) after this many seconds; re-run to continue")
    ap.add_argument("--xla-cache", default="", help="v1.4: JAX compilation cache directory (default <out>/_tmp/jax_cache)")
    args = ap.parse_args()
    args.out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(args.out, exist_ok=True)
    if args.progress:
        _worker_env()
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import config as C
        C.apply_profile(args.profile)
        if args.reps:
            C.PASS1_REPS = dict(A=args.reps, A_ref=2 * args.reps, B=args.reps, B_ref=2 * args.reps, R=args.reps)
        import run
        print(run.progress(args.out))
        lp = os.path.join(args.out, "logs")
        if os.path.isdir(lp):
            for f in sorted(os.listdir(lp)):
                lines = open(os.path.join(lp, f)).read().strip().splitlines()
                if lines:
                    print(f"{f}: {lines[-1][:160]}")
        return
    if args.worker is not None:
        worker_main(args)
        return
    slots = [int(x) for x in args.slots.split(",")] if args.slots else list(range(args.workers))
    nslots = args.nslots or args.workers
    cmd = [sys.executable, os.path.abspath(__file__), "--out", args.out, "--workers", str(args.workers),
           "--profile", args.profile, "--studies", args.studies, "--hours", str(args.hours), "--batch", str(args.batch),
           "--reps", str(args.reps), "--nslots", str(nslots), "--segment-seconds", str(args.segment_seconds)]
    if args.plan:
        cmd += ["--plan", os.path.abspath(args.plan)]
    if args.max_rss_mb:
        cmd += ["--max-rss-mb", str(args.max_rss_mb)]
    if args.rep_min:
        cmd += ["--rep-min", str(args.rep_min)]
    if args.xla_cache:
        cmd += ["--xla-cache", args.xla_cache]
    if args.tiny:
        cmd.append("--tiny")
    if args.dry:
        cmd.append("--dry")
    if args.quiet:
        cmd = ["nice", "-n", "10"] + cmd
    procs = [subprocess.Popen(cmd + ["--worker", str(k)]) for k in slots]
    print(f"started {len(procs)} workers (profile {args.profile}); logs in {args.out}/logs", flush=True)
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        print("stopped; re-run the same command to resume")
        return
    sys.argv = [sys.argv[0], "--out", args.out, "--profile", args.profile, "--progress"]
    main()


if __name__ == "__main__":
    main()
