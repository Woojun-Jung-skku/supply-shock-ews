"""study_b.py — Study B, alarm performance (§5.3).

Grid: alpha' {0.5, 2} x mu {0.05, 0.2} x regime {stable, gradual, abrupt} at (I, T) = (30, 120)
= 12 cells; 100 replications (reference cell 200) as the final target, filled in passes.
Methods on every replication: M2 (proposal, refitted at the 14 origins 36, 42, ..., 114), the
model-based baselines Poisson M0, no-pooling NB, complete-pooling NB (refitted every
config.B_BASELINE_STRIDE-th origin and propagated by the same particle filter in between), the
sigma-rule on the plug-in SRI, EWMA and CUSUM on counts, XGBoost, LSTM and TFT at every origin.
v1.3: the replications of all 12 cells are run together (one GPU call per origin and model).
Results: <out>/B/<cell>__all.jsonl, one line per replication with every method.
"""
import hashlib
import os
import pickle
import time
import numpy as np
import config as C
import dgp
import alarm
import ml
import evaluate as EV
import nuts_gpu as NG
import store

I_, T_ = C.REF["I"], C.REF["T"]


def cells():
    out = []
    for a in C.B_ALPHAS:
        for mu in C.B_MUS:
            for reg in C.REGIMES:
                ref = (a, mu, reg) == (C.REF["alpha"], C.REF["mu"], C.REF["regime"])
                out.append(dict(id=f"B-a{a}-mu{mu}-{reg}", alpha=a, mu=mu, regime=reg, I=I_, T=T_,
                                reps=C.B_REPS_REF if ref else C.B_REPS, reference=ref))
    return out


def simulate(cell, r, study="B", **kw):
    s = dgp.simulate(cell["I"], cell["T"], float(cell["mu"]), float(cell["alpha"]), cell["regime"],
                     seed=dgp.seed_of(study, cell["id"], r), **kw)
    s["miss"] = kw.get("miss", C.MISS_DEFAULT)
    s["cold"] = kw.get("cold", 0.0)
    s["uid"] = f"{cell['id']}:{r}"
    return s


def _xgb_one(args):
    N, X, Z, y, t, pm, seed = args
    return ml.xgb_predict(N, X, Z, y, t, pm, seed, n_jobs=1)


def _is_oom(e):
    m = str(e)
    return any(k in m for k in ("RESOURCE_EXHAUSTED", "out of memory", "OUT_OF_MEMORY", "MemoryAllocation", "OOM"))


def _nn_chunks(name, Ns, Xs, Zs, ys, t, pm, seed, log):
    """Train a network for all replications in chunks; halve the chunk on GPU out-of-memory."""
    R = len(Ns)
    out, cfg = [None] * R, [None] * R
    c0 = 0
    while c0 < R:
        c1 = min(R, c0 + C.NN_CHUNK)
        try:
            p, c = ml.nn_predict_batch(name, Ns[c0:c1], Xs[c0:c1], Zs[c0:c1], ys[c0:c1], t, pm, seed)
        except Exception as e:
            if not _is_oom(e) or C.NN_CHUNK <= 1:
                raise
            C.NN_CHUNK = max(1, C.NN_CHUNK // 2)
            log(f"GPU out of memory in {name} training — NN_CHUNK -> {C.NN_CHUNK}")
            continue
        for j, r in enumerate(range(c0, c1)):
            out[r], cfg[r] = p[j], c[j]
        c0 = c1
    return out, cfg


def ml_baselines(sims, tmp, tag, seed, log=None):
    """XGBoost, LSTM, TFT at every origin; probs (R, M, I) for decision months 36..117.
    XGBoost runs on the CPU cores in parallel over replications; the networks are trained on the
    GPU for config.NN_CHUNK replications at a time. Progress is checkpointed after every origin, and
    finished replications are cached per replication under <tmp>/done."""
    log = log or NG.LOG[0]
    from joblib import Parallel, delayed
    os.makedirs(os.path.join(tmp, "done"), exist_ok=True)
    cached = {r: os.path.join(tmp, "done", f"{s['uid'].replace(':', '_')}__ml.pkl") for r, s in enumerate(sims)}
    have = {r: p for r, p in cached.items() if os.path.exists(p)}
    results = {r: pickle.load(open(p, "rb")) for r, p in have.items()}
    todo = [r for r in range(len(sims)) if r not in have]
    if have:
        log(f"{tag} ML: {len(have)}/{len(sims)} replications already done (cache), training {len(todo)}")
    if todo:
        sub = [sims[r] for r in todo]
        R = len(sub)
        Ns = [s["N"] for s in sub]
        Xs = [dgp.locf(s["X"]) for s in sub]
        Zs = [s["Z"] for s in sub]
        ys = [ml.labels(n) for n in Ns]
        ck = os.path.join(tmp, f"{tag}__ml_partial.pkl")
        st = pickle.load(open(ck, "rb")) if os.path.exists(ck) else None
        if st is None or st["R"] != R:
            st = dict(R=R, done=0, out={k: [[] for _ in range(R)] for k in ("xgb", "lstm", "tft")},
                      cfg={k: [[] for _ in range(R)] for k in ("xgb", "lstm", "tft")})
        workers = C.XGB_WORKERS or max(1, os.cpu_count() or 1)
        t_start = time.time()
        for oi, t in enumerate(C.ORIGINS):
            if oi < st["done"]:
                continue
            if NG.time_left() < C.ML_SECONDS_MAX:                     # segmented runs: stop between origins
                raise NG.TimeUp(f"{tag} ML: deadline before origin {t}")
            t0 = time.time()
            t_end = (C.ORIGINS[oi + 1] - 1) if oi + 1 < len(C.ORIGINS) else C.DECISION_END
            pm = np.arange(t, t_end + 1)
            res = Parallel(n_jobs=workers)(delayed(_xgb_one)((Ns[r], Xs[r], Zs[r], ys[r], t, pm, seed + r)) for r in range(R))
            for r, (p, c) in enumerate(res):
                st["out"]["xgb"][r].append(p); st["cfg"]["xgb"][r].append(c)
            t_x = time.time() - t0
            for name in ("lstm", "tft"):
                p, c = _nn_chunks(name, Ns, Xs, Zs, ys, t, pm, seed + oi, log)
                for r in range(R):
                    st["out"][name][r].append(p[r]); st["cfg"][name][r].append(c[r])
            st["done"] = oi + 1
            pickle.dump(st, open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck)
            log(f"{tag} ML origin {t} ({oi + 1}/{len(C.ORIGINS)}): xgb {t_x:.0f}s, nets {time.time() - t0 - t_x:.0f}s")
            NG.AFTER_ORIGIN[0]()
        for j, r in enumerate(todo):
            rec = ({k: np.concatenate(st["out"][k][j], 0) for k in st["out"]}, {k: st["cfg"][k][j] for k in st["cfg"]})
            pickle.dump(rec, open(cached[r] + ".tmp", "wb")); os.replace(cached[r] + ".tmp", cached[r])
            results[r] = rec
        log(f"{tag} ML baselines done in {time.time() - t_start:.0f}s")
    probs = {k: [results[r][0][k] for r in range(len(sims))] for k in ("xgb", "lstm", "tft")}
    cfg = {k: [results[r][1][k] for r in range(len(sims))] for k in ("xgb", "lstm", "tft")}
    return probs, cfg


def _tag(items):
    h = hashlib.sha1("|".join(f"{c['id']}:{r}" for c, r in items).encode()).hexdigest()[:10]
    return f"B-{len(items)}reps-{h}"


def run_items(items, out, batch, study="B", models=C.B_MODELS, log=None):
    """Run every method on the (cell, replication) pairs `items` in GPU batches of `batch`."""
    log = log or NG.LOG[0]
    tmp = store.tmp_dir(out, study, "batches")
    for b0 in range(0, len(items), batch):
        chunk = items[b0:b0 + batch]
        tag = _tag(chunk)
        sims = [simulate(c, r, study) for c, r in chunk]
        t0 = time.time()
        res = {}
        for name in models:
            res[name], _ = alarm.run(sims, name, tag, tmp, (study,), len(chunk), log=log)
            log(f"{tag} {name} pipeline done: {time.time() - t0:.0f}s")
        probs, cfg = ml_baselines(sims, tmp, tag, dgp.seed_of("ml", tag) % 2 ** 31, log=log)
        months = res[models[0]][0]["months"]
        for j, (c, r) in enumerate(chunk):
            s = sims[j]
            rec = dict(rep=r, cell=c["id"], design=C.DESIGN, t0=s["truth"]["t0"])
            for name in models:
                rec[name] = EV.model_metrics(res[name][j], s, c["regime"], with_extras=(name == "M2"))
                rec[name]["origins"] = res[name][j]["origins"]
            for name in ("xgb", "lstm", "tft"):
                rec[name] = EV.prob_method_metrics(probs[name][j], s, months, c["regime"])
                rec[name]["configs"] = cfg[name][j]
            rec["charts"] = EV.chart_metrics(s, months, c["regime"], dgp.seed_of("chart", c["id"], r))
            rec["any_failed_fit"] = int(any(rec[m]["fit"]["n_failed"] for m in models))
            rec["seconds_batch"] = round(time.time() - t0, 1)
            store.append(out, study, c["id"], "all", rec)
        NG.clear_done()
        log(f"{tag}: saved {len(chunk)} replications of {len({c['id'] for c, _ in chunk})} cells, {time.time() - t0:.0f}s")
    return len(items)


def todo_items(out, reps, study="B"):
    items = []
    for c in cells():
        n = reps if isinstance(reps, int) else reps["B_ref" if c["reference"] else "B"]
        n = min(n, c["reps"])
        done = store.load(out, study, c["id"], "all")
        items += [(c, r) for r in range(n) if r not in done]
    return items


def run_pass(out, batch, reps, study="B", log=None):
    return run_items(todo_items(out, reps, study), out, batch, study, log=log)
