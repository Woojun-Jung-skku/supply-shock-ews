"""ml.py — machine-learning baselines of §5.1: gradient-boosted trees (XGBoost), LSTM and a temporal
fusion transformer, "trained on the same x_it with the same rolling-origin splits and
hyper-parameters tuned by inner validation only".

At origin t the labelled decision months are s = 12..t-3 (the 3-month label of month s is observed
by t). The last ML_VAL_MONTHS of them form the inner validation split; hyper-parameters and the
early-stopping point are chosen on it by log loss. The chosen model predicts the decision months
t .. next origin - 1. Missing indicators are carried forward (LOCF), as for all baselines.
LSTM / TFT are written in JAX and trained for a whole batch of replications (and learning rates)
at once with vmap.
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
import config as C

L = C.ML_LOOKBACK


# ------------------------------------------------------------------ features
def labels(N):
    """y[s, i] = 1{N_{i,s+1} + N_{i,s+2} + N_{i,s+3} >= 1} for decision month s (0-based month s)."""
    I, T = N.shape
    y = np.zeros((T + 1, I), np.float32)
    for s in range(T - 2):
        y[s] = N[:, s:s + 3].sum(1) > 0
    return y


def tab_features(N, X, Z, s):
    """GBM features at decision month s: counts of the last 6 months, 12-month total, x_s, z."""
    return np.column_stack([N[:, s - 6:s], N[:, s - 12:s].sum(1), X[:, s], Z]).astype(np.float32)


def seq_features(N, X, s):
    """(I, L, 1 + K): log1p count of month r and x_r for r = s-L+1..s."""
    r = np.arange(s - L + 1, s + 1)
    cnt = np.log1p(N[:, r - 1])[..., None]
    return np.concatenate([cnt, X[:, r]], -1).astype(np.float32)


def split_months(t):
    s_all = np.arange(12, t - 2)
    return s_all[:-C.ML_VAL_MONTHS], s_all[-C.ML_VAL_MONTHS:]


# ------------------------------------------------------------------ gradient boosting
def xgb_predict(N, X, Z, y, t, pred_months, seed, n_jobs=2):
    import xgboost as xgb
    tr, va = split_months(t)
    Xtr = np.vstack([tab_features(N, X, Z, s) for s in tr]); ytr = np.concatenate([y[s] for s in tr])
    Xva = np.vstack([tab_features(N, X, Z, s) for s in va]); yva = np.concatenate([y[s] for s in va])
    Xpr = np.stack([tab_features(N, X, Z, s) for s in pred_months])
    if ytr.min() == ytr.max():
        return np.full((len(pred_months), N.shape[0]), ytr.mean(), np.float32), dict(cfg=None)
    best = None
    for k, g in enumerate(C.XGB_GRID):
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=g["max_depth"], learning_rate=g["learning_rate"],
                                subsample=0.8, tree_method="hist", n_jobs=n_jobs, random_state=seed,
                                eval_metric="logloss", early_stopping_rounds=30)
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        loss = clf.best_score
        if best is None or loss < best[0]:
            best = (loss, k, clf)
    p = best[2].predict_proba(Xpr.reshape(-1, Xpr.shape[-1]))[:, 1].reshape(len(pred_months), -1)
    return p.astype(np.float32), dict(cfg=best[1], val_logloss=round(float(best[0]), 4))


# ------------------------------------------------------------------ small JAX layers
def _dense(key, i, o):
    return dict(W=jax.random.normal(key, (i, o)) * math.sqrt(1.0 / i), b=jnp.zeros(o))


def _ap(p, x):
    return x @ p["W"] + p["b"]


def _ln(x, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps)


def _lstm_init(key, i, h):
    k1, k2 = jax.random.split(key)
    return dict(Wx=jax.random.normal(k1, (i, 4 * h)) * math.sqrt(1.0 / i),
                Wh=jax.random.normal(k2, (h, 4 * h)) * math.sqrt(1.0 / h),
                b=jnp.zeros(4 * h).at[h:2 * h].set(1.0))


def _lstm(p, xs, h0, c0):
    """xs (n, L, i) -> outputs (n, L, h)."""
    def step(carry, x):
        h, c = carry
        z = x @ p["Wx"] + h @ p["Wh"] + p["b"]
        i, f, g, o = jnp.split(z, 4, -1)
        c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return (h, c), h
    _, hs = jax.lax.scan(step, (h0, c0), jnp.swapaxes(xs, 0, 1))
    return jnp.swapaxes(hs, 0, 1)


# ------------------------------------------------------------------ LSTM classifier
def lstm_init(key, nin, nstat, H):
    k = jax.random.split(key, 3)
    return dict(lstm=_lstm_init(k[0], nin, H), stat=_dense(k[1], nstat, H), out=_dense(k[2], 2 * H, 1))


def lstm_apply(p, xs, zs):
    n = xs.shape[0]
    H = p["stat"]["W"].shape[1]
    hs = _lstm(p["lstm"], xs, jnp.zeros((n, H)), jnp.zeros((n, H)))
    return _ap(p["out"], jnp.concatenate([hs[:, -1], jax.nn.elu(_ap(p["stat"], zs))], -1))[:, 0]


# ------------------------------------------------------------------ temporal fusion transformer (Lim et al. 2021)
def _grn_init(key, i, h, o, ctx=None):
    k = jax.random.split(key, 6)
    p = dict(w2=_dense(k[0], i, h), w1=_dense(k[1], h, h), g4=_dense(k[2], h, o), g5=_dense(k[3], h, o))
    if ctx:
        p["w3"] = dict(W=jax.random.normal(k[4], (ctx, h)) * math.sqrt(1.0 / ctx), b=jnp.zeros(h))
    if i != o:
        p["skip"] = _dense(k[5], i, o)
    return p


def _grn(p, a, c=None):
    e2 = _ap(p["w2"], a)
    if c is not None:
        e2 = e2 + c @ p["w3"]["W"]
    e1 = _ap(p["w1"], jax.nn.elu(e2))
    glu = jax.nn.sigmoid(_ap(p["g4"], e1)) * _ap(p["g5"], e1)
    skip = _ap(p["skip"], a) if "skip" in p else a
    return _ln(skip + glu)


def _glu_init(key, h):
    k1, k2 = jax.random.split(key)
    return dict(a=_dense(k1, h, h), b=_dense(k2, h, h))


def _glu(p, x):
    return jax.nn.sigmoid(_ap(p["a"], x)) * _ap(p["b"], x)


def tft_init(key, nin, nstat, H, heads=2):
    k = iter(jax.random.split(key, 32))
    p = dict(
        emb_t=[_dense(next(k), 1, H) for _ in range(nin)],
        emb_s=[_dense(next(k), 1, H) for _ in range(nstat)],
        vsn_s=_grn_init(next(k), nstat * H, H, nstat), vsn_s_v=[_grn_init(next(k), H, H, H) for _ in range(nstat)],
        ctx=[_grn_init(next(k), H, H, H) for _ in range(4)],          # selection, h0, c0, enrichment
        vsn_t=_grn_init(next(k), nin * H, H, nin, ctx=H), vsn_t_v=[_grn_init(next(k), H, H, H) for _ in range(nin)],
        lstm=_lstm_init(next(k), H, H), gate1=_glu_init(next(k), H),
        enrich=_grn_init(next(k), H, H, H, ctx=H),
        q=[_dense(next(k), H, H // heads) for _ in range(heads)], kk=[_dense(next(k), H, H // heads) for _ in range(heads)],
        v=_dense(next(k), H, H // heads), attn_o=_dense(next(k), H // heads, H), gate2=_glu_init(next(k), H),
        pos=_grn_init(next(k), H, H, H), gate3=_glu_init(next(k), H), out=_dense(next(k), H, 1))
    return p


def tft_apply(p, xs, zs):
    n, Lt, nin = xs.shape
    nstat = zs.shape[1]
    # static covariate encoder
    es = jnp.stack([_ap(p["emb_s"][j], zs[:, j:j + 1]) for j in range(nstat)], 1)          # (n, nstat, H)
    ws = jax.nn.softmax(_grn(p["vsn_s"], es.reshape(n, -1)), -1)
    vs = jnp.stack([_grn(p["vsn_s_v"][j], es[:, j]) for j in range(nstat)], 1)
    cs = (ws[..., None] * vs).sum(1)
    c_sel, c_h, c_c, c_e = [_grn(p["ctx"][j], cs) for j in range(4)]
    # temporal variable selection
    et = jnp.stack([_ap(p["emb_t"][j], xs[..., j:j + 1]) for j in range(nin)], 2)          # (n, L, nin, H)
    wt = jax.nn.softmax(_grn(p["vsn_t"], et.reshape(n, Lt, -1), c_sel[:, None, :]), -1)
    vt = jnp.stack([_grn(p["vsn_t_v"][j], et[:, :, j]) for j in range(nin)], 2)
    xt = (wt[..., None] * vt).sum(2)                                                        # (n, L, H)
    # sequence-to-sequence layer
    hs = _lstm(p["lstm"], xt, c_h, c_c)
    phi = _ln(xt + _glu(p["gate1"], hs))
    th = _grn(p["enrich"], phi, c_e[:, None, :])
    # interpretable multi-head attention (shared values), causal
    mask = jnp.tril(jnp.ones((Lt, Lt), bool))
    v = _ap(p["v"], th)
    heads = []
    for hq, hk in zip(p["q"], p["kk"]):
        a = _ap(hq, th) @ jnp.swapaxes(_ap(hk, th), 1, 2) / math.sqrt(v.shape[-1])
        heads.append(jax.nn.softmax(jnp.where(mask, a, -1e9), -1) @ v)
    att = _ap(p["attn_o"], jnp.mean(jnp.stack(heads), 0))
    de = _ln(th + _glu(p["gate2"], att))
    ps = _grn(p["pos"], de)
    out = _ln(phi + _glu(p["gate3"], ps))
    return _ap(p["out"], out[:, -1])[:, 0]


NETS = dict(lstm=(lstm_init, lstm_apply), tft=(tft_init, tft_apply))


# ------------------------------------------------------------------ training (vmapped over replications x learning rates)
def _bce(logit, y, w=None):
    l = jnp.maximum(logit, 0) - logit * y + jnp.log1p(jnp.exp(-jnp.abs(logit)))
    if w is None:
        return jnp.mean(l)
    return jnp.sum(w * l) / jnp.maximum(jnp.sum(w), 1.0)


def _make_trainer(name, H, nin, nstat, steps=C.NN_STEPS, every=C.NN_EVAL_EVERY):
    """Trainer for padded data: xs (n_max, L, nin), zs, y, w (sequence weights, 0 = padding); the
    padded shapes are the same at every origin, so one compiled trainer per (net, H, chunk size)."""
    init, apply = NETS[name]

    def train(key, lr, xs, zs, y, w, xv, zv, yv, wv):
        key, kb = jax.random.split(key)
        p = init(key, nin, nstat, H)
        m = jax.tree.map(jnp.zeros_like, p)
        v = jax.tree.map(jnp.zeros_like, p)
        B = C.ML_BATCH
        if B:
            # v1.4: minibatch Adam — each step uses B training sequences drawn uniformly (with
            # replacement) from the real (unpadded) ones; full-batch steps cost ~10x more on a CPU
            pw = w / jnp.maximum(w.sum(), 1.0)
            def loss_fn(p, k):
                ix = jax.random.choice(k, w.shape[0], (B,), p=pw)
                return _bce(apply(p, xs[ix], zs[ix]), y[ix])
        else:
            loss_fn = lambda p, k: _bce(apply(p, xs, zs), y, w)
        b1, b2, eps = 0.9, 0.999, 1e-8

        def chunk(carry, _):
            p, m, v, t, best_l, best_p = carry

            def one(c, _):
                p, m, v, t = c
                g = jax.grad(loss_fn)(p, jax.random.fold_in(kb, t.astype(jnp.int32)))
                t = t + 1
                m = jax.tree.map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
                v = jax.tree.map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
                p = jax.tree.map(lambda w_, a, b: w_ - lr * (a / (1 - b1 ** t)) / (jnp.sqrt(b / (1 - b2 ** t)) + eps), p, m, v)
                return (p, m, v, t), None
            (p, m, v, t), _ = jax.lax.scan(one, (p, m, v, t), None, length=every)
            vl = _bce(apply(p, xv, zv), yv, wv)
            better = vl < best_l
            best_p = jax.tree.map(lambda a, b: jnp.where(better, a, b), p, best_p)
            return (p, m, v, t, jnp.where(better, vl, best_l), best_p), vl
        carry = (p, m, v, 0.0, jnp.inf, p)
        (p, m, v, t, best_l, best_p), _ = jax.lax.scan(chunk, carry, None, length=steps // every)
        return best_p, best_l
    return jax.jit(jax.vmap(jax.vmap(train, in_axes=(0, 0, None, None, None, None, None, None, None, None)),
                            in_axes=(0, None, 0, 0, 0, 0, 0, 0, 0, 0)))


_TRAINERS = {}
_PREDICT = {}


def _pad(a, n):
    """Pad the leading axis of a (n_seq, ...) array with zeros to length n; returns (padded, weights)."""
    w = np.zeros(n, np.float32)
    w[:a.shape[0]] = 1.0
    out = np.zeros((n,) + a.shape[1:], a.dtype)
    out[:a.shape[0]] = a
    return out, w


def n_seq_max(I, n_months_max=None):
    """Padded number of training sequences: the largest training set of any origin."""
    n_months_max = n_months_max or len(split_months(C.ORIGINS[-1])[0])
    return I * n_months_max


def nn_predict_batch(name, sims_N, sims_X, sims_Z, ys, t, pred_months, seed):
    """Train `name` (lstm | tft) for a batch of replications at origin t; returns probs
    (R, len(pred_months), I) and the chosen configuration per replication. Data are padded to the
    largest origin's training-set size (and to the longest prediction segment) so that every origin
    reuses the same compiled trainer."""
    tr, va = split_months(t)
    R, I = len(sims_N), sims_N[0].shape[0]
    n_tr = n_seq_max(I)
    n_va = I * C.ML_VAL_MONTHS
    n_pr = I * max(C.ORIGINS[1] - C.ORIGINS[0], C.DECISION_END - C.ORIGINS[-1] + 1)
    build = lambda months, r: (np.concatenate([seq_features(sims_N[r], sims_X[r], s) for s in months]),
                               np.concatenate([sims_Z[r]] * len(months)).astype(np.float32),
                               np.concatenate([ys[r][s] for s in months]))

    def stack_pad(months, n):
        X_, Z_, Y_, W_ = [], [], [], []
        for r in range(R):
            x, z, y = build(months, r)
            xp, w = _pad(x, n); zp, _ = _pad(z, n); yp, _ = _pad(y, n)
            X_.append(xp); Z_.append(zp); Y_.append(yp); W_.append(w)
        return [jnp.asarray(np.stack(a)) for a in (X_, Z_, Y_, W_)]
    xs, zs, y, w = stack_pad(tr, n_tr)
    xv, zv, yv, wv = stack_pad(va, n_va)
    pr_x, pr_z, _, pr_w = stack_pad(pred_months, n_pr)
    nin, nstat = xs.shape[-1], zs.shape[-1]
    lrs = jnp.asarray(C.NN_LR, jnp.float32)
    best = None
    for H in C.NN_HIDDEN:
        key = (name, H, nin, nstat, xs.shape, xv.shape)
        if key not in _TRAINERS:
            _TRAINERS[key] = _make_trainer(name, H, nin, nstat)
        keys = jax.random.split(jax.random.PRNGKey(seed + H), R * len(C.NN_LR)).reshape(R, len(C.NN_LR), -1)
        params, vloss = _TRAINERS[key](keys, lrs, xs, zs, y, w, xv, zv, yv, wv)      # vloss (R, nlr)
        pkey = (name, H, pr_x.shape)
        if pkey not in _PREDICT:
            apply = NETS[name][1]
            _PREDICT[pkey] = jax.jit(jax.vmap(jax.vmap(apply, in_axes=(0, None, None))))
        logits = _PREDICT[pkey](params, pr_x, pr_z)
        vl = np.asarray(vloss)
        pr = np.asarray(jax.nn.sigmoid(logits))                               # (R, nlr, n_pr)
        for r in range(R):
            j = int(np.nanargmin(vl[r]))
            if best is None:
                best = [None] * R
            if best[r] is None or vl[r, j] < best[r][0]:
                best[r] = (float(vl[r, j]), dict(hidden=H, lr=float(C.NN_LR[j])), pr[r, j])
    n_real = len(pred_months) * I
    probs = np.stack([b[2][:n_real].reshape(len(pred_months), I) for b in best])
    cfg = [dict(b[1], val_logloss=round(b[0], 4)) for b in best]
    return probs.astype(np.float32), cfg
