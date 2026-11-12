"""backtest_multi.py — rolling-origin backtest of the two-channel alarm and its baselines on a real multi-item panel
(build_panel_multi.py), evaluated against the trade-based events (N1), the independent restriction events (N2) and
their union (N3).

Protocol (the Study B protocol of the paper, Section 5.3): origins every REFIT months from ORIGIN0; the hierarchical
model M2 (and M1, M0 if requested) is fitted at each origin on the data up to it (gp_ews.fit: NUTS, two chains of 500
warm-up and 600 draws started near the prior medians; a fit whose rank-normalised split-R-hat reaches 1.05 or whose
bulk ESS falls below 100 on any stored quantity is repeated once with 1,000 warm-up and 1,000 draws and kept either
way, fits/diagnostics.csv), propagated month by month as particles between origins, and read every month: p_it(3) thresholded at pi = (0.048, 0.167, 0.5) with 2-of-2
persistence, and the level-shift probability q thresholded at the bootstrap-calibrated value from the first origin.
Baselines at the same decision months: sigma-rule on the plug-in SRI (24-month window, limits 1, 1.5, 2 sd), EWMA and
CUSUM on counts (Poisson-calibrated to a nominal ARL0 of 24 with in-control means from months 13-36), a Farrington-type
threshold (36-month quasi-Poisson, NB predictive quantile), and XGBoost trained on the pooled panel with the same
indicators and the count history, re-trained at every origin and thresholded at the same pi levels, plus a variant recalibrated by
isotonic regression on the twelve labelled months before each origin (xgbcal).

Metrics mirror code/metrics.py: dr_h1/dr_h3 (share of event item-months with the level in force in the month before /
in the three months before), false onsets per item-year (onsets not followed by an event within three months), alarm
months per item-year, lead time; AUC-PR, Brier and ECE of the three-month event probability (an event in months t..t+2, the window of p_it(3)
dated t) for the probability methods; expected cost per item-year under Model A for r on a log grid, with never/always acting.

usage: python backtest_multi.py PANEL_CSV OUT_DIR [--fit-target N1] [--origin0 36] [--refit 6] [--h 3]
                                [--methods M2,M0,sigma,ewma,cusum,farr,xgb,xgbcal] [--fast] [--boot 120] [--seed 0]
  --fit-target  which event series the count models are fitted on (N1 by default; N3 = union)
  --fast        short MCMC (200 warm-up, 300 draws, 1 chain) for a smoke test
Outputs: fits/fit_<t>_<variant>.npz, paths.npz (p, q, levels per method), metrics_<target>.csv, cost_<target>.csv,
         cost_bands_<target>.csv, events_N2_detail.csv, summary.json, fig_items_with_restrictions.png
"""
import argparse, csv, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("panel"); ap.add_argument("out")
ap.add_argument("--fit-target", default="N1"); ap.add_argument("--origin0", type=int, default=36); ap.add_argument("--refit", type=int, default=6)
ap.add_argument("--h", type=int, default=3); ap.add_argument("--methods", default="M2,M0,sigma,ewma,cusum,farr,xgb,xgbcal")
ap.add_argument("--fast", action="store_true"); ap.add_argument("--boot", type=int, default=120); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gp-ews", default=os.environ.get("GP_EWS", HERE), help="folder holding gp_ews.py (default: this folder)")
ap.add_argument("--code", default=os.environ.get("CODE_DIR", os.path.join(HERE, "..", "code")), help="repository code/ (charts.py, config.py)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); os.makedirs(os.path.join(a.out, "fits"), exist_ok=True)
sys.path.insert(0, a.gp_ews); sys.path.insert(0, a.code)
import gp_ews as G                                   # the self-contained implementation of the model (backtest/gp_ews.py)
import mcmc_diag as D                                # convergence diagnostics (R-hat, ESS)
import charts as CH                                  # EWMA / CUSUM / sigma-rule of the simulation study (imports config only)
import config as C
from scipy.stats import nbinom, poisson
from sklearn.metrics import average_precision_score

METHODS = a.methods.split(",")


def propagate_safe(pack, N_new, X_new, m_base, rng, ess_frac=0.5):
    """gp_ews.propagate with a guard: if every particle has zero likelihood (underflow, e.g. the Poisson variant meeting a
    large count), the weights are left uniform instead of becoming NaN."""
    S = pack["S"]; g = pack["group"]
    pack["m"] = pack["m"] + pack["d"] + pack["sig"][:, None] * rng.standard_normal((S, pack["m"].shape[1]))
    pack["d"] = pack["d"] + pack["tau"][:, None] * rng.standard_normal((S, pack["d"].shape[1]))
    mu = np.exp(np.clip(pack["m"][:, g] + (pack["x_prev"] @ pack["beta"].T).T + pack["logC"], -20, 10))
    al = pack["alpha"][:, g]
    logw = nbinom.logpmf(N_new[None, :], al, al / (al + mu)).sum(1)
    if np.isfinite(logw).any():
        logw = logw - logw[np.isfinite(logw)].max(); w = np.exp(np.where(np.isfinite(logw), logw, -np.inf)); w /= w.sum()
    else:
        w = np.full(S, 1.0 / S)
    if 1.0 / (w ** 2).sum() < ess_frac * S:
        idx = G.systematic(w, rng)
        for k in ("m", "d", "beta", "logC", "alpha", "sig", "tau"):
            pack[k] = pack[k][idx]
        w = np.full(S, 1.0 / S)
    pack["w"] = w; pack["x_prev"] = X_new
    return np.asarray((w[:, None] * ((pack["m"] - m_base[None, :]) > np.log(G.RHO_SHIFT))).sum(0))


G.propagate = propagate_safe                                       # used by walk() and by the bootstrap calibration below


def calibrate_q_safe(pack0, x_hold, months, target_arl=24, B=200, seed=1):
    """Bootstrap ARL0 calibration as gp_ews.calibrate_q (stable paths simulated from the posterior at the first origin),
    extended as in the simulation code: thresholds on the grid 0.50-0.99, persistence strengthened from 2 to 3 to 4
    consecutive months if no threshold reaches the target. Returns (threshold, persistence, implied ARL0)."""
    rng = np.random.default_rng(seed); grid = np.round(np.arange(0.50, 0.995, 0.05), 2)
    S = pack0["S"]; g = pack0["group"]; Gn = pack0["m"].shape[1]; qs = np.zeros((B, months))
    for b in range(B):
        j = rng.integers(S); m = pack0["m"][j].copy(); sig = pack0["sig"][j]; beta, logC, al = pack0["beta"][j], pack0["logC"][j], pack0["alpha"][j]
        sub = {k: pack0[k].copy() for k in ("m", "d", "beta", "logC", "alpha", "sig", "tau")}
        sub.update(S=S, group=g, x_prev=x_hold.copy(), w=np.full(S, 1 / S)); m_base = pack0["m"].mean(0)
        for t in range(months):
            m = m + sig * rng.standard_normal(Gn); mu = np.exp(np.clip(m[g] + x_hold @ beta + logC, -20, 10))
            Nn = rng.negative_binomial(al[g], al[g] / (al[g] + mu)); qs[b, t] = propagate_safe(sub, Nn, x_hold, m_base, rng).max()
    for persist in (2, 3, 4):
        for c in grid:
            hit = qs >= c; pers = hit.copy()
            for r_ in range(1, persist):
                pers[:, r_:] &= hit[:, :-r_]; pers[:, :r_] = False
            al0 = qs.size / max(pers.sum(), 1)
            if al0 >= target_arl:
                return float(c), persist, float(al0)
    return float(grid[-1]), 4, float(qs.size / max(pers.sum(), 1))
PIS = G.PI_LEVELS
R_GRID = np.exp(np.linspace(np.log(1.0), np.log(100.0), 400))
R_REPORT = [2, 5, 10, 20, 50]
rng = np.random.default_rng(a.seed)
log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)

# ------------------------------------------------------------------ panel
rows = list(csv.DictReader(open(a.panel, encoding="utf-8")))
items = sorted({r["cmd"] for r in rows}); periods = sorted({r["period"] for r in rows})
I, T, K = len(items), len(periods), 4
groups_named = sorted({r["group"] for r in rows}); gidx = {g: k for k, g in enumerate(groups_named)}
group = np.zeros(I, int); N = {k: np.zeros((I, T), int) for k in ("N1", "N2", "N2a", "N3")}
raw = np.zeros((I, T, K)); qty = np.zeros((I, T)); uv = np.full((I, T), np.nan); have = np.zeros((I, T), bool)
ev2 = {}
for r in rows:
    i, t = items.index(r["cmd"]), periods.index(r["period"])
    group[i] = gidx[r["group"]]; have[i, t] = True
    for k in N:
        N[k][i, t] = int(r[k])
    qty[i, t] = float(r["total_qty_kg"]); uv[i, t] = float(r["unit_value"]) if r["unit_value"] else np.nan
    raw[i, t, 0] = float(r["hhi12"]); raw[i, t, 1] = float(r["top_share12"])
    if int(r["N2"]) > 0:
        ev2[(i, t)] = r["events2"]
for i in range(I):                                                  # LOCF for unit values, zeros where never observed
    last = np.nan
    for t in range(T):
        if np.isnan(uv[i, t]):
            uv[i, t] = last
        last = uv[i, t]
uv = np.where(np.isnan(uv), np.nanmean(uv) if np.isfinite(np.nanmean(uv)) else 1.0, uv)
with np.errstate(divide="ignore", invalid="ignore"):
    raw[:, 12:, 2] = np.log(uv[:, 12:]) - np.log(uv[:, :-12])
    raw[:, 3:, 3] = np.log(qty[:, 3:] + 1) - np.log(qty[:, :-3] + 1)
raw = np.where(np.isfinite(raw), raw, 0.0)
z = np.stack([raw[:, :36, 1].mean(1), np.log(np.maximum(np.nanmean(uv[:, :36], 1), 1e-6))], 1)
for g in range(len(groups_named)):                                   # centred within group
    z[group == g] -= z[group == g].mean(0)
Nfit = N[a.fit_target]
ORIGINS = list(range(a.origin0, T, a.refit))
DEC = np.arange(a.origin0, T)                                        # decision months (0-based)
log(f"panel {I} items, {T} months, {len(groups_named)} groups {groups_named}; origins {ORIGINS[0]}..{ORIGINS[-1]} ({len(ORIGINS)}); fit target {a.fit_target}")


def xmat(upto):
    mu_, sd = raw[:, 12:upto].reshape(-1, K).mean(0), raw[:, 12:upto].reshape(-1, K).std(0) + 1e-9
    Xs = np.clip((raw - mu_) / sd, -5, 5)
    return np.concatenate([np.zeros((I, 1, K)), Xs[:, :-1]], 1)         # lagged one month


# ------------------------------------------------------------------ model-based channels (M2 / M1 / M0)
RHAT_MAX, ESS_MIN = 1.05, 100          # convergence check per fit (rank-normalised split-R-hat, bulk ESS; Supplementary Note 14)


def _stored(s):
    return dict(beta=s["beta"], logC=s["logC"], alpha=s["alpha_g"], m=s["m_t"][:, :, -1], d=s["delta_t"][:, :, -1],
                m_path_mean=s["m_t"].mean(0), sig=s.get("sigma_eta", np.zeros(s["beta"].shape[0])), tau=s.get("tau_zeta", np.zeros(s["beta"].shape[0])))


def _record_diag(t, variant, attempt, kw, res, wr, we, ok):
    path = os.path.join(a.out, "fits", "diagnostics.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["origin", "variant", "attempt", "warmup", "samples", "chains", "max_rhat", "min_ess", "pass"] +
                       [f"rhat_{k}" for k in D.CHECK] + [f"ess_{k}" for k in D.CHECK])
        w.writerow([t, variant, attempt, kw["warmup"], kw["samples"], kw["chains"], round(wr, 4), round(we), int(ok)] +
                   [round(res[k][0], 4) if k in res else "" for k in D.CHECK] + [round(res[k][1]) if k in res else "" for k in D.CHECK])


def fit_origin(t, variant):
    """fit at origin t unless a fit file exists; every fit (new or existing) is checked (rank-normalised split-R-hat below
    RHAT_MAX and bulk ESS at least ESS_MIN on the stored quantities) and a failing fit is repeated once with 1,000 warm-up
    and 1,000 draws per chain (seed + 1000) and kept whether or not the repeat passes; diagnostics go to fits/diagnostics.csv."""
    path = os.path.join(a.out, "fits", f"fit_{t}_{variant}.npz")
    marker = path + ".refit"
    Xl = xmat(t)
    mu0 = np.log(max(Nfit[:, :12].mean(), 0.02))
    kw = dict(warmup=200, samples=300, chains=1) if a.fast else dict(warmup=500, samples=600, chains=2)
    if os.path.exists(path):
        if os.path.exists(marker) or os.path.exists(path + ".checked") or a.fast:
            return
        f = np.load(path); st = {k: f[k] for k in f.files}
        ok, res, wr, we = D.check(st, kw["chains"], RHAT_MAX, ESS_MIN)
        _record_diag(t, variant, 0, kw, res, wr, we, ok)                  # attempt 0: an existing fit file, checked
        if ok:
            open(path + ".checked", "w").write(f"max R-hat {wr:.4f}, min ESS {we:.0f}\n")
            return
        log(f"  existing fit {variant} at origin {t} fails the check (max R-hat {wr:.3f}, min ESS {we:.0f}): repeating")
    else:
        s = G.fit(Nfit[:, :t], Xl[:, :t], z, group, mu0, variant, seed=100 + t, **kw)
        st = _stored(s); np.savez(path, **st)
        ok, res, wr, we = D.check(st, kw["chains"], RHAT_MAX, ESS_MIN)
        _record_diag(t, variant, 1, kw, res, wr, we, ok)
        log(f"  fit {variant} at origin {t}: alpha 90% {np.percentile(st['alpha'], [5, 95]).round(2)} beta med {np.median(st['beta'], 0).round(2)}"
            f" | max R-hat {wr:.3f}, min ESS {we:.0f}{'' if ok else ' -> FAIL, repeating with 1000/1000'}")
        if ok or a.fast:
            if ok:
                open(path + ".checked", "w").write(f"max R-hat {wr:.4f}, min ESS {we:.0f}\n")
            return
    kw2 = dict(warmup=1000, samples=1000, chains=2)
    s = G.fit(Nfit[:, :t], Xl[:, :t], z, group, mu0, variant, seed=1100 + t, **kw2)
    st = _stored(s); np.savez(path, **st); open(marker, "w").write("repeated with 1000 warm-up and 1000 draws per chain\n")
    ok, res, wr, we = D.check(st, kw2["chains"], RHAT_MAX, ESS_MIN)
    _record_diag(t, variant, 2, kw2, res, wr, we, ok)
    log(f"  repeat fit {variant} at origin {t}: alpha 90% {np.percentile(st['alpha'], [5, 95]).round(2)} beta med {np.median(st['beta'], 0).round(2)}"
        f" | max R-hat {wr:.3f}, min ESS {we:.0f}{' (still failing; kept)' if not ok else ''}")


def pack_from(t, variant):
    f = np.load(os.path.join(a.out, "fits", f"fit_{t}_{variant}.npz"))
    S = f["beta"].shape[0]; Xl = xmat(t)
    return dict(S=S, beta=f["beta"], logC=f["logC"], alpha=f["alpha"], m=f["m"], d=f["d"], sig=f["sig"], tau=f["tau"], group=group,
                x_prev=Xl[:, t].copy(), w=np.full(S, 1 / S)), f


def walk(variant):
    """monthly p (T,I), q (T,), levels (T,I), q-alarm (T,) for one model variant."""
    for t in ORIGINS:
        fit_origin(t, variant)
    f0 = np.load(os.path.join(a.out, "fits", f"fit_{ORIGINS[0]}_{variant}.npz"))
    m_base = f0["m"].mean(0)
    pack0, _ = pack_from(ORIGINS[0], variant)
    qthr, qpers, arl0 = calibrate_q_safe({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in pack0.items()}, pack0["x_prev"],
                                         months=min(60, T - ORIGINS[0]), B=a.boot, seed=3 + a.seed)
    Xl_full = xmat(ORIGINS[-1])
    p = np.full((T, I), np.nan); q = np.full(T, np.nan); pack = None; r_ = np.random.default_rng(7 + a.seed)
    for t in range(ORIGINS[0], T):
        if t in ORIGINS:
            pack, _ = pack_from(t, variant)
        else:
            q[t] = float(np.max(G.propagate(pack, Nfit[:, t - 1], Xl_full[:, t], m_base, r_)))   # reweight on last month's counts
        w = pack["w"]
        p[t] = G.weighted_p(pack, h=a.h)
        if t in ORIGINS:
            q[t] = float(np.max((w[:, None] * ((pack["m"] - m_base[None, :]) > np.log(G.RHO_SHIFT))).sum(0)))
    lev = G.alarm_levels(np.nan_to_num(p, nan=0.0))
    qa = np.nan_to_num(q, nan=0.0) >= qthr
    q_alarm = qa.copy()
    for r_ in range(1, qpers):                                         # qpers-of-qpers persistence
        sh = np.zeros_like(qa); sh[r_:] = qa[:-r_]; q_alarm &= sh
    head = lev.copy(); head[q_alarm] = np.maximum(head[q_alarm], 2)
    return dict(p=p, q=q, levels=lev, q_alarm=q_alarm, headline=head, q_threshold=qthr, q_persistence=qpers, arl0_boot=arl0)


# ------------------------------------------------------------------ baselines
def plug_sri(upto_excl, win=24):
    """plug-in SRI from the trailing `win` months ending at upto_excl-1: NB with moment-matched dispersion (Poisson if var <= mean)."""
    seg = Nfit[:, max(0, upto_excl - win):upto_excl].astype(float)
    m = seg.mean(1); v = seg.var(1, ddof=1) if seg.shape[1] > 1 else m
    with np.errstate(divide="ignore", invalid="ignore"):
        al = np.where(v > m, m ** 2 / np.maximum(v - m, 1e-9), np.inf)
    return np.where(np.isfinite(al), (1 + m / np.where(np.isfinite(al), al, 1)) ** (-np.where(np.isfinite(al), al, 1)), np.exp(-m))


def sigma_levels():
    sri = np.vstack([plug_sri(t) for t in range(12, T)])                 # SRI known at the end of month t-1, dated t
    hist = sri[DEC[0] - 12 - C.SIGMA_WINDOW:DEC[0] - 12]
    return CH.sigma_rule(sri[DEC[0] - 12:], hist)                         # (M, I)


def farrington_alarms(Nc, t_first, t_last, W=36, level=1 - 1 / 24, GAMMA=2.58, MU_FLOOR=0.25 / 24, MIN_HIST=12):
    """Farrington-type threshold (copy of followup/a3_farrington.py: 36-month quasi-Poisson, Anscombe down-weighting,
    NB predictive (1 - 1/24) quantile); alarms (M, I) at decision months t_first..t_last."""
    def glm(y, s, w, trend):
        I_, n = y.shape
        X = np.column_stack([np.ones(n), s]) if trend else np.ones((n, 1))
        b = np.zeros((I_, X.shape[1])); b[:, 0] = np.log(np.maximum((w * y).sum(1) / np.maximum(w.sum(1), 1e-9), MU_FLOOR))
        for _ in range(25):
            eta = b @ X.T; mu = np.exp(np.clip(eta, -30, 30)); zz = eta + (y - mu) / mu; Wt = w * mu
            XtW = X.T[None] * Wt[:, None, :]; A = XtW @ X + 1e-9 * np.eye(X.shape[1]); rhs = np.einsum("ipn,in->ip", XtW, zz)
            b_new = np.linalg.solve(A, rhs[..., None])[..., 0]
            if np.abs(b_new - b).max() < 1e-8:
                b = b_new; break
            b = b_new
        eta = b @ X.T; mu = np.exp(np.clip(eta, -30, 30)); df = np.maximum(w.sum(1) - X.shape[1], 1.0)
        phi = np.maximum((w * (y - mu) ** 2 / mu).sum(1) / df, 1.0); cov = np.linalg.inv(A)
        se_b = np.sqrt(np.maximum(phi, 1.0) * cov[:, -1, -1]) if trend else np.zeros(I_)
        return b, mu, phi, se_b
    I_, T_ = Nc.shape; months = np.arange(t_first, t_last + 1); A = np.zeros((len(months), I_), bool)
    for mi, t in enumerate(months):
        cur = t - 1; h0 = max(cur - W, 0); y = Nc[:, h0:cur].astype(float); n = y.shape[1]
        s = (np.arange(h0, cur) - cur).astype(float); w = np.ones_like(y)
        if n < MIN_HIST:
            mu_hat = np.maximum(y.mean(1), MU_FLOOR); phi = np.ones(I_)
        else:
            b, mu, phi, se_b = glm(y, s, w, True)
            r = 1.5 * (y ** (2 / 3) - mu ** (2 / 3)) / mu ** (1 / 6) / np.sqrt(phi)[:, None]; w = np.ones_like(y); big = np.abs(r) > GAMMA
            w[big] = (GAMMA / np.abs(r[big])) ** 2
            b, mu, phi, se_b = glm(y, s, w, True); keep = np.abs(b[:, 1]) > 1.96 * np.maximum(se_b, 1e-12)
            b0, mu0, phi0, _ = glm(y, s, w, False)
            mu_hat = np.maximum(np.where(keep, np.exp(np.clip(b[:, 0], -30, 30)), np.exp(np.clip(b0[:, 0], -30, 30))), MU_FLOOR)
            phi = np.where(keep, phi, phi0)
        al = np.where(phi > 1.0, mu_hat / np.maximum(phi - 1.0, 1e-12), np.inf)
        fin = np.isfinite(al); al_f = np.where(fin, al, 1.0)
        qq = np.where(fin, nbinom.ppf(level, al_f, al_f / (al_f + mu_hat)), poisson.ppf(level, mu_hat))
        A[mi] = Nc[:, cur] > qq
    return A


def xgb_probs(recal=None):
    """pooled XGBoost: at each origin, train on all (item, month) rows with a complete 3-month label, predict until the next origin.
    recal: None, or "isotonic" / "platt" -- the last twelve labelled months before the origin are held out of training and the
    model's probabilities are recalibrated on them (isotonic regression, or Platt scaling of the logit), then applied until the
    next origin (the recalibration variant of Section 6.2)."""
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    Xl = xmat(T)
    feats = np.zeros((I, T, K + 6))
    for t in range(T):
        feats[:, t, :K] = Xl[:, t]
        for l_ in (1, 2, 3):
            feats[:, t, K + l_ - 1] = Nfit[:, t - l_] if t - l_ >= 0 else 0
        lo = max(0, t - 12); feats[:, t, K + 3] = Nfit[:, lo:t].mean(1) if t > 0 else 0; feats[:, t, K + 4] = Nfit[:, lo:t].max(1) if t > 0 else 0
        feats[:, t, K + 5] = group
    y = np.zeros((I, T));                                               # label at t: an event in months t..t+h-1 (the window of p_it(h), dated t on data through t-1)
    for t in range(T - a.h + 1):
        y[:, t] = (Nfit[:, t:t + a.h].sum(1) >= 1)
    p = np.full((T, I), np.nan)
    for oi, t in enumerate(ORIGINS):
        t_end = ORIGINS[oi + 1] if oi + 1 < len(ORIGINS) else T
        tr_t = np.arange(12, t - a.h + 1)                                 # rows whose label window ends at or before month t-1
        va_t = None
        if recal and len(tr_t) > 24:
            va_t, tr_t = tr_t[-12:], tr_t[:-12]                               # inner validation: the last twelve labelled months
        Xtr = feats[:, tr_t].reshape(-1, K + 6); ytr = y[:, tr_t].reshape(-1)
        if ytr.sum() < 5 or (1 - ytr).sum() < 5:
            p[t:t_end] = ytr.mean(); continue
        mdl = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                                reg_lambda=1.0, n_jobs=2, random_state=a.seed, verbosity=0)
        mdl.fit(Xtr, ytr)
        cal = lambda q: q
        if va_t is not None:
            pv = mdl.predict_proba(feats[:, va_t].reshape(-1, K + 6))[:, 1]; yv = y[:, va_t].reshape(-1)
            if yv.sum() >= 5 and (1 - yv).sum() >= 5:
                if recal == "isotonic":
                    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(pv, yv); cal = iso.predict
                else:
                    lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
                    lr = LogisticRegression(C=1e6).fit(lg(pv).reshape(-1, 1), yv); cal = lambda q: lr.predict_proba(lg(q).reshape(-1, 1))[:, 1]
        for tt in range(t, t_end):
            p[tt] = cal(mdl.predict_proba(feats[:, tt])[:, 1])
    return p


# ------------------------------------------------------------------ metrics (code/metrics.py definitions)
def event_metrics(A, months, Nc):
    shock = Nc >= 1; s0, s1 = months[0], months[-1]; res = {}
    for h in (1, 2, 3):
        det = tot = 0
        for tau in range(s0 + h, s1 + 2):
            if tau > T:
                break
            hit = shock[:, tau - 1]
            if not hit.any():
                continue
            win = A[(tau - h - s0):(tau - s0)]; det += (hit & win.any(0)).sum(); tot += hit.sum()
        res[f"dr_h{h}"] = round(det / tot, 4) if tot else None
    prev = np.vstack([np.zeros((1, A.shape[1]), bool), A[:-1]]); onset = A & ~prev; leads = []; fa = 0
    for m_idx, i in zip(*np.nonzero(onset)):
        t = months[m_idx]; nxt = np.flatnonzero(shock[i, t:])
        if nxt.size:
            leads.append(int(nxt[0] + 1))
        if not shock[i, t:t + 3].any():
            fa += 1
    years = A.shape[1] * len(months) / 12.0
    res.update(lead_med=float(np.median(leads)) if leads else None, lead_q25=float(np.percentile(leads, 25)) if leads else None,
               lead_q75=float(np.percentile(leads, 75)) if leads else None, n_onsets=int(onset.sum()), fa_per_itemyear=round(fa / years, 4),
               alarm_share=round(float(A.mean()), 4), alarm_months_per_itemyear=round(12 * float(A.mean()), 3))
    return res


def prob_metrics(p, y):
    p = np.clip(p.ravel(), 0, 1); y = y.ravel().astype(bool)
    res = dict(brier=round(float(((p - y) ** 2).mean()), 5), base_rate=round(float(y.mean()), 4), mean_p=round(float(p.mean()), 4))
    res["aucpr"] = round(float(average_precision_score(y, p)), 4) if y.any() and not y.all() else None
    bins = np.minimum((p * 10).astype(int), 9); ece = 0.0
    for b in range(10):
        k = bins == b
        if k.any():
            ece += k.mean() * abs(p[k].mean() - y[k].mean())
    res["ece"] = round(float(ece), 5)
    return res


def label3(Nc):
    """event label scored against the probabilities dated t (data through t-1): at least one event in months t..t+h-1,
    the window of p_it(h) (gp_ews.weighted_p steps 1..h from the level of month t-1)."""
    y = np.zeros((T, I))
    for t in range(T):
        y[t] = Nc[:, t:t + a.h].sum(1) >= 1
    return y


# ------------------------------------------------------------------ run
paths = {}
t0 = time.time()
for v in [m for m in ("M2", "M1", "M0") if m in METHODS]:
    log(f"model {v}: fitting {len(ORIGINS)} origins")
    paths[v] = walk(v)
    log(f"model {v} done ({(time.time() - t0) / 60:.1f} min); q threshold {paths[v]['q_threshold']:.2f} with {paths[v]['q_persistence']}-of-{paths[v]['q_persistence']} persistence, bootstrap ARL0 {paths[v]['arl0_boot']:.0f}")
if "sigma" in METHODS:
    paths["sigma"] = dict(levels_dec=sigma_levels())
if "ewma" in METHODS or "cusum" in METHODS:
    ch = CH.chart_alarms(Nfit, dgp_seed := 11 + a.seed, int(DEC[0]), int(DEC[-1]))
    paths["ewma"] = dict(alarm_dec=ch["ewma"]); paths["cusum"] = dict(alarm_dec=ch["cusum"])
if "farr" in METHODS:
    paths["farr"] = dict(alarm_dec=farrington_alarms(Nfit, int(DEC[0]), int(DEC[-1])))
if "xgb" in METHODS:
    pX = xgb_probs(); paths["xgb"] = dict(p=pX, levels=G.alarm_levels(np.nan_to_num(pX, nan=0.0)))
    log("xgb done")
if "xgbcal" in METHODS:
    pX = xgb_probs("isotonic"); paths["xgbcal"] = dict(p=pX, levels=G.alarm_levels(np.nan_to_num(pX, nan=0.0)))
    log("xgbcal (isotonic recalibration on the twelve months before each origin) done")

for name, P in paths.items():
    if "p" in P:
        nn = int(np.isnan(P["p"][DEC]).sum())
        if nn:
            log(f"warning: {name} has {nn} NaN probabilities at decision months (filled with 0)")
            P["p"] = np.nan_to_num(P["p"], nan=0.0)

# alarm indicator at decision months per method and level
def alarms_of(name, k):
    P = paths[name]
    if "levels" in P:
        return (P["levels"][DEC] >= k)
    if "levels_dec" in P:
        return (P["levels_dec"] >= k)
    return P["alarm_dec"]


E_SERIES = {"N1": N["N1"], "N2": N["N2"], "N3": N["N3"]}
summary = dict(items=items, groups=groups_named, T=T, origins=ORIGINS, fit_target=a.fit_target,
               q_threshold={v: paths[v]["q_threshold"] for v in paths if "q_threshold" in paths[v]},
               q_persistence={v: paths[v]["q_persistence"] for v in paths if "q_persistence" in paths[v]},
               arl0_boot={v: round(paths[v]["arl0_boot"], 1) for v in paths if "arl0_boot" in paths[v]},
               events={k: int(E_SERIES[k][:, DEC[0]:].sum()) for k in E_SERIES}, item_years=round(I * len(DEC) / 12, 2))
for tgt, Nc in E_SERIES.items():
    y3 = label3(Nc)[DEC]
    E = float((Nc[:, DEC[0] + 1:] >= 1).sum()) / (I * len(DEC) / 12.0)                  # realised event item-months per item-year
    rows_m, rows_c = [], []
    for name in paths:
        levels_k = [1, 2, 3] if ("levels" in paths[name] or "levels_dec" in paths[name]) else [None]
        for k in levels_k:
            A = alarms_of(name, k if k else 1)
            em = event_metrics(A, DEC, Nc)
            rec = dict(target=tgt, method=name, level=k, **em)
            if "p" in paths[name]:
                rec.update(prob_metrics(paths[name]["p"][DEC], y3) if k in (None, 2) else {})
            if name in ("M2", "M1", "M0") and k == 2:
                Aq = np.repeat(paths[name]["q_alarm"][DEC][:, None], I, 1)
                rec.update({"q_" + kk: vv for kk, vv in event_metrics(Aq, DEC, Nc).items() if kk in ("dr_h1", "dr_h3", "alarm_share", "n_onsets", "lead_med")})
                Ah = paths[name]["headline"][DEC] >= 2
                rec.update({"headline_" + kk: vv for kk, vv in event_metrics(Ah, DEC, Nc).items() if kk in ("dr_h1", "fa_per_itemyear", "alarm_months_per_itemyear")})
            rows_m.append(rec)
            am = 12 * float(A.mean()); dr = em["dr_h1"] or 0.0
            for r in R_REPORT:
                rows_c.append(dict(target=tgt, method=name, level=k, r=r, EC=round(am + r * (1 - dr) * E, 4), AM=round(am, 4), DR=dr, E=round(E, 4)))
    for r in R_REPORT:
        rows_c.append(dict(target=tgt, method="never", level=None, r=r, EC=round(r * E, 4), AM=0, DR=0, E=round(E, 4)))
        rows_c.append(dict(target=tgt, method="always", level=None, r=r, EC=12.0, AM=12, DR=1, E=round(E, 4)))
    # envelope curves and the band where M2 beats both trivial policies
    bands = []
    for name in paths:
        levels_k = [1, 2, 3] if ("levels" in paths[name] or "levels_dec" in paths[name]) else [None]
        curves = []
        for k in levels_k:
            A = alarms_of(name, k if k else 1); am = 12 * float(A.mean()); dr = event_metrics(A, DEC, Nc)["dr_h1"] or 0.0
            curves.append(am + R_GRID * (1 - dr) * E)
        env = np.min(np.vstack(curves), 0); triv = np.minimum(R_GRID * E, 12.0)
        beats = (env < R_GRID * E) & (env < 12.0); sav = 1 - env / triv; i_ = int(np.argmax(sav))
        bands.append(dict(target=tgt, method=name, band_lo=round(float(R_GRID[beats].min()), 2) if beats.any() else None,
                          band_hi=round(float(R_GRID[beats].max()), 2) if beats.any() else None, max_saving=round(float(sav[i_]), 4), r_at_max=round(float(R_GRID[i_]), 2)))
    for fn, rr in ((f"metrics_{tgt}.csv", rows_m), (f"cost_{tgt}.csv", rows_c), (f"cost_bands_{tgt}.csv", bands)):
        keys = sorted({kk for r_ in rr for kk in r_}, key=lambda s: (s not in ("target", "method", "level", "r"), s))
        with open(os.path.join(a.out, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rr)
    summary[f"E_{tgt}"] = round(E, 4)

# restriction events in detail: the posture in force at each implementation month t (dated t, set on data through t-1,
# the convention of dr_h1) and the one in force the month before (dated t-1); months since the item first reached level 2
det = []
for (i, t), txt in sorted(ev2.items()):
    if t < DEC[0] + 1:
        continue
    rec = dict(item=items[i], group=groups_named[group[i]], month=periods[t], measure=txt)
    for name in paths:
        if "levels" in paths[name]:
            L = paths[name]["levels"]; rec[f"{name}_level_at"] = int(L[t, i]); rec[f"{name}_level_tm1"] = int(L[t - 1, i])
            rec[f"{name}_max_level_tm2_t"] = int(L[max(DEC[0], t - 2):t + 1, i].max())
            if "q_alarm" in paths[name]:
                rec[f"{name}_q_alarm_at"] = bool(paths[name]["q_alarm"][t]); rec[f"{name}_p_at"] = round(float(paths[name]["p"][t, i]), 3)
            first = np.flatnonzero(L[DEC[0]:t + 1, i] >= 2)
            rec[f"{name}_months_alert_before"] = int(t - (DEC[0] + first[0])) if first.size else None
        elif "levels_dec" in paths[name]:
            rec[f"{name}_level_at"] = int(paths[name]["levels_dec"][t - DEC[0], i]); rec[f"{name}_level_tm1"] = int(paths[name]["levels_dec"][t - 1 - DEC[0], i]) if t - 1 >= DEC[0] else None
        else:
            rec[f"{name}_alarm_at"] = bool(paths[name]["alarm_dec"][t - DEC[0], i]); rec[f"{name}_alarm_tm1"] = bool(paths[name]["alarm_dec"][t - 1 - DEC[0], i]) if t - 1 >= DEC[0] else None
    det.append(rec)
if det:
    keys = sorted({k for r_ in det for k in r_}, key=lambda s: (s not in ("item", "group", "month", "measure"), s))
    with open(os.path.join(a.out, "events_N2_detail.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(det)
summary["n_restriction_events_in_window"] = len(det)
np.savez(os.path.join(a.out, "paths.npz"), **{f"{n}_{k}": v for n, P in paths.items() for k, v in P.items() if isinstance(v, np.ndarray)},
         items=np.array(items), periods=np.array(periods), DEC=DEC)
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)

# figure: p and levels for the items with restriction events (M2), restriction months marked
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if "M2" in paths and det:
        its = sorted({r["item"] for r in det})[:8]
        fig, axes = plt.subplots(len(its), 1, figsize=(7.5, 1.6 * len(its)), sharex=True, squeeze=False)
        for ax, it in zip(axes[:, 0], its):
            i = items.index(it); ax.plot(np.arange(T), paths["M2"]["p"][:, i], color="black", lw=1.0)
            for pi in PIS:
                ax.axhline(pi, color="grey", lw=0.5, ls=":")
            for (ii, t), _ in ev2.items():
                if ii == i:
                    ax.axvline(t, color="red", lw=0.8)
            for t in np.flatnonzero(paths["M2"]["q_alarm"]):
                ax.axvspan(t - 0.5, t + 0.5, color="#dddddd", zorder=0)
            ax.set_ylabel(f"HS {it}\np(3)", fontsize=7); ax.set_ylim(0, 1.02)
        tick = [t for t, per in enumerate(periods) if per.endswith("01")][::2]
        axes[-1, 0].set_xticks(tick); axes[-1, 0].set_xticklabels([periods[t][:4] for t in tick], fontsize=7)
        fig.suptitle("M2 three-month shock probability; red = restriction implemented by a main partner; grey = q-alarm months", fontsize=8)
        fig.savefig(os.path.join(a.out, "fig_items_with_restrictions.png"), dpi=200, bbox_inches="tight")
except Exception as e:
    log("figure skipped:", e)
log("done:", json.dumps({k: summary[k] for k in ("events", "item_years", "n_restriction_events_in_window", "q_threshold")}, default=str))
