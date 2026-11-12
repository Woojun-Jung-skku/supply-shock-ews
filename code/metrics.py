"""metrics.py — performance measures of §5.1.

Estimation (Study A, robustness): posterior 5/50/95% quantiles and truth are stored per parameter;
bias, RMSE and 90% coverage are then computed for alpha', C, beta, the trend paths and the
changepoint date.
Alarms (Study B, robustness), at decision months t = 36..117 (0-based month index t: data through
month t, prediction for months t+1..t+3):
  detection rate at h = 1, 2, 3   share of realised shocks (i, tau) with an alarm on item i in some
                                  decision month of [tau - h, tau - 1]
  lead time                       months from an alarm onset to the next realised shock (median, IQR)
  false alarms per item-year      alarm onsets with no shock in the next 3 months, per item-year
  AUC-PR, Brier, ECE              probability methods, event "at least one shock within 3 months"
  ARL0, ARL1                      q channel per group (and charts per item): ARL0 = stable-window
                                  unit-months / alarm months; ARL1 = mean delay from t0 to the first
                                  persistent alarm (censored at the end of the window)
  conformal                       realised coverage and mean size of the sets, before / after t0
"""
import numpy as np
from sklearn.metrics import average_precision_score


def quantiles(a, axis=0):
    q = np.percentile(a, [5, 50, 95], axis=axis)
    return dict(q05=q[0], q50=q[1], q95=q[2])


def recovery(post, truth, variant):
    """post: site -> (draws, ...) numpy; returns compact per-replication record."""
    out = {}
    for name in ("beta", "w", "kappa_w", "alpha", "logC", "tau_C", "sigma_eta", "tau_zeta", "m_alpha", "tau_alpha"):
        if name in post:
            out[name] = {k: np.round(v, 4) for k, v in quantiles(post[name]).items()}
    # trend paths: coverage and RMSE of the posterior median
    for name, tr in (("m_t", truth["m"]), ("delta_t", truth["delta"])):
        if name in post:
            q = quantiles(post[name])
            out[name] = dict(cover=float(((q["q05"] <= tr) & (tr <= q["q95"])).mean()),
                             rmse=float(np.sqrt(((q["q50"] - tr) ** 2).mean())),
                             width=float((q["q95"] - q["q05"]).mean()))
    lc = post["logC"]
    q = quantiles(lc)
    tr = truth["logC"]
    cov = (q["q05"] <= tr) & (tr <= q["q95"])
    out["logC_summary"] = dict(rmse=float(np.sqrt(((q["q50"] - tr) ** 2).mean())),
                               corr=float(np.corrcoef(q["q50"], tr)[0, 1]), cover=float(cov.mean()))
    if truth["cold_items"].any():                                  # (iv) cold-start items vs the rest
        c = truth["cold_items"]
        out["logC_cold"] = dict(rmse=float(np.sqrt(((q["q50"][c] - tr[c]) ** 2).mean())), cover=float(cov[c].mean()))
        out["logC_warm"] = dict(rmse=float(np.sqrt(((q["q50"][~c] - tr[~c]) ** 2).mean())), cover=float(cov[~c].mean()))
    return out


def recovery_q(q, truth, variant):
    """As `recovery`, from precomputed 5/50/95% quantiles: q[site] = dict(q05, q50, q95) (numpy)."""
    out = {}
    for name in ("beta", "w", "kappa_w", "alpha", "logC", "tau_C", "sigma_eta", "tau_zeta", "m_alpha", "tau_alpha"):
        if name in q:
            out[name] = {k: np.round(v, 4) for k, v in q[name].items()}
    for name, tr in (("m_t", truth["m"]), ("delta_t", truth["delta"])):
        if name in q:
            qq = q[name]
            out[name] = dict(cover=float(((qq["q05"] <= tr) & (tr <= qq["q95"])).mean()),
                             rmse=float(np.sqrt(((qq["q50"] - tr) ** 2).mean())),
                             width=float((qq["q95"] - qq["q05"]).mean()))
    qq = q["logC"]
    tr = truth["logC"]
    cov = (qq["q05"] <= tr) & (tr <= qq["q95"])
    out["logC_summary"] = dict(rmse=float(np.sqrt(((qq["q50"] - tr) ** 2).mean())),
                               corr=float(np.corrcoef(qq["q50"], tr)[0, 1]), cover=float(cov.mean()))
    if truth["cold_items"].any():                                  # (iv) cold-start items vs the rest
        c = truth["cold_items"]
        out["logC_cold"] = dict(rmse=float(np.sqrt(((qq["q50"][c] - tr[c]) ** 2).mean())), cover=float(cov[c].mean()))
        out["logC_warm"] = dict(rmse=float(np.sqrt(((qq["q50"][~c] - tr[~c]) ** 2).mean())), cover=float(cov[~c].mean()))
    return out


def changepoint_q(cp, t0):
    """As `changepoint`, from device summaries cp = dict(q05, q50, q95, p_t0) per group."""
    return dict(med=cp["q50"].tolist(), lo=cp["q05"].tolist(), hi=cp["q95"].tolist(), t0=int(t0),
                abs_err=float(np.abs(cp["q50"] - t0).mean()),
                cover=float(((cp["q05"] <= t0) & (t0 <= cp["q95"])).mean()),
                max_prob=float(cp["p_t0"].max()))


def changepoint(post_m, post_d, t0):
    """Change date per draw and group = column of the largest level innovation
    |m_{t+1} - m_t - delta_t|; truth is column t0 (the first changed month)."""
    innov = np.abs(np.diff(post_m, axis=-1) - post_d[..., :-1])     # (S, G, T-1)
    tau = innov.argmax(-1) + 1
    q = quantiles(tau)
    return dict(med=q["q50"].tolist(), lo=q["q05"].tolist(), hi=q["q95"].tolist(), t0=int(t0),
                abs_err=float(np.abs(q["q50"] - t0).mean()),
                cover=float(((q["q05"] <= t0) & (t0 <= q["q95"])).mean()),
                max_prob=float((tau == t0).mean(0).max()))


# ------------------------------------------------------------------ alarm metrics
def _shock_months(N, Nmask):
    return (N >= 1) & Nmask                                            # (I, T): column j = month j+1


def event_metrics(A, months, N, Nmask, t0, regime):
    """A (M, I) bool alarms at decision months `months` (M,)."""
    I, T = N.shape
    s0, s1 = months[0], months[-1]
    shock = _shock_months(N, Nmask)
    res = {}
    for h in (1, 2, 3):
        det = tot = 0
        for tau in range(s0 + h, s1 + 2):                             # shock month tau (1-based)
            if tau > T:
                break
            hit = shock[:, tau - 1]
            if not hit.any():
                continue
            win = A[(tau - h - s0):(tau - s0)]                          # decision months tau-h..tau-1
            det += (hit & win.any(0)).sum()
            tot += hit.sum()
        res[f"dr_h{h}"] = round(det / tot, 4) if tot else None
    prev = np.vstack([np.zeros((1, I), bool), A[:-1]])
    onset = A & ~prev
    leads, fa, fa_stable = [], 0, 0
    for m_idx, i in zip(*np.nonzero(onset)):
        t = months[m_idx]
        nxt = np.flatnonzero(shock[i, t:])                            # months t+1.. -> columns t..
        if nxt.size:
            leads.append(int(nxt[0] + 1))
        if not shock[i, t:t + 3].any():
            fa += 1
            if regime == "stable" or t < t0:
                fa_stable += 1
    years = I * len(months) / 12.0
    st_months = len(months) if regime == "stable" else max(t0 - s0, 1)
    res.update(lead_med=float(np.median(leads)) if leads else None,
               lead_q25=float(np.percentile(leads, 25)) if leads else None,
               lead_q75=float(np.percentile(leads, 75)) if leads else None, n_onsets=int(onset.sum()),
               fa_per_itemyear=round(fa / years, 4),
               fa_per_itemyear_stable=round(fa_stable / (I * st_months / 12.0), 4),
               alarm_share=round(float(A.mean()), 4))
    return res


def run_length(A, months, t0, regime):
    """A (M, U) persistent alarms per unit (group or item): ARL0 in the stable window, ARL1 after t0."""
    stable = A if regime == "stable" else A[months < t0]
    arl0 = stable.size / max(stable.sum(), 1)
    out = dict(arl0=round(float(arl0), 2), stable_alarm_units=round(float(stable.any(0).mean()), 3))
    if regime != "stable":
        post = A[months >= t0]
        hit = post.any(0)
        delay = np.where(hit, post.argmax(0), post.shape[0])
        out.update(arl1=round(float(delay.mean()), 2), arl1_censored=int((~hit).sum()))
    return out


def prob_metrics(p, y):
    p = np.clip(p.ravel(), 0, 1)
    y = y.ravel().astype(bool)
    res = dict(brier=round(float(((p - y) ** 2).mean()), 5), base_rate=round(float(y.mean()), 4),
               mean_p=round(float(p.mean()), 4))
    res["aucpr"] = round(float(average_precision_score(y, p)), 4) if y.any() and not y.all() else None
    bins = np.minimum((p * 10).astype(int), 9)
    ece = 0.0
    rel = []
    for b in range(10):
        k = bins == b
        if k.any():
            ece += k.mean() * abs(p[k].mean() - y[k].mean())
            rel.append([b, int(k.sum()), round(float(p[k].mean()), 4), round(float(y[k].mean()), 4)])
    res.update(ece=round(float(ece), 5), reliability=rel)
    return res


def levels(p, pis, persist=2):
    """Triage level (23) with r-of-r persistence: max k with p >= pi_k in `persist` consecutive months."""
    lev = np.zeros(p.shape, int)
    for k, pi in enumerate(pis, 1):
        hit = p >= pi
        pers = hit.copy()
        for r in range(1, persist):
            sh = np.zeros_like(hit)
            sh[r:] = hit[:-r]
            pers &= sh
        lev = np.maximum(lev, k * pers)
    return lev


def conformal(pmf, N, Nmask, months, t0, alpha, gamma, min_calib):
    """APS sets (Romano et al. 2020) for N_{i,t+1} with online ACI per item (Gibbs & Candes 2021).
    pmf (M, I, nmax+1) predictive pmf at decision months; realised count is column t (month t+1)."""
    M, I, K1 = pmf.shape
    order = np.argsort(-pmf, -1, kind="stable")
    srt = np.take_along_axis(pmf, order, -1)
    csum = np.cumsum(srt, -1)
    score_all = np.empty_like(pmf)                                    # s(n) = sum of q(m) with q(m) >= q(n)
    np.put_along_axis(score_all, order, csum, -1)
    a_it = np.full(I, alpha)
    calib = []
    cover, size, when = [], [], []
    for m in range(M):
        t = months[m]
        if t >= N.shape[1]:
            break
        n_obs = np.minimum(N[:, t], K1 - 1)
        obs = Nmask[:, t]
        s_obs = score_all[m, np.arange(I), n_obs]
        if len(calib) >= min_calib:
            cal = np.asarray(calib)
            lev = np.clip(1.0 - a_it, 0.0, 1.0)
            qhat = np.quantile(cal, np.minimum(np.ceil((cal.size + 1) * lev) / cal.size, 1.0), method="higher")
            inset = score_all[m] <= qhat[:, None]                     # (I, K1)
            covered = inset[np.arange(I), n_obs]
            err = (~covered).astype(float)
            a_it = np.where(obs, a_it + gamma * (alpha - err), a_it)
            cover.extend(covered[obs].tolist()); size.extend(inset.sum(1)[obs].tolist()); when.extend([t] * obs.sum())
        calib.extend(s_obs[obs].tolist())
    if not cover:
        return dict(conf_cover=None)
    cover, size, when = np.array(cover), np.array(size), np.array(when)
    pre, post = when < t0, when >= t0
    f = lambda v, k: round(float(v[k].mean()), 4) if k.any() else None
    return dict(conf_cover=f(cover, np.ones_like(pre)), conf_size=f(size, np.ones_like(pre)),
                conf_cover_pre=f(cover, pre), conf_cover_post=f(cover, post),
                conf_size_pre=f(size, pre), conf_size_post=f(size, post), conf_n=int(cover.size))
