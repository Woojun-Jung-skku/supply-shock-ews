"""evaluate.py — turns filter outputs and baseline predictions of one replication into the §5.1
performance measures (see metrics.py for definitions)."""
import numpy as np
import config as C
import metrics as MT
import charts as CH
import pfilter as PF
from ml import labels


def _valid(sim, months):
    """Label of decision month s is usable when months s+1..s+3 are observed for the item."""
    mk = sim["Nmask"]
    return np.stack([mk[:, s:s + 3].all(1) for s in months])


def _fit_summary(fits):
    return dict(n_origins=len(fits), n_failed=int(sum(f["failed"] for f in fits)),
                n_refit=int(sum(f["n_refit"] for f in fits)),
                rhat_max=max(f["rhat_max"] for f in fits), n_div=int(sum(f["n_div"] for f in fits)),
                seconds=round(sum(f["seconds"] for f in fits), 1),
                failed_origins=[f["origin"] for f in fits if f["failed"]])


def model_metrics(res, sim, regime, with_extras):
    months = res["months"]
    N, mk, group = sim["N"], sim["Nmask"], sim["group"]
    t0 = sim["truth"]["t0"]
    y = labels(N)[months]
    val = _valid(sim, months)
    out = dict(fit=_fit_summary(res["fits"]), calib=res["calib"])
    out["prob"] = MT.prob_metrics(res["p"][val], y[val])
    lev = MT.levels(res["p"], C.PI_LEVELS, C.PERSIST)
    cal = res["calib"]
    qa = PF.persistent(res["q"] >= cal["q_thr"], cal["q_persist"])
    out["q"] = MT.run_length(qa, months, t0, regime)
    if with_extras:                                                   # M2: severity, AR(1) forecast, sigma-rule
        out["prob_ar1x"] = MT.prob_metrics(res["p_ar"][val], y[val])
        sev_body = res["p"] * res["E_S"] >= C.SEVERE_LOSS
        sev_adj = res["p"] * res["E_S_adj"] >= C.SEVERE_LOSS
        lev3 = lev == 3
        out["severe"] = dict(level3_months=int(lev3.sum()), severe_body=int((lev3 & sev_body).sum()),
                             severe_tail=int((lev3 & sev_adj).sum()),
                             changed_by_tail=int(((lev3 & sev_body) != (lev3 & sev_adj)).sum()),
                             sev_failed=int(sum(f["failed"] for f in (res.get("sev_fits") or []))))
        lev = np.where(lev3 & ~sev_adj, 2, lev)                         # severe needs the tail-adjusted loss
        if "S_q05" in res:                                                # v1.24 (B5): loss-fraction intervals and gamma recovery
            out["severity"] = severity_metrics(res, sim, months)
        a = np.clip(res["alpha_hat"], 1e-6, None)
        sri = (1.0 + res["mu_hat"] / a) ** (-a)
        slev = CH.sigma_rule(sri, res["sri_hist"])
        out["sigma_rule"] = {f"level{k}": MT.event_metrics(slev >= k, months, N, mk, t0, regime) for k in (1, 2, 3)}
        out["sigma_rule"]["run_length"] = MT.run_length(slev >= 1, months, t0, regime)
    head = np.maximum(lev, 2 * qa[:, group])
    out["triage"] = {f"level{k}": MT.event_metrics(lev >= k, months, N, mk, t0, regime) for k in (1, 2, 3)}
    out["headline"] = {f"level{k}": MT.event_metrics(head >= k, months, N, mk, t0, regime) for k in (1, 2)}
    out["conformal"] = MT.conformal(res["pmf"], N, mk, months, t0, C.CONF_ALPHA, C.ACI_GAMMA, C.CONF_MIN_CALIB)
    return out


def prob_method_metrics(p, sim, months, regime):
    N, mk = sim["N"], sim["Nmask"]
    y = labels(N)[months]
    val = _valid(sim, months)
    lev = MT.levels(p, C.PI_LEVELS, C.PERSIST)
    out = dict(prob=MT.prob_metrics(p[val], y[val]))
    out["triage"] = {f"level{k}": MT.event_metrics(lev >= k, months, N, mk, sim["truth"]["t0"], regime) for k in (1, 2, 3)}
    return out


def chart_metrics(sim, months, regime, seed):
    al = CH.chart_alarms(sim["N"], seed, months[0], months[-1])
    t0 = sim["truth"]["t0"]
    return {k: dict(events=MT.event_metrics(a, months, sim["N"], sim["Nmask"], t0, regime),
                    run_length=MT.run_length(a, months, t0, regime)) for k, a in al.items()}


def severity_metrics(res, sim, months, alpha=0.1):
    """v1.24 (B5): the loss-fraction interval of Section 4.3 evaluated on the realised losses of the decision months.
    Base interval = posterior predictive 5-95% quantiles of S | N >= 1 from the severity fit of the latest origin
    (res['S_q05'], res['S_q95'], (M, I)); 'raw' coverage of that interval, and 'cqr' coverage after conformalised
    quantile regression: score = max(q05 - S, S - q95) on the losses realised at earlier decision months
    (expanding calibration set, at least C.CONF_MIN_CALIB scores), interval widened by the (1 - alpha) quantile of
    the scores (Romano et al.). Also the posterior summaries of gamma and phi at the last origin."""
    S = sim["S"]
    I, T = sim["N"].shape
    lo, hi = res["S_q05"], res["S_q95"]
    obs = np.full(lo.shape, np.nan)
    for mi, t in enumerate(months):                                       # decision month t (1-based) realises month t+1 = column t
        if t < T:
            obs[mi] = S[:, t]
    ok = np.isfinite(obs) & sim["Nmask"][:, months.clip(max=T - 1)].T
    raw_cov = float(((obs >= lo) & (obs <= hi))[ok].mean()) if ok.any() else None
    width = float((hi - lo)[ok].mean()) if ok.any() else None
    # CQR calibration in time: scores of losses realised at earlier decision months
    scores = np.maximum(lo - obs, obs - hi)
    cov_c, wid_c, n_eval = 0, 0.0, 0
    for mi in range(len(months)):
        past = scores[:mi][ok[:mi]]
        if past.size < C.CONF_MIN_CALIB:
            continue
        n = past.size
        qhat = np.quantile(past, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
        now = ok[mi]
        if not now.any():
            continue
        l2, h2 = np.clip(lo[mi] - qhat, 0, 1), np.clip(hi[mi] + qhat, 0, 1)
        cov_c += int(((obs[mi] >= l2) & (obs[mi] <= h2))[now].sum()); wid_c += float((h2 - l2)[now].sum()); n_eval += int(now.sum())
    last = (res.get("sev_fits") or [None])[-1]
    return dict(n_losses=int(ok.sum()), raw_cover=raw_cov, raw_width=width,
                cqr_cover=(cov_c / n_eval if n_eval else None), cqr_width=(wid_c / n_eval if n_eval else None), cqr_n=n_eval,
                gamma_q05=(last or {}).get("gamma_q05"), gamma_q50=(last or {}).get("gamma_q50"), gamma_q95=(last or {}).get("gamma_q95"),
                phi_q50=(last or {}).get("phi_q50"), phi_q05=(last or {}).get("phi_q05"), phi_q95=(last or {}).get("phi_q95"),
                S_mean_obs=float(obs[ok].mean()) if ok.any() else None)
