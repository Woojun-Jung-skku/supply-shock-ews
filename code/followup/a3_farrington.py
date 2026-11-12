"""A3 — Farrington-type baseline on the Study B replications (manuscript Section 5.3, Table 6 and Table S9).

A rolling, per-item outbreak-detection threshold in the spirit of Farrington et al. (1996) and
Noufaily et al. (2013), without seasonality (the simulated counts have none):
  * at decision month t the baseline is the item's counts of the previous W = 36 months (months
    t-36 .. t-1; fewer at the first origins), fitted by a quasi-Poisson GLM with an intercept and a
    linear time trend, the trend kept only when its Wald test is significant at 5% (Farrington 1996);
  * past outliers are down-weighted once by Anscombe residuals, weight (gamma / |r|)^2 for |r| > 2.58;
  * the dispersion phi = max(1, Pearson chi^2 / df) turns the fitted mean into a negative-binomial
    predictive, N ~ NB(mu_hat, alpha = mu_hat / (phi - 1)) (Poisson when phi = 1), i.e. the
    dispersion-corrected limit of Section 4.4;
  * the threshold is the (1 - 1/24) quantile of that predictive, the same nominal in-control run
    length of 24 months at which the EWMA and CUSUM charts are Poisson-calibrated; the item alarms at
    decision month t when its count of month t exceeds the threshold (no persistence, one level).
Evaluated with the §5.1 event measures and run lengths exactly as the EWMA / CUSUM rows of Table 6
(metrics.event_metrics, metrics.run_length) on the same simulated data (dgp.seed_of('B', cell, rep)).
usage: python followup/a3_farrington.py ANALYSIS_DIR OUT_DIR   (ANALYSIS_DIR holds B_replications.csv)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import config as C
C.apply_profile("t4_min")
import study_b
import metrics as MT
import arl as ARL

W_HIST = 36
LEVEL = 1.0 - 1.0 / C.ARL0_TARGET
GAMMA = 2.58
MU_FLOOR = 0.25 / 24                      # as the charts (charts.chart_alarms)
MIN_HIST = 12


def _glm(y, s, w, trend):
    """Vectorised quasi-Poisson IRLS over items: y, w (I, n), s (n,) time covariate (0 at the current month).
    Returns a, b (I,), phi (I,), se_b (I,)."""
    I, n = y.shape
    X = np.column_stack([np.ones(n), s]) if trend else np.ones((n, 1))
    b = np.zeros((I, X.shape[1]))
    b[:, 0] = np.log(np.maximum((w * y).sum(1) / np.maximum(w.sum(1), 1e-9), MU_FLOOR))
    for _ in range(25):
        eta = b @ X.T                                             # (I, n)
        mu = np.exp(np.clip(eta, -30, 30))
        z = eta + (y - mu) / mu
        Wt = w * mu                                               # IRLS weights
        XtW = X.T[None] * Wt[:, None, :]                          # (I, p, n)
        A = XtW @ X                                               # (I, p, p)
        rhs = np.einsum("ipn,in->ip", XtW, z)
        A += 1e-9 * np.eye(X.shape[1])
        b_new = np.linalg.solve(A, rhs[..., None])[..., 0]
        if np.abs(b_new - b).max() < 1e-8:
            b = b_new
            break
        b = b_new
    eta = b @ X.T
    mu = np.exp(np.clip(eta, -30, 30))
    df = np.maximum(w.sum(1) - X.shape[1], 1.0)
    phi = np.maximum((w * (y - mu) ** 2 / mu).sum(1) / df, 1.0)
    cov = np.linalg.inv(A)                                        # (I, p, p)
    se_b = np.sqrt(np.maximum(phi, 1.0) * cov[:, -1, -1]) if trend else np.zeros(I)
    return b, mu, phi, se_b


def _anscombe_weights(y, mu, phi):
    r = 1.5 * (y ** (2 / 3) - mu ** (2 / 3)) / mu ** (1 / 6) / np.sqrt(phi)[:, None]
    w = np.ones_like(y)
    big = np.abs(r) > GAMMA
    w[big] = (GAMMA / np.abs(r[big])) ** 2
    return w


def farrington_alarms(N, t_first, t_last, W=W_HIST, level=LEVEL):
    """Alarm indicators (len(months), I) at decision months t_first..t_last (0-based month index as in
    charts.chart_alarms: decision month t sees the counts up to month t = column t - 1)."""
    I, T = N.shape
    months = np.arange(t_first, t_last + 1)
    A = np.zeros((len(months), I), bool)
    thr = np.zeros((len(months), I))
    mu_out = np.zeros((len(months), I))
    for mi, t in enumerate(months):
        cur = t - 1
        h0 = max(cur - W, 0)
        y = N[:, h0:cur].astype(float)
        n = y.shape[1]
        s = (np.arange(h0, cur) - cur).astype(float)             # -n .. -1
        w = np.ones_like(y)
        if n < MIN_HIST:
            mu_hat = np.maximum(y.mean(1), MU_FLOOR)
            phi = np.ones(I)
        else:
            # pass 1 with trend, re-weight outliers, pass 2; keep the trend where significant
            b, mu, phi, se_b = _glm(y, s, w, True)
            w = _anscombe_weights(y, mu, phi)
            b, mu, phi, se_b = _glm(y, s, w, True)
            keep = np.abs(b[:, 1]) > 1.96 * np.maximum(se_b, 1e-12)
            b0, mu0, phi0, _ = _glm(y, s, w, False)
            mu_hat = np.where(keep, np.exp(np.clip(b[:, 0], -30, 30)), np.exp(np.clip(b0[:, 0], -30, 30)))
            phi = np.where(keep, phi, phi0)
            mu_hat = np.maximum(mu_hat, MU_FLOOR)
        alpha = np.where(phi > 1.0, mu_hat / np.maximum(phi - 1.0, 1e-12), np.inf)
        q = np.where(np.isfinite(alpha), nbinom.ppf(level, np.where(np.isfinite(alpha), alpha, 1.0),
                                                    np.where(np.isfinite(alpha), alpha / (alpha + mu_hat), 0.5)),
                     poisson.ppf(level, mu_hat))
        thr[mi] = q
        mu_out[mi] = mu_hat
        A[mi] = N[:, cur] > q
    return dict(alarms=A, threshold=thr, mu_hat=mu_out, months=months)


def evaluate_rep(cell, r):
    s = study_b.simulate(cell, r)
    N, mk, t0 = s["N"], s["Nmask"], s["truth"]["t0"]
    fa = farrington_alarms(N, C.ORIGINS[0], C.DECISION_END)
    A, months = fa["alarms"], fa["months"]
    ev = MT.event_metrics(A, months, N, mk, t0, cell["regime"])
    rl = MT.run_length(A, months, t0, cell["regime"])
    return dict(cell=cell["id"], rep=r, alpha=float(cell["alpha"]), mu=float(cell["mu"]), regime=cell["regime"], t0=int(t0),
                farr_dr1=ev["dr_h1"], farr_dr3=ev["dr_h3"], farr_fa_stable=ev["fa_per_itemyear_stable"],
                farr_fa=ev["fa_per_itemyear"], farr_lead=ev["lead_med"], farr_lead_q25=ev["lead_q25"], farr_lead_q75=ev["lead_q75"], farr_alarm_share=ev["alarm_share"],
                farr_n_onsets=ev["n_onsets"], farr_arl0=rl["arl0"], farr_stable_alarm_units=rl["stable_alarm_units"],
                farr_arl1=rl.get("arl1"), farr_arl1_cens=rl.get("arl1_censored"),
                thr_mean=float(fa["threshold"].mean()), mu_hat_mean=float(fa["mu_hat"].mean()))


def main():
    an, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    B = pd.read_csv(os.path.join(an, "B_replications.csv"))
    cells = {c["id"]: c for c in study_b.cells()}
    rows = []
    t0 = time.time()
    for k, (cid, r) in enumerate(zip(B.cell, B.rep)):
        rows.append(evaluate_rep(cells[cid], int(r)))
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{len(B)} replications, {time.time() - t0:.0f}s", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out, "A3_farrington_replications.csv"), index=False)
    # Table 6 row, aggregated as tools/table5b.py does for EWMA / CUSUM
    def ms(v):
        v = pd.to_numeric(v, errors="coerce").dropna().values
        return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)), int(v.size)) if v.size else (np.nan, np.nan, 0)
    row = {k: ms(d["farr_" + k]) for k in ("dr1", "dr3", "fa_stable", "lead", "arl1")}
    pr = ARL.pooled(d, "farr_arl0", "farr_stable_alarm_units", ARL.UNITS["item"])
    row["arl0"] = (pr["est"], pr["se"], pr["n"], pr["lo"], pr["hi"], pr["alarms"], pr["unit_months"], pr["no_alarm_share"])
    lines = ["Farrington-type baseline (A3), Study B, pooled over the twelve cells (mean over replications, MCSE):",
             f"  detected <= 1 month  {row['dr1'][0]:.3f} ({row['dr1'][1]:.3f})",
             f"  detected <= 3 months {row['dr3'][0]:.3f} ({row['dr3'][1]:.3f})",
             f"  false alarms / item-year (stable windows) {row['fa_stable'][0]:.3f} ({row['fa_stable'][1]:.3f})",
             f"  lead time median {row['lead'][0]:.1f} ({row['lead'][1]:.1f}); IQR (mean of replication quartiles) {d.farr_lead_q25.mean():.1f}-{d.farr_lead_q75.mean():.1f}",
             f"  ARL0 pooled {pr['est']:.1f} (bootstrap SE {pr['se']:.1f}; 90% {pr['lo']:.1f}-{pr['hi']:.1f}); alarm months {pr['alarms']} / unit-months {pr['unit_months']}",
             f"  ARL1 {row['arl1'][0]:.1f} ({row['arl1'][1]:.1f}), censored {int(d['farr_arl1_cens'].dropna().sum())} of {int(d['farr_arl1_cens'].notna().sum()) * 30} item windows",
             f"  n = {len(d)} replications; mean threshold {d.thr_mean.mean():.2f}, mean fitted mu {d.mu_hat_mean.mean():.3f}"]
    by = d.groupby(["alpha", "mu", "regime"]).agg(dr1=("farr_dr1", "mean"), fa=("farr_fa_stable", "mean"), arl1=("farr_arl1", "mean"), n=("rep", "size"))
    lines.append("\nper cell:\n" + by.round(3).to_string())
    for reg in ("stable", "gradual", "abrupt"):
        dd = d[d.regime == reg]
        p2 = ARL.pooled(dd, "farr_arl0", "farr_stable_alarm_units", ARL.UNITS["item"])
        lines.append(f"ARL0 {reg}: {p2['est']:.1f} (90% {p2['lo']:.1f}-{p2['hi']:.1f})")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(out, "A3_farrington_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
