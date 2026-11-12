"""B1 — CUSUM and EWMA designed under the negative binomial, on the Study B replications (manuscript Section 5.3, Table S14).

Same conventions as the Poisson-calibrated charts of charts.py (Section 5.1): phase I = months 13-36 per item
(in-control mean mu0_i), charts start at month 37, decision months 36..117, alarm = statistic above its limit,
no persistence, one level; the limit is the (1 - 1/24) quantile of the statistic's in-control distribution
(nominal ARL0 = 24 months in the paper's alarm-month units), obtained by simulation.
What changes: (i) the in-control distribution is negative binomial with the dispersion estimated in phase I
by the pooled moment estimator within the item's group, alpha_g = sum_i mu0_i^2 / sum_i (s_i^2 - mu0_i)
(Poisson when the pooled excess variance is not positive); (ii) the CUSUM reference value is the
negative-binomial likelihood-ratio value for a rho-fold shift at that dispersion,
k = alpha log((alpha + rho mu0)/(alpha + mu0)) / log(rho (alpha + mu0)/(alpha + rho mu0)),
which reduces to the Lucas (1985) value as alpha -> inf; (iii) the EWMA statistic is unchanged and only
its limit is set under the negative binomial.
Evaluated as the EWMA / CUSUM rows of Table 6 (metrics.event_metrics, metrics.run_length) on the same
simulated data (dgp.seed_of('B', cell, rep)); the Poisson charts are recomputed alongside as a check against
analysis/B_replications.csv.
usage: python followup/b1_nbcharts.py ANALYSIS_DIR OUT_DIR   (POOL=panel for the panel-pooled dispersion)
"""
import os, sys, time, types
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE); sys.path.insert(0, os.path.join(CODE, 'tools'))
import config as C
C.apply_profile("t4_min")
import dgp
import charts as CH
import metrics as MT
import arl as ARL

I_, T_ = C.REF["I"], C.REF["T"]
def cells():
    out = []
    for a in C.B_ALPHAS:
        for mu in C.B_MUS:
            for reg in C.REGIMES:
                out.append(dict(id=f"B-a{a}-mu{mu}-{reg}", alpha=a, mu=mu, regime=reg, I=I_, T=T_))
    return out
def simulate(cell, r, study="B", **kw):
    s = dgp.simulate(cell["I"], cell["T"], float(cell["mu"]), float(cell["alpha"]), cell["regime"],
                     seed=dgp.seed_of(study, cell["id"], r), **kw)
    return s

ALPHA_MAX = 1e6
POOL = os.environ.get('POOL', 'group')   # 'group': within the item's group (main); 'panel': all items


def nb_params(N, group):
    """Phase-I in-control mean per item and pooled moment dispersion per group."""
    a, b = C.CHART_PHASE1
    ph = N[:, a:b].astype(float)
    mu0 = np.maximum(ph.mean(1), 0.25 / 24)
    s2 = ph.var(1, ddof=1)
    alpha = np.full(N.shape[0], np.inf)
    if POOL == 'panel':
        group = np.zeros_like(group)
    for g in np.unique(group):
        idx = group == g
        den = float((s2[idx] - mu0[idx]).sum()); num = float((mu0[idx] ** 2).sum())
        alpha[idx] = min(num / den, ALPHA_MAX) if den > 0 else np.inf
    return mu0, alpha


def nb_k(mu0, alpha, rho):
    mu1 = rho * mu0
    fin = np.isfinite(alpha)
    k = (mu1 - mu0) / np.log(mu1 / mu0)                                # Lucas (Poisson) value
    a = alpha[fin]; m0 = mu0[fin]; m1 = mu1[fin]
    k[fin] = a * np.log((a + m1) / (a + m0)) / np.log(m1 * (a + m0) / (m0 * (a + m1)))
    return k


def nb_sim(rng, mu0, alpha, M):
    I = mu0.size
    lam = np.empty((I, M))
    fin = np.isfinite(alpha)
    lam[fin] = rng.gamma(alpha[fin][:, None], (mu0[fin] / alpha[fin])[:, None], size=(fin.sum(), M))
    lam[~fin] = mu0[~fin][:, None]
    return rng.poisson(lam).astype(float)


def nb_chart_alarms(N, group, seed, t_first, t_last, rho=C.RHO_SHIFT, lam=C.EWMA_LAMBDA):
    rng = np.random.default_rng(seed)
    a, b = C.CHART_PHASE1
    mu0, alpha = nb_params(N, group)
    sim = nb_sim(rng, mu0, alpha, CH._CAL_MONTHS)
    k = nb_k(mu0, alpha, rho)
    h_e = CH._limit(CH._ewma(sim, lam, mu0)[:, 100:], C.ARL0_TARGET)
    h_c = CH._limit(CH._cusum_k(sim, k)[:, 100:], C.ARL0_TARGET)
    run = N[:, b:].astype(float)
    e = CH._ewma(run, lam, mu0); c = CH._cusum_k(run, k)
    months = np.arange(t_first, t_last + 1)
    idx = months - 1 - b; ok = idx >= 0
    ea = np.zeros((len(months), N.shape[0]), bool); ca = np.zeros_like(ea)
    ea[ok] = (e[:, idx[ok]] > h_e[:, None]).T
    ca[ok] = (c[:, idx[ok]] > h_c[:, None]).T
    return dict(nbewma=ea, nbcusum=ca, alpha_hat=alpha, mu0=mu0, k=k, h_c=h_c, h_e=h_e, months=months)


def evaluate_rep(cell, r):
    s = simulate(cell, r)
    N, mk, t0, group = s["N"], s["Nmask"], s["truth"]["t0"], s["group"]
    months = np.arange(C.ORIGINS[0], C.DECISION_END + 1)
    nb = nb_chart_alarms(N, group, dgp.seed_of("nbchart", cell["id"], r), months[0], months[-1])
    po = CH.chart_alarms(N, dgp.seed_of("chart", cell["id"], r), months[0], months[-1])   # check against B_replications
    row = dict(cell=cell["id"], rep=r, alpha=float(cell["alpha"]), mu=float(cell["mu"]), regime=cell["regime"], t0=int(t0),
               alpha_hat_geo=float(np.exp(np.log(np.minimum(nb["alpha_hat"], ALPHA_MAX)).mean())),
               alpha_hat_poisson_share=float(np.isinf(nb["alpha_hat"]).mean()),
               k_mean=float(nb["k"].mean()), h_c_mean=float(nb["h_c"].mean()), h_e_mean=float(nb["h_e"].mean()))
    for key, A in (("nbcusum", nb["nbcusum"]), ("nbewma", nb["nbewma"]), ("cusum", po["cusum"]), ("ewma", po["ewma"])):
        ev = MT.event_metrics(A, months, N, mk, t0, cell["regime"]); rl = MT.run_length(A, months, t0, cell["regime"])
        row.update({f"{key}_dr1": ev["dr_h1"], f"{key}_dr3": ev["dr_h3"], f"{key}_fa_stable": ev["fa_per_itemyear_stable"],
                    f"{key}_fa": ev["fa_per_itemyear"], f"{key}_lead": ev["lead_med"], f"{key}_lead_q25": ev["lead_q25"],
                    f"{key}_lead_q75": ev["lead_q75"], f"{key}_alarm_share": ev["alarm_share"], f"{key}_n_onsets": ev["n_onsets"],
                    f"{key}_arl0": rl["arl0"], f"{key}_stable_alarm_units": rl["stable_alarm_units"],
                    f"{key}_arl1": rl.get("arl1"), f"{key}_arl1_cens": rl.get("arl1_censored")})
    return row


def main():
    an, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    B = pd.read_csv(os.path.join(an, "B_replications.csv"))
    cmap = {c["id"]: c for c in cells()}
    rows = []; t0 = time.time()
    for k, (cid, r) in enumerate(zip(B.cell, B.rep)):
        rows.append(evaluate_rep(cmap[cid], int(r)))
        if (k + 1) % 10 == 0: print(f"{k + 1}/{len(B)} replications, {time.time() - t0:.0f}s", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out, f"B1_nbcharts_replications_{POOL}.csv"), index=False)
    # check: recomputed Poisson charts against the archived replication table
    chk = B.merge(d[["cell", "rep", "cusum_arl0", "ewma_arl0", "cusum_dr1", "ewma_fa_stable"]], on=["cell", "rep"], suffixes=("_tab", "_new"))
    lines = ["check: Poisson charts recomputed vs analysis/B_replications.csv (max abs diff): " +
             ", ".join(f"{c} {np.nanmax(np.abs(chk[c + '_tab'] - chk[c + '_new'])):.4f}" for c in ("cusum_arl0", "ewma_arl0", "cusum_dr1", "ewma_fa_stable"))]
    def ms(v):
        v = pd.to_numeric(v, errors="coerce").dropna().values
        return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)), int(v.size)) if v.size else (np.nan, np.nan, 0)
    for key, name in (("nbcusum", "NB-designed CUSUM"), ("nbewma", "NB-designed EWMA"), ("cusum", "Poisson CUSUM (Table 6)"), ("ewma", "Poisson EWMA (Table 6)")):
        pr = ARL.pooled(d, f"{key}_arl0", f"{key}_stable_alarm_units", ARL.UNITS["item"])
        r1 = ms(d[f"{key}_dr1"]); r3 = ms(d[f"{key}_dr3"]); fa = ms(d[f"{key}_fa_stable"]); ld = ms(d[f"{key}_lead"]); a1 = ms(d[f"{key}_arl1"])
        lines += [f"{name}: det<=1 {r1[0]:.3f} ({r1[1]:.3f}), det<=3 {r3[0]:.3f} ({r3[1]:.3f}), FA/item-year stable {fa[0]:.3f} ({fa[1]:.3f}), "
                  f"lead median {ld[0]:.1f} ({ld[1]:.1f}) IQR {d[key + '_lead_q25'].mean():.1f}-{d[key + '_lead_q75'].mean():.1f}, "
                  f"ARL0 pooled {pr['est']:.1f} (SE {pr['se']:.1f}; 90% {pr['lo']:.1f}-{pr['hi']:.1f}; alarm months {pr['alarms']} / {pr['unit_months']}), "
                  f"ARL1 {a1[0]:.1f} ({a1[1]:.1f}) n={a1[2]}"]
        for reg in ("stable", "gradual", "abrupt"):
            dd = d[d.regime == reg]; p2 = ARL.pooled(dd, f"{key}_arl0", f"{key}_stable_alarm_units", ARL.UNITS["item"])
            lines.append(f"   ARL0 {reg}: {p2['est']:.1f} (90% {p2['lo']:.1f}-{p2['hi']:.1f}; {p2['alarms']} / {p2['unit_months']}); ARL1 {ms(dd[key + '_arl1'])[0]:.1f}")
    lines.append(f"dispersion estimate: geometric mean of alpha_hat over items {np.exp(np.log(d.alpha_hat_geo).mean()):.2f}; "
                 f"share of items treated as Poisson {d.alpha_hat_poisson_share.mean():.3f}; by true alpha: " +
                 ", ".join(f"alpha'={a:g}: {np.exp(np.log(d[d.alpha == a].alpha_hat_geo).mean()):.2f} (Poisson share {d[d.alpha == a].alpha_hat_poisson_share.mean():.2f})" for a in sorted(d.alpha.unique())))
    by = d.groupby(["alpha", "mu", "regime"]).agg(nbc_dr1=("nbcusum_dr1", "mean"), nbc_fa=("nbcusum_fa_stable", "mean"), nbc_arl1=("nbcusum_arl1", "mean"),
                                                  nbe_dr1=("nbewma_dr1", "mean"), nbe_fa=("nbewma_fa_stable", "mean"), nbe_arl1=("nbewma_arl1", "mean"), n=("rep", "size"))
    lines.append("\nper cell:\n" + by.round(3).to_string())
    txt = "\n".join(lines); print(txt)
    open(os.path.join(out, f"B1_nbcharts_summary_{POOL}.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
