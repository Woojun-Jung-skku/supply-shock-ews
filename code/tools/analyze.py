"""v1.4: summarise the result JSONL files of Studies A, B and the robustness study into the tables
behind §5.2-5.5 (CSV + a markdown digest). Several result folders can be merged (e.g. the container
and the Mac runs); a replication present in two folders is counted once.
usage: python tools/analyze.py OUT_DIR RESULTS_DIR [RESULTS_DIR ...]"""
import sys, os, json, glob, math, re, collections
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

BETA, W_TRUE, KAPPA = np.array(C.BETA), np.array(C.W_TRUE), C.KAPPA_TRUE


def load(dirs, study):
    recs = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, study, "*.jsonl"))):
            variant = os.path.basename(p)[:-6].split("__")[1]
            for line in open(p):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                recs[(r["cell"], variant, r["rep"])] = r
    return recs


def cell_factors_a(cid):
    m = re.match(r"A-a([\d.inf]+)-mu([\d.]+)-(\w+)-I(\d+)-T(\d+)", cid)
    return dict(alpha=m.group(1), mu=m.group(2), regime=m.group(3), I=int(m.group(4)), T=int(m.group(5)))


def cell_factors_b(cid):
    m = re.match(r"B-a([\d.inf]+)-mu([\d.]+)-(\w+)", cid)
    return dict(alpha=m.group(1), mu=m.group(2), regime=m.group(3))


def q3(d):
    return np.asarray(d["q05"], float), np.asarray(d["q50"], float), np.asarray(d["q95"], float)


def rec_a(key, r):
    cell, variant, rep = key
    f = cell_factors_a(cell)
    rv = r["recovery"]
    out = dict(cell=cell, variant=variant, rep=rep, **f, failed=r["failed"], n_refit=r["n_refit"],
               seconds=r["seconds"], rhat_max=r["rhat_max"], ess_bulk_min=r["ess_bulk_min"], n_div=r["n_div"],
               design=r.get("design"))
    a_true = r["truth"]["alpha"]
    if "alpha" in rv:
        lo, md, hi = q3(rv["alpha"])
        out["alpha_q50_med"] = float(np.median(md))
        out["alpha_width_log"] = float(np.mean(np.log(hi) - np.log(lo)))
        if a_true is not None and np.isfinite(a_true):
            out["alpha_cover"] = float(np.mean((lo <= a_true) & (a_true <= hi)))
            out["alpha_logerr"] = float(np.mean(np.log(md) - math.log(a_true)))
            out["alpha_abs_logerr"] = float(np.mean(np.abs(np.log(md) - math.log(a_true))))
            out["alpha_relwidth"] = float(np.mean((hi - lo) / a_true))
        else:                                                   # Poisson truth: how far into the tail the posterior goes
            out["alpha_q05_med"] = float(np.median(lo))
    for name, tr in (("kappa_w", np.array([KAPPA])), ("beta", BETA), ("w", W_TRUE)):
        if name in rv:
            lo, md, hi = (np.atleast_1d(x) for x in q3(rv[name]))
            out[f"{name}_cover"] = float(np.mean((lo <= tr) & (tr <= hi)))
            out[f"{name}_err"] = float(np.mean(md - tr))
            out[f"{name}_abserr"] = float(np.mean(np.abs(md - tr)))
    for name in ("sigma_eta", "tau_zeta", "tau_C", "tau_alpha"):
        if name in rv:
            out[f"{name}_q50"] = float(np.asarray(rv[name]["q50"]).mean())
    ls = rv["logC_summary"]
    out.update(logC_rmse=ls["rmse"], logC_corr=ls["corr"], logC_cover=ls["cover"])
    for name in ("m_t", "delta_t"):
        if name in rv:
            out.update({f"{name}_cover": rv[name]["cover"], f"{name}_rmse": rv[name]["rmse"], f"{name}_width": rv[name]["width"]})
    if "changepoint" in r:
        cp = r["changepoint"]
        out.update(cp_abs_err=cp["abs_err"], cp_cover=cp["cover"], cp_max_prob=cp["max_prob"])
    return out


_HAWKES_MU = {}


def hawkes_reference(cell, rep):
    """Eq. (15) evaluated at the data's own mean monthly rate per group (the base rate 0.2 of the
    design is the median item's rate; the fitted group alpha' is dominated by the higher-rate items).
    Re-simulates the replication from its seed (the DGP is deterministic)."""
    key = (cell, rep)
    if key not in _HAWKES_MU:
        import study_b, study_robust, dgp
        c = next(c for c in study_robust.cells() if c["id"] == cell)
        s = study_b.simulate(c, rep, "R", **c["kw"])
        n, kd = c["kw"]["hawkes"]
        # true mean rate per item-month = mu_it of the DGP; recover it from the seed-free parts:
        # the mean count per group is an unbiased estimate of the group's mean rate
        mu_g = np.array([s["N"][s["group"] == g].mean() for g in range(C.G)])
        _HAWKES_MU[key] = np.array([dgp.alpha_match(m, n, kd) for m in mu_g]) if n > 0 else np.full(C.G, np.inf)
    return _HAWKES_MU[key]


def rec_r(key, r):
    cell, arm, rep = key
    rf = r["recovery_fit"]
    out = dict(cell=cell, arm=arm, rep=rep, failed=rf["failed"], n_refit=rf["n_refit"], seconds=rf["seconds"],
               alpha_true=r["truth"]["alpha"])
    rv = r["recovery"]
    lo, md, hi = q3(rv["alpha"])
    a = r["truth"]["alpha"]
    out["alpha_q50_med"] = float(np.median(md))
    if a is not None and np.isfinite(a):
        out["alpha_cover"] = float(np.mean((lo <= a) & (a <= hi)))
        out["alpha_logerr"] = float(np.mean(np.log(md) - math.log(a)))
    if cell.startswith("R-hawkes") and a is not None and np.isfinite(a):
        ag = hawkes_reference(cell, rep)                       # eq. (15) at each group's mean rate
        out["alpha_ref_g"] = float(np.mean(ag))
        out["alpha_cover_g"] = float(np.mean((lo <= ag) & (ag <= hi)))
        out["alpha_logerr_g"] = float(np.mean(np.log(md) - np.log(ag)))
    lo, md, hi = (np.atleast_1d(x) for x in q3(rv["kappa_w"]))
    out["kappa_cover"] = float(np.mean((lo <= KAPPA) & (KAPPA <= hi)))
    out.update(logC_rmse=rv["logC_summary"]["rmse"], logC_cover=rv["logC_summary"]["cover"],
               m_t_cover=rv["m_t"]["cover"], m_t_rmse=rv["m_t"]["rmse"])
    if "logC_cold" in rv:
        out.update(logC_cold_rmse=rv["logC_cold"]["rmse"], logC_cold_cover=rv["logC_cold"]["cover"],
                   logC_warm_rmse=rv["logC_warm"]["rmse"], logC_warm_cover=rv["logC_warm"]["cover"])
    m2 = r.get("M2")
    if m2:
        out.update(model_row("M2", m2, prefix=""))
        if "charts" in r:
            for ch in ("ewma", "cusum"):
                out[f"{ch}_arl1"] = r["charts"][ch]["run_length"].get("arl1")
                out[f"{ch}_fa_stable"] = r["charts"][ch]["events"]["fa_per_itemyear_stable"]
    if r.get("M2_cold_items"):
        out["cold_items_aucpr"] = r["M2_cold_items"].get("aucpr")
        out["cold_items_brier"] = r["M2_cold_items"].get("brier")
    return out


def model_row(name, m, prefix=None):
    """Flatten the metrics of one model-based method of a Study B replication."""
    p = f"{name}_" if prefix is None else prefix
    o = {}
    pr = m.get("prob") or {}
    o.update({p + "brier": pr.get("brier"), p + "aucpr": pr.get("aucpr"), p + "ece": pr.get("ece"), p + "base_rate": pr.get("base_rate")})
    q = m.get("q") or {}
    o.update({p + "q_arl0": q.get("arl0"), p + "q_arl1": q.get("arl1"), p + "q_arl1_cens": q.get("arl1_censored"),
              p + "q_stable_alarm_units": q.get("stable_alarm_units")})
    cal = m.get("calib") or {}
    o.update({p + "q_thr": cal.get("q_thr"), p + "q_persist": cal.get("q_persist")})
    for lv in ("level1", "level2", "level3"):
        t = (m.get("triage") or {}).get(lv)
        if t:
            o.update({p + f"tri{lv[-1]}_dr1": t["dr_h1"], p + f"tri{lv[-1]}_dr3": t["dr_h3"], p + f"tri{lv[-1]}_fa": t["fa_per_itemyear"],
                      p + f"tri{lv[-1]}_fa_stable": t["fa_per_itemyear_stable"], p + f"tri{lv[-1]}_lead": t["lead_med"]})
    cf = m.get("conformal") or {}
    o.update({p + "conf_cover": cf.get("conf_cover"), p + "conf_size": cf.get("conf_size"),
              p + "conf_cover_pre": cf.get("conf_cover_pre"), p + "conf_cover_post": cf.get("conf_cover_post")})
    fit = m.get("fit") or {}
    o.update({p + "fit_failed_origins": fit.get("n_failed"), p + "fit_refits": fit.get("n_refit"), p + "fit_seconds": fit.get("seconds")})
    return o


def rec_b(key, r):
    cell, _, rep = key
    f = cell_factors_b(cell)
    out = dict(cell=cell, rep=rep, **f, t0=r["t0"], any_failed_fit=r["any_failed_fit"], seconds_batch=r.get("seconds_batch"))
    for name in C.B_MODELS:
        if name in r:
            out.update(model_row(name, r[name]))
    sr = r["M2"].get("sigma_rule")
    if sr:
        for lv in ("level1", "level2", "level3"):
            out.update({f"sigma_{lv[-1]}_dr1": sr[lv]["dr_h1"], f"sigma_{lv[-1]}_dr3": sr[lv]["dr_h3"],
                        f"sigma_{lv[-1]}_fa": sr[lv]["fa_per_itemyear"], f"sigma_{lv[-1]}_fa_stable": sr[lv]["fa_per_itemyear_stable"],
                        f"sigma_{lv[-1]}_lead": sr[lv]["lead_med"]})
        out.update(sigma_arl0=sr["run_length"]["arl0"], sigma_arl1=sr["run_length"].get("arl1"),
                       sigma_stable_alarm_units=sr["run_length"].get("stable_alarm_units"))
    hl = r["M2"].get("headline")
    if hl:
        for lv in ("level1", "level2"):
            out.update({f"head{lv[-1]}_dr1": hl[lv]["dr_h1"], f"head{lv[-1]}_dr3": hl[lv]["dr_h3"],
                        f"head{lv[-1]}_fa_stable": hl[lv]["fa_per_itemyear_stable"], f"head{lv[-1]}_lead": hl[lv]["lead_med"]})
    sev = r["M2"].get("severe")
    if sev:
        out.update(sev_level3=sev["level3_months"], sev_body=sev["severe_body"], sev_tail=sev["severe_tail"])
    for ml in ("xgb", "lstm", "tft"):
        m = r[ml]
        out.update({f"{ml}_brier": m["prob"]["brier"], f"{ml}_aucpr": m["prob"]["aucpr"], f"{ml}_ece": m["prob"]["ece"]})
        for lv in ("level1", "level2", "level3"):
            t = m["triage"][lv]
            out.update({f"{ml}_tri{lv[-1]}_dr1": t["dr_h1"], f"{ml}_tri{lv[-1]}_dr3": t["dr_h3"],
                        f"{ml}_tri{lv[-1]}_fa_stable": t["fa_per_itemyear_stable"], f"{ml}_tri{lv[-1]}_lead": t["lead_med"]})
    for ch in ("ewma", "cusum"):
        c = r["charts"][ch]
        out.update({f"{ch}_dr1": c["events"]["dr_h1"], f"{ch}_dr3": c["events"]["dr_h3"], f"{ch}_fa_stable": c["events"]["fa_per_itemyear_stable"],
                    f"{ch}_lead": c["events"]["lead_med"], f"{ch}_arl0": c["run_length"]["arl0"], f"{ch}_arl1": c["run_length"].get("arl1"),
                    f"{ch}_stable_alarm_units": c["run_length"].get("stable_alarm_units")})
    return out


def summarise(df, by, cols):
    g = df.groupby(by)
    s = g[cols].mean(numeric_only=True)
    s.insert(0, "n", g.size())
    return s.reset_index()


def main():
    out_dir = sys.argv[1]
    dirs = sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    digest = []
    A = load(dirs, "A")
    if A:
        da = pd.DataFrame([rec_a(k, r) for k, r in A.items()])
        da.to_csv(os.path.join(out_dir, "A_replications.csv"), index=False)
        cols = [c for c in da.columns if c not in ("cell", "variant", "rep", "alpha", "mu", "regime", "I", "T", "design")]
        summarise(da, ["cell", "variant", "alpha", "mu", "regime", "I", "T"], cols).to_csv(os.path.join(out_dir, "A_cells.csv"), index=False)
        h = da[da.variant == "hier"]
        for fac in ("alpha", "mu", "regime", "I", "T"):
            summarise(h, [fac], cols).to_csv(os.path.join(out_dir, f"A_hier_by_{fac}.csv"), index=False)
        digest.append(f"Study A: {len(da)} fits, {da.cell.nunique()} cells; failed {da.failed.mean():.1%}, refit {da.n_refit.mean():.1%}")
    B = load(dirs, "B")
    if B:
        db = pd.DataFrame([rec_b(k, r) for k, r in B.items()])
        db.to_csv(os.path.join(out_dir, "B_replications.csv"), index=False)
        cols = [c for c in db.columns if c not in ("cell", "rep", "alpha", "mu", "regime", "t0")]
        summarise(db, ["cell", "alpha", "mu", "regime"], cols).to_csv(os.path.join(out_dir, "B_cells.csv"), index=False)
        summarise(db, ["regime"], cols).to_csv(os.path.join(out_dir, "B_by_regime.csv"), index=False)
        summarise(db, ["alpha"], cols).to_csv(os.path.join(out_dir, "B_by_alpha.csv"), index=False)
        digest.append(f"Study B: {len(db)} replications, {db.cell.nunique()} cells; replications with a failed origin fit {db.any_failed_fit.mean():.1%}")
    R = load(dirs, "R")
    if R:
        dr = pd.DataFrame([rec_r(k, r) for k, r in R.items()])
        dr.to_csv(os.path.join(out_dir, "R_replications.csv"), index=False)
        cols = [c for c in dr.columns if c not in ("cell", "arm", "rep")]
        summarise(dr, ["cell", "arm"], cols).to_csv(os.path.join(out_dir, "R_cells.csv"), index=False)
        digest.append(f"Robustness: {len(dr)} replications, {dr.cell.nunique()} cells")
    open(os.path.join(out_dir, "digest.txt"), "w").write("\n".join(digest) + "\n")
    print("\n".join(digest))


if __name__ == "__main__":
    main()
