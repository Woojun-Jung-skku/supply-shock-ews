"""A7 — complete pooling with the true item-effect scale tau_C in {0, 0.6} (manuscript §5.2 (i)).

Reference cell of the pooling comparison (gradual, alpha' = 2, mu-bar = 0.2, I = 30, T = 120), replications
0..4 with the Study A seeds, the item effects u_i ~ N(0, tau_C^2) re-drawn at the arm's tau_C (same
normal draws, scaled), everything else as in Table 4. Fits: complete pooling (one alpha', u = 0) at
tau_C = 0 (correctly specified) and 0.6 (heterogeneity doubled), and the hierarchical model at 0.6.
The tau_C = 0.3 fits are the existing Study A results.
Writes <out>/A7_tauC.jsonl (one line per fit, metrics.recovery_q) and A7_tauC_summary.txt.
usage: python followup/a7_tauC.py OUT_DIR [arms]   arms e.g. complete:0 complete:0.6 hier:0.6
"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from numpy.polynomial.hermite_e import hermegauss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as C
C.apply_profile("t4_min")
import dgp
import models as Mo
import nuts_gpu as NG
import metrics as MT
import study_a

CELL = [c for c in study_a.cells() if c["reference"]][0]            # A-a2-mu0.2-gradual-I30-T120
REPS = (0, 1, 2, 3, 4)


def simulate(r, tau_c):
    old = C.SIGMA_U
    C.SIGMA_U = tau_c
    try:
        s = dgp.simulate(CELL["I"], CELL["T"], float(CELL["mu"]), float(CELL["alpha"]), CELL["regime"],
                         seed=dgp.seed_of("A", CELL["id"], r))
    finally:
        C.SIGMA_U = old
    s["miss"] = C.MISS_DEFAULT
    return s


def pseudo_true(mu_vals, a_true, tau, nmax=80):
    """Dispersion that maximises the expected NB log-likelihood when u ~ N(0, tau^2) is omitted."""
    xs, ws = hermegauss(40); ws = ws / ws.sum()
    n = np.arange(nmax + 1)
    def nb(n, mu, a):
        return gammaln(n + a) - gammaln(a) - gammaln(n + 1) + a * np.log(a / (a + mu)) + n * np.log(mu / (a + mu))
    def f(a):
        tot = 0.0
        for mu in mu_vals:
            p = sum(w * np.exp(nb(n, mu * np.exp(tau * x), a_true)) for x, w in zip(xs, ws))
            tot += (p * nb(n, mu, a)).sum()
        return -tot
    return minimize_scalar(f, bounds=(0.05, 50), method="bounded").x


def pseudo_true_cell(tau, reps=REPS, nmax=60, sub=4000, seed=0):
    """Pseudo-true dispersion of complete pooling on the item means of the reference-cell replications themselves
    (mu_it without u_i = exp(m_g(t) + beta'x + w'z_i), the month-0 indicator approximated by 0), maximising the expected
    NB log-likelihood under the true process over the dispersion AND a free common scale c (the fitted model adjusts
    its level): returns (alpha_mean_free, c, alpha_mean_held)."""
    from scipy.optimize import minimize
    xs, ws = hermegauss(40); ws = ws / ws.sum()
    n = np.arange(nmax)
    def nb(n, mu, a):
        return gammaln(n + a) - gammaln(a) - gammaln(n + 1) + a * np.log(a / (a + mu)) + n * np.log(mu / (a + mu))
    mu0 = []
    for r in reps:
        s = simulate(r, tau)
        tr = s["truth"]; I, T = s["N"].shape
        wz = s["Z"] @ tr["w"]
        Xl = np.concatenate([np.zeros((I, 1, s["X"].shape[2])), np.nan_to_num(s["X"][:, :T - 1])], 1)
        mu0.append(np.exp(tr["m"][s["group"]] + np.einsum("itk,k->it", Xl, tr["beta"]) + wz[:, None]).ravel())
    mu0 = np.concatenate(mu0)
    rng = np.random.default_rng(seed)
    m0 = mu0[rng.choice(mu0.size, min(sub, mu0.size), replace=False)]
    P = np.zeros((m0.size, n.size))
    for x, w in zip(xs, ws):
        P += w * np.exp(nb(n[None, :], (m0 * np.exp(tau * x))[:, None], 2.0))
    def f(th):
        a, c = np.exp(th)
        return -(P * nb(n[None, :], (c * m0)[:, None], a)).sum()
    free = minimize(f, [np.log(1.5), 0.0], method="Nelder-Mead")
    held = minimize(lambda t: f([t[0], 0.0]), [np.log(1.5)], method="Nelder-Mead")
    return float(np.exp(free.x[0])), float(np.exp(free.x[1])), float(np.exp(held.x[0]))


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    arms = sys.argv[2:] or ["complete:0", "complete:0.6", "hier:0.6"]
    path = os.path.join(out, "A7_tauC.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            rec = json.loads(line)
            done.add((rec["variant"], rec["tau_C"], rec["rep"]))
    log = NG.LOG[0]
    I, T = CELL["I"], CELL["T"]
    cap = Mo.mcap(I, T, C.K, C.MISS_DEFAULT)
    for arm in arms:
        variant, tau = arm.split(":")
        tau = float(tau)
        todo = [r for r in REPS if (variant, tau, r) not in done]
        if not todo:
            continue
        sims = [simulate(r, tau) for r in todo]
        ds = [Mo.prep(s, T, "latent", cap) for s in sims]
        fitter = study_a.fitter_for(variant, ds[0], I, T)
        seeds = [dgp.seed_of("A-fit", CELL["id"], variant, r) % 2 ** 31 for r in todo]
        t0 = time.time()
        res, info, _ = fitter.fit(ds, seeds, batch=len(ds))
        host = {k: (np.asarray(v) if not isinstance(v, dict) else {kk: np.asarray(vv) for kk, vv in v.items()})
                for k, v in res.items()}
        with open(path, "a") as f:
            for j, r in enumerate(todo):
                q = {k: {kk: vv[j] for kk, vv in v.items()} for k, v in host.items() if k != "cp"}
                tr = sims[j]["truth"]
                rec = dict(rep=r, cell=CELL["id"], variant=variant, tau_C=tau, design=C.DESIGN, **info[j])
                rec["truth"] = dict(alpha=tr["alpha"], kappa_w=tr["kappa_w"], tau_C=tau)
                rec["recovery"] = MT.recovery_q(q, tr, variant)
                f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)) + "\n")
        log(f"A7 {variant} tau_C={tau}: {len(todo)} fits in {time.time() - t0:.0f}s, failed {sum(i['failed'] for i in info)}")
    # summary
    recs = [json.loads(l) for l in open(path)]
    lines = ["A7: complete pooling and the item-effect scale (reference cell of the pooling comparison, reps 0-4)"]
    mus = np.outer(0.2 * np.exp(0.02 * np.r_[np.zeros(60), np.arange(1, 61)]), np.exp(np.linspace(-1.0, 1.0, 9))).ravel()
    for tau in (0.0, 0.3, 0.6):
        cv2 = (1 + 1 / 2.0) * np.exp(tau ** 2) - 1
        pf, c, ph = pseudo_true_cell(tau)
        lines.append(f"tau_C = {tau}: variance-matching alpha' {1 / cv2:.2f} ({100 * (1 / cv2 / 2 - 1):+.0f}%), "
                     f"KL pseudo-true on a stylised mean grid {pseudo_true(mus, 2.0, tau):.2f}; on the reference-cell means: "
                     f"{pf:.2f} with the level free (scale {c:.3f}), {ph:.2f} with the level held (manuscript: level free)")
    for variant in ("complete", "hier"):
        for tau in (0.0, 0.3, 0.6):
            rr = [x for x in recs if x["variant"] == variant and x["tau_C"] == tau]
            if not rr:
                continue
            a50 = np.array([x["recovery"]["alpha"]["q50"] for x in rr]).ravel()
            a05 = np.array([x["recovery"]["alpha"]["q05"] for x in rr]).ravel()
            a95 = np.array([x["recovery"]["alpha"]["q95"] for x in rr]).ravel()
            cov = ((a05 <= 2.0) & (2.0 <= a95)).mean()
            lcr = np.mean([x["recovery"]["logC_summary"]["cover"] for x in rr])
            lrm = np.mean([x["recovery"]["logC_summary"]["rmse"] for x in rr])
            fails = sum(x["failed"] for x in rr)
            lines.append(f"{variant:9s} tau_C={tau}: n={len(rr)} fits (failed {fails}); alpha' posterior median: "
                         f"geo-mean {np.exp(np.log(a50).mean()):.2f} ({100 * (np.exp(np.log(a50).mean()) / 2 - 1):+.0f}%), "
                         f"range {a50.min():.2f}-{a50.max():.2f}; 90% cover {cov:.2f}; log width {np.mean(np.log(a95) - np.log(a05)):.2f}; "
                         f"log C cover {lcr:.2f}, RMSE {lrm:.2f}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(out, "A7_tauC_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
