"""mcmc_diag.py — rank-normalised split-R̂ and bulk ESS (Vehtari et al., 2021) for the stored draws of a rolling-origin
fit (fits/fit_<t>_<variant>.npz, draws stored chain-major: chains x samples).  Used by backtest_multi.py for the
convergence check of the backtest (Supplementary Note 14) and as a command-line report:
    python mcmc_diag.py OUT_DIR [--chains 2] [--rhat 1.01] [--ess 400]
"""
import argparse, glob, os, sys
import numpy as np
from scipy.special import ndtri

CHECK = ("beta", "alpha", "logC", "m", "d", "sig", "tau")          # quantities checked (alpha = alpha_g; m, d = last month)


def _rank_normalise(x):
    """x: (chains, draws, ...) -> rank-normalised values over all chains and draws, per parameter."""
    C, n = x.shape[:2]
    flat = x.reshape(C * n, -1)
    out = np.empty_like(flat, dtype=float)
    for j in range(flat.shape[1]):
        r = np.argsort(np.argsort(flat[:, j], kind="stable"), kind="stable") + 1.0          # ranks 1..S (ties broken by order)
        out[:, j] = ndtri((r - 3 / 8) / (C * n + 1 / 4))
    return out.reshape(x.shape)


def _rhat_basic(z):
    """split-R̂ of (chains, draws, P) values (already transformed)."""
    C, n = z.shape[:2]
    h = n // 2
    z = np.concatenate([z[:, :h], z[:, h:2 * h]], 0)                                        # split chains
    C2 = z.shape[0]
    mean_c = z.mean(1); var_c = z.var(1, ddof=1)
    W = var_c.mean(0); B = mean_c.var(0, ddof=1) * h
    var_plus = (h - 1) / h * W + B / h
    return np.sqrt(var_plus / np.maximum(W, 1e-300))


def _acov(x):
    """autocovariance of a 1-d sequence via FFT (biased, as in Stan)."""
    n = len(x); x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conjugate(f))[:n] / n
    return ac


def _ess_basic(z):
    """bulk ESS of (chains, draws, P) values (Geyer initial positive and monotone sequence, as in Stan/ArviZ)."""
    C, n = z.shape[:2]
    h = n // 2
    z = np.concatenate([z[:, :h], z[:, h:2 * h]], 0); C2 = z.shape[0]; n = h
    out = np.empty(z.shape[2])
    for j in range(z.shape[2]):
        acs = np.array([_acov(z[c, :, j]) for c in range(C2)])                             # (C2, n) biased autocovariances
        chain_var = acs[:, 0] * n / (n - 1)
        mean_var = chain_var.mean()
        var_plus = mean_var * (n - 1) / n + (z[:, :, j].mean(1).var(ddof=1) if C2 > 1 else 0.0)
        rho = 1 - (mean_var - acs.mean(0)) / var_plus                                       # rho[0] ~ 1
        tau = -1.0; t = 0; prev = np.inf
        while t + 1 < n:
            pair = rho[t] + rho[t + 1]
            if pair <= 0:
                break
            pair = min(pair, prev); prev = pair
            tau += 2 * pair; t += 2
        out[j] = C2 * n / max(tau, 1e-9)
    return out


def diagnostics(draws, chains):
    """draws: dict name -> (S, ...) chain-major; returns dict name -> (max rank-R̂, min bulk ESS) and the overall worst."""
    res = {}
    for k in CHECK:
        if k not in draws:
            continue
        x = np.asarray(draws[k], float)
        S = x.shape[0]; n = S // chains
        x = x[:chains * n].reshape(chains, n, -1)
        if x.shape[2] == 0 or np.allclose(x.std(), 0):
            res[k] = (1.0, float(S)); continue
        zr = _rank_normalise(x)
        zf = _rank_normalise(np.abs(x - np.median(x.reshape(-1, x.shape[2]), 0)))           # folded, for scale
        rh = np.maximum(_rhat_basic(zr), _rhat_basic(zf))
        ess = _ess_basic(zr)
        res[k] = (float(np.nanmax(rh)), float(np.nanmin(ess)))
    worst_r = max(v[0] for v in res.values()); worst_e = min(v[1] for v in res.values())
    return res, worst_r, worst_e


def check(draws, chains, rhat_max=1.01, ess_min=400):
    res, wr, we = diagnostics(draws, chains)
    return (wr < rhat_max and we >= ess_min), res, wr, we


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--rhat", type=float, default=1.01); ap.add_argument("--ess", type=float, default=400)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.out, "fits", "fit_*_M*.npz")), key=lambda s: (s.split("_")[-1], int(s.split("_")[-2])))
    print("fit            pass   maxR̂   minESS  " + "  ".join(f"{k:>12s}" for k in CHECK))
    for f in files:
        z = np.load(f, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        ok, res, wr, we = check(d, a.chains, a.rhat, a.ess)
        print(f"{os.path.basename(f):14s} {'ok' if ok else 'FAIL':5s} {wr:6.3f} {we:7.0f}  " +
              "  ".join(f"{res[k][0]:5.3f}/{res[k][1]:6.0f}" if k in res else " " * 12 for k in CHECK))
