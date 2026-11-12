"""charts.py — conventional baselines of §5.1: sigma-rule on the plug-in SRI, EWMA and CUSUM on counts.

EWMA and CUSUM are Poisson-calibrated, as such charts are in practice: the in-control mean of each
item is its mean count over months 13-36, and the limit is set so that the per-month alarm
probability under Poisson variation at that mean is 1/ARL0_TARGET (nominal ARL0 = 24 months).
Section 4.4 / Fig. 5 is the argument that this nominal value is not what the chart delivers under
over-dispersion; the simulation measures the delivered value.
"""
import numpy as np
from scipy.signal import lfilter
import config as C

_CAL_MONTHS = 20000


def _ewma(N, lam, z0):
    """N (I, T) -> EWMA statistic (I, T), z_0 = z0."""
    zi = (1 - lam) * z0
    out, _ = lfilter([lam], [1, -(1 - lam)], N, axis=1, zi=zi[:, None])
    return out


def _limit(stat, target):
    """Smallest h with P(stat > h) <= 1/target per item (stat (I, n) under the in-control model)."""
    return np.quantile(stat, 1.0 - 1.0 / target, axis=1, method="higher")


def chart_alarms(N, seed, t_first, t_last, rho=C.RHO_SHIFT, lam=C.EWMA_LAMBDA):
    """Alarm indicators (months t_first..t_last as decision months, 0-based month index) for EWMA and
    CUSUM; decision month t uses counts up to month t (columns < t)."""
    rng = np.random.default_rng(seed)
    a, b = C.CHART_PHASE1
    mu0 = np.maximum(N[:, a:b].mean(1), 0.25 / 24)
    sim = rng.poisson(mu0[:, None], size=(N.shape[0], _CAL_MONTHS)).astype(float)
    mu1 = rho * mu0
    k = (mu1 - mu0) / np.log(mu1 / mu0)                      # Lucas (1985) reference value
    h_e = _limit(_ewma(sim, lam, mu0)[:, 100:], C.ARL0_TARGET)
    h_c = _limit(_cusum_k(sim, k)[:, 100:], C.ARL0_TARGET)
    run = N[:, b:].astype(float)                              # charts start after phase I (month 37 on)
    e = _ewma(run, lam, mu0)
    c = _cusum_k(run, k)
    months = np.arange(t_first, t_last + 1)
    # decision month t sees counts of months <= t, i.e. column t - 1 -> index t - 1 - b of `run`
    idx = months - 1 - b
    ok = idx >= 0
    ea = np.zeros((len(months), N.shape[0]), bool)
    ca = np.zeros_like(ea)
    ea[ok] = (e[:, idx[ok]] > h_e[:, None]).T
    ca[ok] = (c[:, idx[ok]] > h_c[:, None]).T
    return dict(ewma=ea, cusum=ca)


def _cusum_k(N, k):
    I, T = N.shape
    c = np.zeros(I)
    out = np.zeros((I, T))
    for t in range(T):
        c = np.maximum(0.0, c + N[:, t] - k)
        out[:, t] = c
    return out


def sigma_rule(sri, hist, limits=C.SIGMA_LIMITS, win=C.SIGMA_WINDOW):
    """sri (M, I): plug-in SRI at decision months; hist (win, I): values before the first one.
    Level k when SRI < mean - limits[k] * sd of the previous `win` values (no persistence)."""
    series = np.vstack([hist, sri])
    M = sri.shape[0]
    lev = np.zeros(sri.shape, int)
    for t in range(M):
        w = series[t:t + win]
        mu, sd = w.mean(0), w.std(0, ddof=1)
        for k, c in enumerate(limits, 1):
            lev[t] = np.where(sri[t] < mu - c * sd, k, lev[t])
    return lev
