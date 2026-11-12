"""v1.4: pooled ARL0 of a change channel over Study B replications.

A replication's ARL0 is (stable-window unit-months) / max(alarm months, 1), so it is censored at the
window length when the channel raises no alarm there — which is the usual case for the q channels.
The pooled estimate is the ratio of sums over replications: total stable-window unit-months / total
alarm months, with a bootstrap over replications. Study B: I = 30 items, G = 3 groups, decision months
36-117 (82 months), change at t0 = 60 (24 decision months before it)."""
import numpy as np

DEC_MONTHS, PRE_MONTHS = 82, 24
UNITS = dict(q=3, item=30)               # q channels run per group, sigma-rule / EWMA / CUSUM per item


def window_size(regime, units):
    return units * np.where(np.asarray(regime) == "stable", DEC_MONTHS, PRE_MONTHS)


def alarm_months(arl0, share, size):
    """Alarm months in the stable window recovered from ARL0 = size / max(alarms, 1) (rounded to 0.01)
    and the share of units with any alarm (0 -> no alarm)."""
    a, s, z = (np.asarray(v, float) for v in (arl0, share, size))
    out = np.where(a < z - 0.5, np.round(z / np.where(a > 0, a, np.nan)), np.where(s > 0, 1.0, 0.0))
    out[~np.isfinite(a)] = np.nan
    return out


def pooled(d, arl0_col, share_col, units, B=2000, seed=0):
    """-> dict(est, se, lo, hi (90% bootstrap), alarms, unit_months, n, no_alarm_share)."""
    size = window_size(d["regime"].values, units).astype(float)
    al = alarm_months(d[arl0_col].values, d[share_col].values, size)
    ok = np.isfinite(al)
    size, al = size[ok], al[ok]
    if not al.size:
        return dict(est=np.nan, se=np.nan, lo=np.nan, hi=np.nan, alarms=0, unit_months=0, n=0, no_alarm_share=np.nan)
    est = size.sum() / max(al.sum(), 1.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, al.size, (B, al.size))
    bs = size[idx].sum(1) / np.maximum(al[idx].sum(1), 1.0)
    lo, hi = np.percentile(bs, [5, 95])
    return dict(est=float(est), se=float(bs.std(ddof=1)), lo=float(lo), hi=float(hi), alarms=int(al.sum()),
                unit_months=int(size.sum()), n=int(al.size), no_alarm_share=float((al == 0).mean()))
