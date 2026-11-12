"""Table 5(b) draft (Study B): one row per method, pooled over the cells (mean over replications and,
in brackets, the Monte Carlo standard error). Columns: probability quality (AUC-PR, Brier, ECE),
event performance of the triage level (detection within h = 1 and 3 months, false alarms per
item-year in the stable windows, median lead time), run lengths of the change channel (ARL0 in the
stable windows, ARL1 after t0) and conformal coverage / set size.
usage: python tools/table5b.py ANALYSIS_DIR [level] [regime]
  level  = triage level used for the event columns (1, 2 or 3; default 2)
  regime = restrict to one regime (stable / gradual / abrupt; default all)"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arl as ARL

METHODS = [  # (label, prefix of probability / triage columns, run-length prefix, conformal prefix)
    ("M2 (proposed)", "M2_", "M2_q_", "M2_"),
    ("M0 Poisson", "M0_", "M0_q_", "M0_"),
    ("NB no pooling", "NBnone_", "NBnone_q_", "NBnone_"),
    ("NB complete pooling", "NBcomplete_", "NBcomplete_q_", "NBcomplete_"),
    ("sigma-rule (plug-in SRI)", "sigma_", "sigma_", None),
    ("EWMA", "ewma_", "ewma_", None),
    ("CUSUM", "cusum_", "cusum_", None),
    ("XGBoost", "xgb_", None, None),
    ("LSTM", "lstm_", None, None),
    ("TFT", "tft_", None, None),
]


def _ms(v):
    v = pd.to_numeric(v, errors="coerce").dropna().values
    if not v.size:
        return np.nan, np.nan, 0
    se = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan
    return float(v.mean()), float(se), int(v.size)


def _col(d, *names):
    for n in names:
        if n in d.columns:
            return d[n]
    return pd.Series(dtype=float)


def table(d, level=2):
    rows = []
    for label, p, rl, cf in METHODS:
        r = dict(method=label)
        for k in ("aucpr", "brier", "ece"):
            r[k] = _ms(_col(d, p + k))
        if p in ("sigma_",):
            ev = lambda k: _col(d, f"sigma_{level}_{k}")
        elif p in ("ewma_", "cusum_"):
            ev = lambda k, p=p: _col(d, p + k)
        else:
            ev = lambda k, p=p: _col(d, f"{p}tri{level}_{k}")
        r["dr1"], r["dr3"] = _ms(ev("dr1")), _ms(ev("dr3"))
        r["fa_stable"], r["lead"] = _ms(ev("fa_stable")), _ms(ev("lead"))
        if rl:
            # ARL0: ratio of sums over replications (a replication's ARL0 is censored at the window length
            # when the channel raises no alarm in it); bootstrap SE over replications. See tools/arl.py.
            share = rl + "stable_alarm_units"
            if share in d.columns and len(d):
                pr = ARL.pooled(d, rl + "arl0", share, ARL.UNITS["q" if rl.endswith("q_") else "item"])
                r["arl0"] = (pr["est"], pr["se"], pr["n"])
                r["arl0_no_alarm_share"] = (pr["no_alarm_share"], np.nan, pr["n"])
            else:
                r["arl0"] = _ms(_col(d, rl + "arl0"))
            r["arl1"] = _ms(_col(d, rl + "arl1"))
        if cf:
            r["conf_cover"], r["conf_size"] = _ms(_col(d, cf + "conf_cover")), _ms(_col(d, cf + "conf_size"))
        rows.append(r)
    return rows


def fmt(rows, digits=3):
    cols = ["aucpr", "brier", "ece", "dr1", "dr3", "fa_stable", "lead", "arl0", "arl1", "conf_cover", "conf_size"]
    head = "| Method | " + " | ".join(cols) + " |"
    out = [head, "|" + "---|" * (len(cols) + 1)]
    for r in rows:
        cells = []
        for c in cols:
            m, se, n = r.get(c, (np.nan, np.nan, 0))
            if not np.isfinite(m):
                cells.append("–")
            else:
                d = 1 if c in ("arl0", "arl1", "lead") else digits
                cells.append(f"{m:.{d}f}" + (f" ({se:.{d}f})" if np.isfinite(se) else ""))
        out.append(f"| {r['method']} | " + " | ".join(cells) + " |")
    n = max((r[c][2] for r in rows for c in cols if c in r), default=0)
    out.append(f"\nmean (Monte Carlo SE) over replications; up to {n} replications per entry. ARL0: total "
               "stable-window unit-months / total alarm months over the replications (bootstrap SE).")
    return "\n".join(out)


if __name__ == "__main__":
    d = pd.read_csv(os.path.join(sys.argv[1], "B_replications.csv"))
    level = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    if len(sys.argv) > 3:
        d = d[d.regime == sys.argv[3]]
    rows = table(d, level)
    txt = fmt(rows)
    print(txt)
    tag = f"_L{level}" + (f"_{sys.argv[3]}" if len(sys.argv) > 3 else "")
    open(os.path.join(sys.argv[1], f"table5b{tag}.md"), "w").write(txt + "\n")
    pd.DataFrame([{k: (v[0] if isinstance(v, tuple) else v) for k, v in r.items()} for r in rows]).to_csv(
        os.path.join(sys.argv[1], f"table5b{tag}.csv"), index=False)
