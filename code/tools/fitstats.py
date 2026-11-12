"""fitstats.py — NUTS fit statistics from the result files (for §5.5 and the run report).

  python tools/fitstats.py <out> [<out2> ...]

Per study and size: fits, first-attempt pass rate, why first attempts failed (R-hat, ESS,
divergences), final failure rate after the one refit, mean leapfrog steps per iteration,
median seconds per fit (first attempt / refit). Study B: M2 origin fits are read from the
per-replication `fits` records when present, otherwise only the recovery fits are counted.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def _records(out, study):
    for p in sorted(glob.glob(os.path.join(out, study, "*.jsonl"))):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield os.path.basename(p), json.loads(line)
                    except json.JSONDecodeError:
                        pass


def _fit_entries(study, fname, rec):
    """Yield (group key, fit info dict) for every NUTS fit recorded in a result line."""
    if study == "A":
        size = rec["cell"].split("-")[-2:]
        yield f"A {'-'.join(size)} {rec['variant']}", rec
    elif study == "R":
        fi = rec.get("recovery_fit")
        if fi:
            yield f"R recovery {rec.get('arm', '')}", fi
        for k, v in (rec.get("M2") or {}).items() if isinstance(rec.get("M2"), dict) else []:
            pass
    elif study == "B":
        for model in ("M2", "M0", "NBnone", "NBcomplete"):
            m = rec.get(model)
            if isinstance(m, dict) and isinstance(m.get("fit"), dict):
                yield f"B {model}", dict(m["fit"], _b=1)


def summarise(outs):
    rows = defaultdict(list)
    seen = set()
    for out in outs:
        for study in ("A", "B", "R"):
            for fname, rec in _records(out, study):
                key = (study, fname, rec.get("rep"), rec.get("cell"), rec.get("variant", rec.get("arm")))
                if key in seen:
                    continue
                seen.add(key)
                for g, fi in _fit_entries(study, fname, rec):
                    rows[g].append(fi)
    lines = []
    hdr = f"{'group':28s} {'fits':>5s} {'pass1':>6s} {'rhat':>5s} {'ess':>5s} {'div':>5s} {'final_fail':>10s} {'steps':>6s} {'s/fit1':>7s} {'s/refit':>8s}"
    lines.append(hdr)
    for g in sorted(rows):
        fis = rows[g]
        n = len(fis)
        if fis and fis[0].get("_b"):                     # Study B: per-replication origin summaries
            no = sum(f["n_origins"] for f in fis)
            nr = sum(f["n_refit"] for f in fis)
            nf = sum(f["n_failed"] for f in fis)
            sec = np.median([f["seconds"] / f["n_origins"] for f in fis])
            lines.append(f"{g:28s} {no:5d} {1 - nr / no:6.2f} {'':>5s} {'':>5s} {'':>5s} {nf / no:10.3f} {'':>6s} {sec:7.0f} {'(per origin, incl. refits)'}")
            continue
        first = [fi["attempts"][0] for fi in fis if fi.get("attempts")]
        p1 = np.mean([a.get("ok", 0) for a in first]) if first else float("nan")
        bad = [a for a in first if not a.get("ok", 0)]
        why_r = np.mean([a.get("rhat_max", 0) >= 1.01 for a in bad]) if bad else 0.0
        why_e = np.mean([min(a.get("ess_bulk_min", 1e9), a.get("ess_tail_min", 1e9)) <= 400 for a in bad]) if bad else 0.0
        why_d = np.mean([a.get("n_div", 0) > 0 for a in bad]) if bad else 0.0
        fail = np.mean([fi.get("failed", 0) for fi in fis])
        steps = np.mean([a.get("mean_steps", np.nan) for a in first]) if first else float("nan")
        s1 = np.median([a.get("seconds", np.nan) for a in first]) if first else float("nan")
        re = [fi["attempts"][1].get("seconds", np.nan) for fi in fis if len(fi.get("attempts", [])) > 1]
        s2 = np.median(re) if re else float("nan")
        lines.append(f"{g:28s} {n:5d} {p1:6.2f} {why_r:5.2f} {why_e:5.2f} {why_d:5.2f} {fail:10.3f} {steps:6.0f} {s1:7.0f} {s2:8.0f}")
    lines.append("rhat/ess/div = share of failed first attempts with that reason (a fit can have several)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarise(sys.argv[1:] or [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")]))
