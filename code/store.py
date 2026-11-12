"""store.py — append-only results. One JSONL file per (study, cell, variant); every finished
replication is one line, written with flush + fsync. Nothing is ever overwritten or deleted:
a replication already in the file is skipped, and a torn last line (a kill during the write) is
ignored when reading. Temporary progress files live under <out>/_tmp/ and are separate from results.
"""
import json
import os
import numpy as np


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def path(out, study, cell, variant):
    return os.path.join(out, study, f"{cell}__{variant}.jsonl")


def load(out, study, cell, variant):
    p = path(out, study, cell, variant)
    reps = {}
    if os.path.exists(p):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            reps[r["rep"]] = r
    return reps


def append(out, study, cell, variant, rec):
    p = path(out, study, cell, variant)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(_clean(rec)) + "\n")
        f.flush()
        os.fsync(f.fileno())


def tmp_dir(out, study, cell):
    d = os.path.join(out, "_tmp", study, cell)
    os.makedirs(d, exist_ok=True)
    return d
