"""Why do Study B origin fits fail? Summarises the attempts stored in the progress pickles."""
import glob, os, pickle, collections, sys
import numpy as np
OUT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/ews_v1_results")
RH, ESS = 1.01, 400
rows = []
for p in sorted(glob.glob(os.path.join(OUT, "_tmp", "B", "batches", "*__*.pkl"))):
    if p.endswith("_ml.pkl") or "ml_partial" in p:
        continue
    st = pickle.load(open(p, "rb"))
    name = os.path.basename(p).split("__")[1]
    for r, rep in enumerate(st["reps"]):
        for f in rep["fits"]:
            for k, a in enumerate(f["attempts"]):
                rows.append(dict(model=name, origin=f["origin"], attempt=k, ok=a["ok"], div=a["n_div"] > 0,
                                 ess=min(a["ess_bulk_min"], a["ess_tail_min"]) <= ESS, rhat=a["rhat_max"] >= RH,
                                 ess_b=a["ess_bulk_min"], ess_t=a["ess_tail_min"], rh=a["rhat_max"], site=a["rhat_site"],
                                 steps=a["mean_steps"], sec=a["seconds"], ndiv=a["n_div"]))
if not rows:
    print("no progress files found under", OUT); sys.exit()
def summ(sel, label):
    n = len(sel)
    if not n: return
    ok = sum(x["ok"] for x in sel)
    print(f"{label}: {n} fits, passed {ok} ({100*ok/n:.0f}%) | fail causes: divergence {sum(x['div'] for x in sel)}, "
          f"ESS<{ESS} {sum(x['ess'] for x in sel)}, R-hat>={RH} {sum(x['rhat'] for x in sel)} | "
          f"ESS bulk min median {np.median([x['ess_b'] for x in sel]):.0f} (10%: {np.percentile([x['ess_b'] for x in sel],10):.0f}), "
          f"tail median {np.median([x['ess_t'] for x in sel]):.0f} (10%: {np.percentile([x['ess_t'] for x in sel],10):.0f}) | "
          f"divergences per fit median {np.median([x['ndiv'] for x in sel]):.0f}, max {max(x['ndiv'] for x in sel)} | "
          f"steps median {np.median([x['steps'] for x in sel]):.0f} | seconds per replication median {np.median([x['sec'] for x in sel]):.0f}")
for model in sorted({x["model"] for x in rows}):
    print("==", model)
    summ([x for x in rows if x["model"] == model and x["attempt"] == 0], "  first fits ")
    summ([x for x in rows if x["model"] == model and x["attempt"] == 1], "  refits     ")
    for o in sorted({x["origin"] for x in rows if x["model"] == model}):
        summ([x for x in rows if x["model"] == model and x["origin"] == o and x["attempt"] == 0], f"  origin {o:3d} first")
sites = collections.Counter(x["site"] for x in rows if x["rhat"])
print("worst R-hat sites among R-hat failures:", dict(sites.most_common(6)))
