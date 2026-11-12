"""v1.4: append replications from another result folder (e.g. the Mac's) that the destination does not
have yet; files are matched by name, replications by their 'rep' index. Nothing is overwritten.
usage: python tools/merge_results.py SRC_DIR DST_DIR"""
import glob, json, os, sys
src, dst = sys.argv[1], sys.argv[2]
tot = 0
for p in sorted(glob.glob(os.path.join(src, "*", "*.jsonl"))):
    study = os.path.basename(os.path.dirname(p))
    d = os.path.join(dst, study, os.path.basename(p))
    os.makedirs(os.path.dirname(d), exist_ok=True)
    have = {json.loads(l)["rep"] for l in open(d) if l.strip()} if os.path.exists(d) else set()
    add = [l for l in open(p) if l.strip() and json.loads(l)["rep"] not in have]
    if add:
        with open(d, "a") as g:
            for l in add:
                g.write(l if l.endswith("\n") else l + "\n")
        print(f"{study}/{os.path.basename(p)}: +{[json.loads(l)['rep'] for l in add]}")
    tot += len(add)
print("added", tot)
