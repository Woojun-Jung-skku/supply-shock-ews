"""drop_item.py — the main run of the backtest evaluated without one or more items, from the archived alarm paths
(paths.npz) and the panel. No model is refitted: the probabilities and levels of the remaining items are those of the
archived run, and only the set of items that is evaluated changes.

Written for bismuth (HS 810600), which has no UN Comtrade record under this code after 2021-12 (Supplementary Note 14):
the panel carries it with zero imports through 2022 and has no row for it afterwards.

The definitions are those of backtest_multi.py (event_metrics, prob_metrics, label3 and the band of cost ratios in
which a policy beats both never and always acting). With no item dropped the functions reproduce the metrics_<target>.csv
and cost_bands_<target>.csv of the run, and the script stops if they do not.

usage: python backtest/drop_item.py OUT_DIR PANEL_CSV HS6[,HS6...] [--target N1] [--h 3]
       e.g. python backtest/drop_item.py results/backtest/main backtest/panel_410.csv 810600
"""
import argparse, csv, os, sys
import numpy as np
from sklearn.metrics import average_precision_score

R_GRID = np.exp(np.linspace(np.log(1.0), np.log(100.0), 400))        # as in backtest_multi.py
LABEL = {"M2": "M2 (proposed)", "M0": "Poisson M0", "xgb": "XGBoost (pooled)", "xgbcal": "XGBoost, recalibrated", "sigma": "Sigma-rule (plug-in SRI)",
         "ewma": "EWMA", "cusum": "CUSUM", "farr": "Farrington-type threshold"}


def load(out, panel_csv, target="N1"):
    """Alarm paths of a run and the event counts of the panel on the same item x month grid (zero where the panel has no row)."""
    Z = np.load(os.path.join(out, "paths.npz"), allow_pickle=True)
    items = [str(x) for x in Z["items"]]; periods = [str(x) for x in Z["periods"]]; DEC = Z["DEC"]
    N = np.zeros((len(items), len(periods)), int); have = np.zeros(N.shape, bool)
    for r in csv.DictReader(open(panel_csv, encoding="utf-8")):
        i, t = items.index(r["cmd"]), periods.index(r["period"])
        N[i, t] = int(r[target]); have[i, t] = True
    paths = {}
    for k in Z.files:
        for suffix in ("_levels_dec", "_alarm_dec", "_levels", "_p"):
            if k.endswith(suffix):
                paths.setdefault(k[:-len(suffix)], {})[suffix[1:]] = Z[k]
                break
    return dict(items=items, periods=periods, DEC=DEC, N=N, have=have, paths={m: P for m, P in paths.items() if set(P) & {"levels", "levels_dec", "alarm_dec"}})


def alarms_of(P, k, DEC, keep):
    if "levels" in P:
        return P["levels"][DEC][:, keep] >= k
    if "levels_dec" in P:
        return P["levels_dec"][:, keep] >= k
    return P["alarm_dec"][:, keep]


def event_metrics(A, months, Nc):
    """A: decision months x items. Detection within one and three months, false onsets and alarm months per item-year."""
    T = Nc.shape[1]; shock = Nc >= 1; s0, s1 = months[0], months[-1]; res = {}
    for h in (1, 3):
        det = tot = 0
        for tau in range(s0 + h, s1 + 2):
            if tau > T:
                break
            hit = shock[:, tau - 1]
            if not hit.any():
                continue
            win = A[(tau - h - s0):(tau - s0)]; det += (hit & win.any(0)).sum(); tot += hit.sum()
        res[f"dr_h{h}"] = det / tot if tot else None
    prev = np.vstack([np.zeros((1, A.shape[1]), bool), A[:-1]]); onset = A & ~prev; fa = 0
    for m_idx, i in zip(*np.nonzero(onset)):
        t = months[m_idx]
        if not shock[i, t:t + 3].any():
            fa += 1
    years = A.shape[1] * len(months) / 12.0
    res.update(fa_per_itemyear=fa / years, alarm_months_per_itemyear=12 * float(A.mean()))
    return res


def prob_metrics(p, y):
    p = np.clip(p.ravel(), 0, 1); y = y.ravel().astype(bool)
    res = dict(brier=float(((p - y) ** 2).mean()), base_rate=float(y.mean()), aucpr=float(average_precision_score(y, p)) if y.any() and not y.all() else None)
    bins = np.minimum((p * 10).astype(int), 9); ece = 0.0
    for b in range(10):
        k = bins == b
        if k.any():
            ece += k.mean() * abs(p[k].mean() - y[k].mean())
    res["ece"] = float(ece)
    return res


def evaluate(D, drop=(), h=3):
    """Metrics of every method on the items that are not dropped: {method: {level: {...}}}, plus the event rate E."""
    keep = [i for i, it in enumerate(D["items"]) if it not in set(drop)]
    DEC = D["DEC"]; Nc = D["N"][keep]; T = Nc.shape[1]; I = len(keep)
    E = float((Nc[:, DEC[0] + 1:] >= 1).sum()) / (I * len(DEC) / 12.0)
    y3 = np.array([Nc[:, t:t + h].sum(1) >= 1 for t in range(T)])[DEC]
    out = dict(E=E, items=I, item_months=I * len(DEC), event_item_months=int((Nc[:, DEC[0]:] >= 1).sum()), base_rate_3m=float(y3.mean()), methods={})
    for name, P in D["paths"].items():
        levels = [1, 2, 3] if ("levels" in P or "levels_dec" in P) else [None]
        res, curves = {}, []
        for k in levels:
            A = alarms_of(P, k if k else 1, DEC, keep); em = event_metrics(A, DEC, Nc)
            if "p" in P and k in (None, 2):
                em.update(prob_metrics(P["p"][DEC][:, keep], y3))
            res[k] = em
            dr = round(em["dr_h1"], 4) if em["dr_h1"] is not None else 0.0          # rounded as in backtest_multi.py
            curves.append(em["alarm_months_per_itemyear"] + R_GRID * (1 - dr) * E)
        env = np.min(np.vstack(curves), 0); triv = np.minimum(R_GRID * E, 12.0)
        beats = (env < R_GRID * E) & (env < 12.0); sav = 1 - env / triv; i_ = int(np.argmax(sav))
        res["band"] = dict(band_lo=float(R_GRID[beats].min()) if beats.any() else None, band_hi=float(R_GRID[beats].max()) if beats.any() else None,
                           max_saving=float(sav[i_]), r_at_max=float(R_GRID[i_]))
        out["methods"][name] = res
    return out


def check_against_archive(D, out, target="N1", h=3):
    """With no item dropped the functions above must give the archived metrics and cost bands of the run."""
    ev = evaluate(D, (), h); n = 0
    for r in csv.DictReader(open(os.path.join(out, f"metrics_{target}.csv"), encoding="utf-8")):
        k = int(float(r["level"])) if r["level"] not in ("", "None") else None
        m = ev["methods"][r["method"]][k]
        for key, nd in (("dr_h1", 4), ("dr_h3", 4), ("fa_per_itemyear", 4), ("alarm_months_per_itemyear", 3), ("aucpr", 4), ("brier", 5), ("ece", 5)):
            if r.get(key) not in (None, "", "None") and m.get(key) is not None:
                if abs(float(r[key]) - round(m[key], nd)) > 1.5 * 10 ** -nd:
                    sys.exit(f"drop_item.py does not reproduce {r['method']} level {k} {key}: archived {r[key]}, recomputed {m[key]:.6f}")
                n += 1
    for r in csv.DictReader(open(os.path.join(out, f"cost_bands_{target}.csv"), encoding="utf-8")):
        b = ev["methods"][r["method"]]["band"]
        for key, nd in (("band_lo", 2), ("band_hi", 2), ("max_saving", 4), ("r_at_max", 2)):
            if r.get(key) not in (None, "", "None") and b.get(key) is not None:
                if abs(float(r[key]) - round(b[key], nd)) > 1.5 * 10 ** -nd:
                    sys.exit(f"drop_item.py does not reproduce the cost band of {r['method']} ({key}): archived {r[key]}, recomputed {b[key]:.6f}")
                n += 1
    return ev, n


def compare(out, panel_csv, drop, target="N1", h=3):
    """(all items, without the dropped items, number of archived values reproduced)."""
    D = load(out, panel_csv, target)
    missing = [d for d in drop if d not in D["items"]]
    if missing:
        sys.exit(f"not in the panel: {missing}")
    full, n = check_against_archive(D, out, target, h)
    return full, evaluate(D, drop, h), n


def record_gap(D, item):
    """(months of the decision window in which the item has no panel row, its event item-months in the decision window)."""
    i = D["items"].index(item); DEC = D["DEC"]
    return int((~D["have"][i, DEC]).sum()), int((D["N"][i, DEC[0]:] >= 1).sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("panel"); ap.add_argument("drop"); ap.add_argument("--target", default="N1"); ap.add_argument("--h", type=int, default=3)
    a = ap.parse_args()
    drop = a.drop.split(",")
    full, sub, n = compare(a.out, a.panel, drop, a.target, a.h)
    print(f"archived metrics and cost bands reproduced with all items ({n} values checked)")
    D = load(a.out, a.panel, a.target)
    for d in drop:
        gap, ev_n = record_gap(D, d)
        print(f"{d}: {gap} of {len(D['DEC'])} decision months without a panel row; {ev_n} event item-months in the decision window")
    for tag, ev in (("all items", full), ("without " + ",".join(drop), sub)):
        print(f"\n== {tag}: {ev['items']} items, {ev['item_months']} item-months on the grid, {ev['event_item_months']} event item-months, "
              f"E = {ev['E']:.3f} per item-year, three-month prevalence {ev['base_rate_3m']:.3f}")
        print(f"{'method':28s} AUC-PR  Brier    ECE  det<=1m det<=3m  FA/iy  AM/iy   band of r      max saving")
        for name, res in ev["methods"].items():
            m = res[2] if 2 in res else res[None]; b = res["band"]
            pr = (f"{m['aucpr']:6.3f} {m['brier']:6.3f} {m['ece']:6.3f}" if "aucpr" in m and m["aucpr"] is not None else "    --     --     --")
            band = f"{b['band_lo']:.1f}-{b['band_hi']:.1f}" if b["band_lo"] is not None else "none"
            print(f"{LABEL.get(name, name):28s} {pr}   {m['dr_h1']:.3f}   {m['dr_h3']:.3f}  {m['fa_per_itemyear']:.3f}  {m['alarm_months_per_itemyear']:5.2f}   {band:12s}  {100 * b['max_saving']:.1f}% at r = {b['r_at_max']:.1f}")
