"""build_panel_multi.py — the monthly event–indicator panel for many items and one reporter, from raw_<reporter>.csv
(pull_comtrade_multi.py) and restrictions.csv (events_restrictions.py).

Per item and month (from the 13th month of data, because the trailing-twelve-month definitions consume a year):
  N1        trade-based shock events (E1). --e1 share --share-min 0.25 is the definition of the main run: shipments
            from a partner with a trailing-12-month share of at least a quarter fall below HALF x their trailing-12-
            month median. --e1 top (the largest partner's shipments do) and --e1 total (the item's total monthly
            imports do; the script's default) are the two sensitivity definitions. With the built-in share of 0.05
            the partner-level definition fired in a third of all item-months of the Korean panel, far outside the
            prevalence range of the simulation design, and was replaced before the main run (Supplementary Note 14)
  N2        restriction events: export-restriction measures implemented in the month by a partner with a
            trailing-12-month share >= SHARE_MIN whose HS scope covers the item (exact HS6 or a 2-/4-digit parent)
  N2a       the same for announcement months (for lead-time analysis; never a target in the backtest)
  N3        N1 + N2 (union; an item-month counts both kinds)
  indicators (dated the month they become observable; the backtest lags them one month):
            hhi12 (partner HHI of trailing-12-month shares), top_share12, unit_value (USD/kg), total_qty_kg, n_partners12
  exposure: top_share and log unit value over the first EXPO_MONTHS months with data (centred within group later)
Items are screened: at least MIN_MONTHS months with positive imports in the window and at least MIN_E1 trade events,
else dropped and listed in <out>_dropped.csv. Partners are Comtrade partner codes (0 = World is skipped).

usage: python build_panel_multi.py RAW_CSV ITEMS_CSV RESTRICTIONS_CSV OUT_CSV [--start 2011-01] [--end 2025-12]
"""
import argparse, csv, math
from collections import defaultdict

SHARE_MIN, HALF, EXPO_MONTHS, MIN_MONTHS, MIN_E1 = 0.05, 0.5, 36, 60, 1

ap = argparse.ArgumentParser()
ap.add_argument("raw"); ap.add_argument("items"); ap.add_argument("restrictions"); ap.add_argument("out")
ap.add_argument("--start", default="2011-01"); ap.add_argument("--end", default="2025-12")
ap.add_argument("--e1", default="total", choices=["total", "top", "share"],
                help="trade-based event: 'total' = the item's total imports fall below HALF x their trailing-12-month median (default); "
                     "'top' = the largest partner's shipments do; 'share' = any partner with a trailing share >= SHARE_MIN does (the first version)")
ap.add_argument("--half", type=float, default=None, help="override HALF, the fraction of the trailing median that defines a fall (default 0.5)")
ap.add_argument("--share-min", type=float, default=None, help="trailing share a partner needs to count for --e1 share (default: SHARE_MIN = 0.05); "
                                                               "restriction events (N2) always use SHARE_MIN")
ap.add_argument("--min-e1", type=int, default=0, help="drop items with fewer trade events than this (default 0: items are screened on data "
                                                     "coverage only, never on the outcome)")
a = ap.parse_args()
if a.half is not None:
    HALF = a.half
E1_SHARE = a.share_min if a.share_min is not None else SHARE_MIN
MIN_E1 = a.min_e1

items = {r["hs6"].strip(): r for r in csv.DictReader(open(a.items, encoding="utf-8")) if r.get("include", "1").strip() == "1"}
q = defaultdict(float); v = defaultdict(float)
for r in csv.DictReader(open(a.raw, encoding="utf-8")):
    if not r["period"] or r["cmd"] not in items:
        continue
    p = int(float(r["partner"] or 0))
    if p == 0:
        continue
    key = (r["cmd"], str(r["period"])[:6], p)
    q[key] += float(r["qty"] or 0); v[key] += float(r["value"] or 0)

restr = []
for r in csv.DictReader(open(a.restrictions, encoding="utf-8")):
    if not r["partner_code"]:
        continue
    codes = [c.strip() for c in r["hs_codes"].split(";") if c.strip()]
    restr.append(dict(p=int(float(r["partner_code"])), codes=codes, imp=r["implemented"].replace("-", ""), ann=r["announced"].replace("-", ""), measure=r["measure"]))


def covers(codes, hs6):
    return any(hs6 == c or (len(c) < 6 and hs6.startswith(c)) for c in codes)


def months_between(s, e):
    y, m = int(s[:4]), int(s[5:7]); ye, me = int(e[:4]), int(e[5:7]); out = []
    while (y, m) <= (ye, me):
        out.append(f"{y}{m:02d}"); m += 1
        if m == 13:
            y += 1; m = 1
    return out


def prev12(per):
    y, m = int(per[:4]), int(per[4:]); out = []
    for _ in range(12):
        m -= 1
        if m == 0:
            y -= 1; m = 12
        out.append(f"{y}{m:02d}")
    return out


periods = months_between(a.start, a.end)
rows, dropped = [], []
for c, meta in items.items():
    partners = sorted({k[2] for k in q if k[0] == c})
    if not partners:
        dropped.append((c, "no data")); continue
    item_rows = []
    for per in periods:
        pr = prev12(per)
        tr = {p: sum(q.get((c, x, p), 0.0) for x in pr) for p in partners}
        tot12 = sum(tr.values())
        if tot12 <= 0:
            continue
        shares = {p: tr[p] / tot12 for p in partners}
        hhi = sum(s * s for s in shares.values())
        top_p = max(shares, key=shares.get)
        ev1 = []
        if a.e1 == "total":                                                 # the item's total monthly imports fall
            hist = sorted(sum(q.get((c, x, p), 0.0) for p in partners) for x in pr)
            med = 0.5 * (hist[5] + hist[6])
            if med > 0 and sum(q.get((c, per, p), 0.0) for p in partners) < HALF * med:
                ev1.append("total")
        else:                                                               # a partner's shipments fall
            for p in partners:
                if (a.e1 == "top" and p == top_p) or (a.e1 == "share" and shares[p] >= E1_SHARE):
                    hist = sorted(q.get((c, x, p), 0.0) for x in pr)
                    med = 0.5 * (hist[5] + hist[6])
                    if med > 0 and q.get((c, per, p), 0.0) < HALF * med:
                        ev1.append(str(p))
        ev2, ev2a, seen2, seen2a = [], [], set(), set()                  # one event per partner and month, whatever the
        for R in restr:                                                    # number of sources that record the measure
            if shares.get(R["p"], 0.0) >= SHARE_MIN and covers(R["codes"], c):
                if R["imp"] == per and R["p"] not in seen2:
                    ev2.append(f"{R['p']}:{R['measure'][:40]}"); seen2.add(R["p"])
                if R["ann"] == per and R["p"] not in seen2a:
                    ev2a.append(f"{R['p']}:{R['measure'][:40]}"); seen2a.add(R["p"])
        tq = sum(q.get((c, per, p), 0.0) for p in partners); tv = sum(v.get((c, per, p), 0.0) for p in partners)
        item_rows.append(dict(cmd=c, group=meta["group"], period=per, total_qty_kg=round(tq, 1), total_value_usd=round(tv),
                              N1=len(ev1), events1=";".join(ev1), N2=len(ev2), events2=";".join(ev2), N2a=len(ev2a), events2a=";".join(ev2a),
                              N3=len(ev1) + len(ev2), hhi12=round(hhi, 4), top_partner=top_p, top_share12=round(shares[top_p], 4),
                              n_partners12=sum(1 for s in shares.values() if s >= 0.01), unit_value=round(tv / tq, 4) if tq > 0 else ""))
    pos = sum(1 for r in item_rows if r["total_qty_kg"] > 0)
    e1 = sum(r["N1"] for r in item_rows)
    if pos < MIN_MONTHS or e1 < MIN_E1:
        dropped.append((c, f"{pos} months with imports, {e1} trade events")); continue
    rows += item_rows

hdr = ["cmd", "group", "period", "total_qty_kg", "total_value_usd", "N1", "events1", "N2", "events2", "N2a", "events2a", "N3", "hhi12", "top_partner", "top_share12", "n_partners12", "unit_value"]
with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
    for r in rows:
        w.writerow(r)
with open(a.out.replace(".csv", "") + "_dropped.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["hs6", "reason"]); w.writerows(dropped)
kept = sorted({r["cmd"] for r in rows})
print(f"panel: {len(kept)} items kept, {len(dropped)} dropped; months {periods[0]}..{periods[-1]}")
for g in sorted({items[c]['group'] for c in kept}):
    cs = [c for c in kept if items[c]["group"] == g]
    e1 = sum(r["N1"] for r in rows if r["cmd"] in cs); e2 = sum(r["N2"] for r in rows if r["cmd"] in cs)
    print(f"  {g}: {len(cs)} items, trade events {e1}, restriction events {e2}")

