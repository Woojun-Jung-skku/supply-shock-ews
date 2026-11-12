"""events_restrictions.py — one table of export-restriction measures (partner, HS scope, announcement and
implementation months) from three sources: the seed list restrictions_seed.csv (hand-checked measures), a Global Trade
Alert CSV export and an OECD Inventory of Export Restrictions on Industrial Raw Materials export. The two downloaded
files have their own column names, which vary between releases: pass a JSON mapping (see --help) and the script prints
the columns it found so the mapping can be fixed.

usage: python events_restrictions.py OUT_CSV --seed restrictions_seed.csv [--gta gta_export.csv --gta-map gta_map.json]
                                            [--oecd oecd_export.csv --oecd-map oecd_map.json] [--items items_hs6.csv]
Output columns: source, partner_iso3, partner_code, measure, announced (YYYY-MM), implemented (YYYY-MM), hs_codes
(';'-separated 2-, 4- or 6-digit codes), scope_level, note. Only measures that touch at least one listed item (exact
HS6 or a 2-/4-digit parent) are kept when --items is given.

Default GTA mapping (edit to match the export): {"jurisdiction": "Implementing Jurisdiction", "iso3": "Implementing Jurisdiction ISO3",
 "type": "Intervention Type", "announced": "Date Announced", "implemented": "Date Implemented", "products": "Affected Products",
 "title": "Title", "export_types": ["Export ban", "Export licensing requirement", "Export quota", "Export tax", "Export-related non-tariff measure, nes"]}
Default OECD mapping (QDD export field codes): {"country": "cou", "product": "HS6 code", "measure": "tom", "year": "year", "start": "dati", "end": "end", "date_order": "dmy", "direction": "dir", "direction_keep": "introduction|increase|extension|revision"} (records whose direction field does not match direction_keep, i.e. eliminations, decreases and unchanged measures, are not events)
The GTA mapping also accepts "affected" + "affected_match" (keep a measure only when this text is in its affected-jurisdictions
field, e.g. "Republic of Korea"), "n_products" + "max_products" (drop economy-wide frameworks that list more HS6 codes than this)
and "exclude_title" (a regular expression; sanctions packages, entity lists, catch-all rules). An implementing field with several
ISO3 codes (an EU measure) gives one row per member state. Both mappings accept "date_order": "mdy" | "dmy" for slash dates. Templates: gta_map.json, oecd_map_qdd.json (edit after reading the printed columns).
ISO3 -> Comtrade partner code uses a small built-in table; unknown codes are written with partner_code empty (fill them from the Comtrade reference list).
"""
import argparse, csv, json, os, re, sys

ISO3_TO_COMTRADE = {"CHN": 156, "JPN": 392, "USA": 842, "IDN": 360, "COD": 180, "ZWE": 716, "RUS": 643, "IND": 699, "AUS": 36, "CHL": 152,
                    "ARG": 32, "BRA": 76, "ZAF": 710, "MYS": 458, "VNM": 704, "THA": 764, "PHL": 608, "TUR": 792, "KAZ": 398, "MNG": 496,
                    "DEU": 276, "FRA": 250, "GBR": 826, "NLD": 528, "BEL": 56, "ITA": 380, "ESP": 724, "CAN": 124, "MEX": 484, "PER": 604,
                    "NAM": 516, "GAB": 266, "GIN": 324, "NGA": 566, "EGY": 818, "SAU": 682, "ARE": 784, "QAT": 634, "IRN": 364, "UKR": 804,
                    "KOR": 410, "TWN": 490, "SGP": 702, "HKG": 344, "NOR": 578, "FIN": 246, "SWE": 752, "AUT": 40, "POL": 616, "CZE": 203,
                    "MDG": 450, "MOZ": 508, "TZA": 834, "RWA": 646, "BDI": 108, "ZMB": 894, "BOL": 68, "MAR": 504, "BIH": 70, "MMR": 104, "LAO": 418,
                    "CHE": 757, "KEN": 404, "GTM": 320, "TJK": 762, "SEN": 686, "AFG": 4, "JAM": 388, "ETH": 231, "UZB": 860, "LKA": 144,
                    "GHA": 288, "NZL": 554, "BLR": 112, "COL": 170, "KGZ": 417, "SLE": 694, "HND": 340, "VEN": 862, "UGA": 800, "DZA": 12,
                    "ISR": 376, "NCL": 540, "TGO": 768, "BWA": 72, "BGR": 100, "HRV": 191, "CYP": 196, "DNK": 208, "EST": 233, "GRC": 300,
                    "HUN": 348, "IRL": 372, "LVA": 428, "LTU": 440, "LUX": 442, "MLT": 470, "PRT": 620, "ROU": 642, "SVK": 703, "SVN": 705,
                    "TUN": 788, "GAB": 266, "NAM": 516, "GIN": 324, "ECU": 218, "URY": 858, "PRY": 600, "PAK": 586, "BGD": 50, "NPL": 524,
                    "KHM": 116, "PNG": 598, "JOR": 400, "OMN": 512, "KWT": 414, "LBN": 422, "IRQ": 368, "SYR": 760, "YEM": 887, "SDN": 729,
                    "CMR": 120, "CIV": 384, "COG": 178, "AGO": 24, "MWI": 454, "MLI": 466, "BFA": 854, "NER": 562, "TCD": 148, "MRT": 478,
                    "ARM": 51, "AZE": 31, "GEO": 268, "MDA": 498, "SRB": 688, "MKD": 807, "ALB": 8, "MNE": 499, "ISL": 352, "CUB": 192,
                    "DOM": 214, "HTI": 332, "TTO": 780, "NIC": 558, "SLV": 222, "CRI": 188, "PAN": 591, "BRN": 96, "FJI": 242, "MNG": 496}


def ym(s, order="mdy"):
    """'2023-08-01' / '08/01/2023' / '2023' / '202308' -> 'YYYY-MM' or ''.  Slash dates are read as month/day/year
    unless order == 'dmy' (the OECD inventory writes DD/MM/YYYY); a first field above 12 is taken as the day either way."""
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    m = re.match(r"^(\d{4})-(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.match(r"^(\d{1,2})[/.](\d{1,2})[/.](\d{4})", s)
    if m:
        a1, b1 = int(m.group(1)), int(m.group(2))
        month = b1 if (order == "dmy" or a1 > 12) and b1 <= 12 else a1
        return f"{m.group(3)}-{month:02d}"
    m = re.match(r"^(\d{4})(\d{2})$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.match(r"^(\d{4})$", s)
    if m:
        return f"{m.group(1)}-01"                                           # year only: January, flagged in note
    return ""


def codes_of(s):
    """Any HS codes (2, 4, 6 digits) in a free-text product field."""
    s = str(s or "")
    out = []
    for tok in re.findall(r"\b\d{2,10}\b", s):
        tok = tok.replace(".", "")
        if len(tok) in (2, 4, 6):
            out.append(tok)
        elif len(tok) in (8, 10):
            out.append(tok[:6])
    return sorted(set(out))


def touches(codes, items):
    for c in codes:
        for it in items:
            if it == c or (len(c) < 6 and it.startswith(c)):
                return True
    return False


ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--seed"); ap.add_argument("--gta"); ap.add_argument("--gta-map"); ap.add_argument("--oecd"); ap.add_argument("--oecd-map")
ap.add_argument("--items")
a = ap.parse_args()
items = [r["hs6"].strip() for r in csv.DictReader(open(a.items, encoding="utf-8"))] if a.items else None
rows = []

if a.seed:
    for r in csv.DictReader(open(a.seed, encoding="utf-8")):
        rows.append(dict(source="seed", partner_iso3=r["partner_iso3"], partner_code=r["partner_code"], measure=r["measure"],
                         announced=ym(r["announced"]), implemented=ym(r["implemented"]), hs_codes=r["hs_scope"], scope_level=r["scope_level"],
                         note=f"{r['note']} | verify: {r['source_to_verify']}"))

if a.gta:
    M = {"jurisdiction": "Implementing Jurisdiction", "iso3": "Implementing Jurisdiction ISO3", "type": "Intervention Type", "announced": "Date Announced",
         "implemented": "Date Implemented", "products": "Affected Products", "title": "Title",
         "export_types": ["Export ban", "Export licensing requirement", "Export quota", "Export tax", "Export-related non-tariff measure, nes"]}
    if a.gta_map:
        M.update(json.load(open(a.gta_map)))
    rd = csv.DictReader(open(a.gta, encoding="utf-8-sig"))
    print("GTA columns:", rd.fieldnames)
    n = 0
    for r in rd:
        if M["type"] in r and r[M["type"]] not in M["export_types"]:
            continue
        if M.get("affected") and M.get("affected_match") and M["affected_match"] not in (r.get(M["affected"]) or ""):
            continue                                                    # aimed at other destinations (sanctions, entity lists)
        if M.get("n_products") and M.get("max_products") and str(r.get(M["n_products"]) or "").isdigit() \
                and int(r[M["n_products"]]) > int(M["max_products"]):
            continue                                                    # economy-wide framework (tariff schedule, sanctions package)
        if M.get("exclude_title") and re.search(M["exclude_title"], r.get(M["title"], "") or "", re.I):
            continue
        codes = codes_of(r.get(M["products"], ""))
        if items and not touches(codes, items):
            continue
        isos = [x.strip() for x in re.split(r"[;,]", r.get(M.get("iso3", ""), "") or "") if x.strip()] or [""]
        for iso in isos:                                                # an EU-wide measure: one row per member state
            rows.append(dict(source="GTA", partner_iso3=iso, partner_code=ISO3_TO_COMTRADE.get(iso, ""), measure=r.get(M["title"], ""),
                             announced=ym(r.get(M["announced"]), M.get("date_order", "mdy")), implemented=ym(r.get(M["implemented"]), M.get("date_order", "mdy")),
                             hs_codes=";".join(codes), scope_level="mixed",
                             note=f"jurisdiction={r.get(M['jurisdiction'], '')}; type={r.get(M['type'], '')}"
                                  + (f"; gta_id={r['intervention_id']}" if r.get("intervention_id") else "")))
        n += 1
    print(f"GTA: {n} export measures touching the item list")

NAME_TO_ISO3 = {"china": "CHN", "china (people's republic of)": "CHN", "people's republic of china": "CHN", "japan": "JPN", "united states": "USA",
                "indonesia": "IDN", "democratic republic of the congo": "COD", "congo, dem. rep.": "COD", "zimbabwe": "ZWE", "russia": "RUS",
                "russian federation": "RUS", "india": "IND", "australia": "AUS", "chile": "CHL", "argentina": "ARG", "brazil": "BRA",
                "south africa": "ZAF", "malaysia": "MYS", "viet nam": "VNM", "vietnam": "VNM", "thailand": "THA", "philippines": "PHL",
                "türkiye": "TUR", "turkey": "TUR", "kazakhstan": "KAZ", "mongolia": "MNG", "canada": "CAN", "mexico": "MEX", "peru": "PER",
                "namibia": "NAM", "gabon": "GAB", "guinea": "GIN", "nigeria": "NGA", "egypt": "EGY", "ukraine": "UKR", "korea": "KOR",
                "madagascar": "MDG", "mozambique": "MOZ", "tanzania": "TZA", "rwanda": "RWA", "burundi": "BDI", "zambia": "ZMB", "bolivia": "BOL",
                "morocco": "MAR", "myanmar": "MMR", "lao people's democratic republic": "LAO", "bosnia and herzegovina": "BIH"}

if a.oecd:
    # Default keys follow the OECD QDD export (methodological note, Table 3): cou = country, HS6 code, year, tom = type of
    # measure, dati = date of introduction or change (DD/MM/YYYY), end = ending date.  Override with --oecd-map.
    M = {"country": "cou", "iso3": "", "product": "HS6 code", "measure": "tom", "year": "year", "start": "dati", "end": "end", "date_order": "dmy"}
    if a.oecd_map:
        M.update(json.load(open(a.oecd_map)))
    rd = csv.DictReader(open(a.oecd, encoding="utf-8-sig"))
    print("OECD columns:", rd.fieldnames)
    # direction of the change: only tightening records become events (introduction, re-introduction, increase, extension,
    # revision, and mixed records that contain one of these); eliminations, decreases and 'left unchanged' are dropped
    dir_col = M.get("direction", "dir"); dir_keep = re.compile(M.get("direction_keep", "introduction|increase|extension|revision"), re.I)
    n = 0; n_dir = 0
    for r in rd:
        if dir_col and dir_col in r and not dir_keep.search(str(r.get(dir_col, "") or "")):
            n_dir += 1
            continue
        codes = codes_of(r.get(M["product"], ""))
        if items and not touches(codes, items):
            continue
        iso = (r.get(M.get("iso3", ""), "") or "") if M.get("iso3") else ""
        if not iso:
            iso = NAME_TO_ISO3.get(str(r.get(M["country"], "")).strip().lower(), "")
        start = ym(r.get(M.get("start", ""), ""), M.get("date_order", "dmy")) or ym(r.get(M["year"], ""))
        rows.append(dict(source="OECD", partner_iso3=iso, partner_code=ISO3_TO_COMTRADE.get(iso, ""), measure=r.get(M["measure"], ""),
                         announced="", implemented=start, hs_codes=";".join(codes), scope_level="mixed",
                         note=f"country={r.get(M['country'], '')}; year-only date" if len(str(r.get(M.get('start', ''), '') or '')) <= 4 else f"country={r.get(M['country'], '')}"))
        n += 1
    print(f"OECD: {n} measures touching the item list ({n_dir} records dropped by direction: eliminations, decreases, unchanged)")

with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["source", "partner_iso3", "partner_code", "measure", "announced", "implemented", "hs_codes", "scope_level", "note"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"written {a.out}: {len(rows)} measures ({sum(1 for r in rows if not r['partner_code'])} without a Comtrade partner code — fill by hand)")
