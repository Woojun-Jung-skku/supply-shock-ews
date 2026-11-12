"""pull_comtrade_multi.py - monthly imports by partner for a list of HS6 items, one or more reporters and a window of
years from the UN Comtrade API.

With a subscription key (environment variable COMTRADE_KEY, or pasted when the script asks for it) the endpoint
data/v1/get is used and several items share one call: one call = one reporter x one year (12 monthly periods) x up
to --batch items. With the default batch of 13, 65 items x 16 years take about 80 calls, well inside the free key's
500 calls a day (100,000 records a call). A response that reaches the record cap is split in two and asked again;
so is a batch the API rejects (HTTP 400/414), down to single items, so one bad code does not hold up the others.
Without a key the keyless preview endpoint is used: since October 2026 it accepts one item and ONE month per call
(500 records, about one call every 2 seconds), so 65 items x 16 years take about 12,500 calls and most of a day.

The run stops by itself when the daily quota is used up (HTTP 403 "Out of call volume quota"), when the key is
rejected (HTTP 401), after --budget seconds or after --max-calls calls. Re-run the same command (for the quota, the
next day) and it continues where it stopped. Finished item-years are recorded in OUT_DIR/done.txt with the same keys
as the first version (reporter|hs6|year), so a pull started with it resumes; failures and API messages go to
OUT_DIR/log.txt. At the end of every run exact duplicate rows, which an interrupted run can leave behind, are removed
from raw_<reporter>.csv.

usage: python pull_comtrade_multi.py ITEMS_CSV OUT_DIR [--reporters 410] [--years 2010-2025] [--budget 1200]
                                     [--batch 13] [--sleep 2] [--max-calls 450]
  ITEMS_CSV   items_hs6.csv (columns hs6, include)
  OUT_DIR     raw_<reporter>.csv per reporter: cmd, period, partner, qty (net weight kg, or quantity), value (USD)
Reporter codes: 410 Korea, 392 Japan, 842 USA, 97 EU (verify the EU aggregate code in the Comtrade partner/reporter list).
"""
import argparse, csv, gzip, http.client, json, os, re, ssl, sys, time, urllib.error, urllib.parse, urllib.request
from collections import deque

ap = argparse.ArgumentParser(description="Monthly imports by partner for many HS6 items from UN Comtrade.")
ap.add_argument("items", help="items_hs6.csv (columns hs6, include)")
ap.add_argument("out", help="output folder (raw_<reporter>.csv, done.txt, log.txt)")
ap.add_argument("--reporters", default="410", help="comma-separated Comtrade reporter codes (default 410, Korea)")
ap.add_argument("--years", default="2010-2025", help="first-last year, e.g. 2010-2025")
ap.add_argument("--budget", type=float, default=1200.0, help="stop starting new calls after this many seconds")
ap.add_argument("--batch", type=int, default=13, help="items per call when a key is used (default 13)")
ap.add_argument("--sleep", type=float, default=2.0, help="pause between calls in seconds (raised on HTTP 429)")
ap.add_argument("--max-calls", type=int, default=450, help="stop after this many calls in one run (free key: 500 a day)")
ap.add_argument("--max-records", type=int, default=100000, help="records per call (free key: 100000)")
ap.add_argument("--timeout", type=float, default=180.0, help="seconds to wait for one response")
ap.add_argument("--threads", type=int, default=1, help="ignored; calls are made one at a time (kept for old commands)")
ap.add_argument("--api-root", default="https://comtradeapi.un.org", help=argparse.SUPPRESS)
a = ap.parse_args()


def clean_key(k):
    return (k or "").strip().strip('"').strip("'").strip()


KEY = clean_key(os.environ.get("COMTRADE_KEY"))
key_from = "COMTRADE_KEY"
if not KEY and sys.stdin is not None and sys.stdin.isatty():
    try:
        KEY = clean_key(input("COMTRADE_KEY is not set. Paste your subscription key and press Enter\n"
                              "(or press Enter alone to use the keyless preview endpoint, 500 records per call): "))
        key_from = "prompt"
    except EOFError:
        KEY = ""
KEYED = bool(KEY)
API = a.api_root.rstrip("/")
BASE = f"{API}/data/v1/get/C/M/HS" if KEYED else f"{API}/public/v1/preview/C/M/HS"
CAP = max(1, a.max_records) if KEYED else 500
BATCH = max(1, a.batch) if KEYED else 1
MONTHLY = not KEYED                                                      # preview endpoint: one period per call
KNOWN_CAPS = {500, 2500, 100000, 250000}          # a response of exactly this size may be truncated: split it
START = time.time()
os.makedirs(a.out, exist_ok=True)

items = []
with open(a.items, encoding="utf-8-sig", newline="") as fi:              # -sig: a CSV saved by Excel starts with a BOM
    for r in csv.DictReader(fi):
        code = (r.get("hs6") or "").strip()
        if code.isdigit() and len(code) < 6:
            code = code.zfill(6)                                         # Excel drops leading zeros
        if code and (r.get("include") or "1").strip() == "1" and code not in items:
            items.append(code)
yrs = [int(x) for x in a.years.split("-")]
y0, y1 = yrs[0], yrs[-1]
reporters = [r.strip() for r in a.reporters.split(",") if r.strip()]
done_path = os.path.join(a.out, "done.txt")
log_path = os.path.join(a.out, "log.txt")
done = set(open(done_path, encoding="utf-8").read().split()) if os.path.exists(done_path) else set()


def months_of(y):
    return [f"{y}{m:02d}" for m in range(1, 13)]


def log(msg):
    with open(log_path, "a", encoding="utf-8") as fl:
        fl.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


writers = {}
for rep in reporters:
    p = os.path.join(a.out, f"raw_{rep}.csv")
    new = not os.path.exists(p) or os.path.getsize(p) == 0
    f = open(p, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["cmd", "period", "partner", "qty", "value"])
        f.flush()
    writers[rep] = (f, w, p)

# ---- work units: (reporter, year, [hs6 codes], [months]); an item-year left half done month by month resumes by month
queue = deque()
n_total = len(reporters) * len(items) * (y1 - y0 + 1)
for rep in reporters:
    for y in range(y0, y1 + 1):
        full, partial = [], []
        for c in items:
            if f"{rep}|{c}|{y}" in done:
                continue
            left = [m for m in months_of(y) if f"{rep}|{c}|{m}" not in done]
            if len(left) == 12:
                full.append(c)
            elif left:
                partial.append((c, left))
            else:                                                       # every month done, year key missing
                with open(done_path, "a", encoding="utf-8") as fd:
                    fd.write(f"{rep}|{c}|{y}\n")
                done.add(f"{rep}|{c}|{y}")
        for i in range(0, len(full), BATCH):
            if MONTHLY:
                for m in months_of(y):
                    queue.append((rep, y, full[i:i + BATCH], [m]))
            else:
                queue.append((rep, y, full[i:i + BATCH], months_of(y)))
        for c, left in partial:
            for m in left:
                queue.append((rep, y, [c], [m]))
if KEYED and queue and len(queue[0][2]) > 1:                           # first call: one item, a quick check
    rep, y, cmds, ms = queue.popleft()
    queue.appendleft((rep, y, cmds[1:], ms))
    queue.appendleft((rep, y, cmds[:1], ms))
n_todo = n_total - sum(1 for rep in reporters for c in items for y in range(y0, y1 + 1) if f"{rep}|{c}|{y}" in done)

print(f"{len(items)} items x {len(reporters)} reporter(s) x {y1 - y0 + 1} years = {n_total} item-years; {n_todo} to do")
if KEYED:
    print(f"key {KEY[:4]}...{KEY[-4:]} (from {key_from}); endpoint {BASE}")
    if not re.fullmatch(r"[0-9a-fA-F]{32}", KEY):
        print("note: the key does not look like a 32-character Comtrade key; trying it anyway")
    print(f"about {len(queue)} calls (up to {BATCH} items per call); the free key allows 500 calls a day")
else:
    print(f"no key: keyless preview endpoint {BASE} (one item and one month per call; about {len(queue)} calls, slow)")
if a.threads != 1:
    print("note: --threads is ignored; calls are made one at a time")


class Stop(Exception):
    pass


st = dict(calls=0, ok=0, consec_fail=0, fails=0, interval=max(0.0, a.sleep), last=0.0,
          ctx=ssl.create_default_context(), certifi=False, send_max=KEYED, quota=False, badkey=False)
bad_codes = set()


def http_get(url):
    """One paced GET; returns (status, body, headers with lower-case names)."""
    if st["calls"] >= a.max_calls:
        raise Stop(f"--max-calls {a.max_calls} reached for this run")
    wait = st["last"] + st["interval"] - time.time()
    if wait > 0:
        time.sleep(wait)
    st["calls"] += 1
    st["last"] = time.time()
    hdrs = {"Accept-Encoding": "gzip"}
    if KEYED:
        hdrs["Ocp-Apim-Subscription-Key"] = KEY
    req = urllib.request.Request(url, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=a.timeout, context=st["ctx"]) as r:
            status, body, h = r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        status, h = e.code, e.headers
        try:
            body = e.read()
        except Exception:
            body = b""
    st["last"] = time.time()
    h = {k.lower(): v for k, v in (h.items() if h else [])}
    if h.get("content-encoding", "").lower() == "gzip" and body:
        body = gzip.decompress(body)
    return status, body, h


def message_of(body):
    text = body.decode("utf-8", "replace") if body else ""
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            for k in ("message", "error", "errorMessage", "Message"):
                if d.get(k):
                    return str(d[k])[:300]
    except ValueError:
        pass
    return " ".join(text.split())[:300]


def retry_after(h, msg, default):
    try:
        return float(h.get("retry-after"))
    except (TypeError, ValueError):
        pass
    m = re.search(r"(\d+)\s*sec", msg or "")
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+):(\d\d):(\d\d)", msg or "")
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    return default


def fetch(rep, cmds, months):
    """Returns ("ok", records) | ("cap", why) | ("bad", why) | ("transient", why); raises Stop."""
    q = {"reporterCode": rep, "period": ",".join(months), "cmdCode": ",".join(cmds), "flowCode": "M",
         "motCode": "0", "customsCode": "C00"}
    if st["send_max"]:
        q["maxRecords"] = str(CAP)
    url = BASE + "?" + urllib.parse.urlencode(q, safe=",")
    transient = rate_waits = 0
    while True:
        try:
            status, body, h = http_get(url)
        except urllib.error.URLError as e:
            if isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
                if not st["certifi"]:
                    try:
                        import certifi
                        st["ctx"] = ssl.create_default_context(cafile=certifi.where())
                        st["certifi"] = True
                        print("    certificate check failed with the system store; trying the certifi bundle")
                        continue
                    except ImportError:
                        pass
                raise Stop("SSL certificate check failed. On macOS run 'Install Certificates.command' in the Python "
                           "folder under Applications, or 'pip install certifi', then re-run.")
            transient += 1
            why = f"network error: {e.reason}"
        except (OSError, http.client.HTTPException, EOFError) as e:
            transient += 1
            why = f"network error: {e!r}"
        else:
            msg = "" if status == 200 else message_of(body)
            if status == 200:
                try:
                    d = json.loads(body.decode("utf-8"))
                except ValueError as e:
                    d = None
                    why = f"unreadable response: {e}"
                if not isinstance(d, dict) or "data" not in d:
                    transient += 1
                    why = why if d is None else f"response without a data field: {str(d)[:200]}"
                elif d["data"] is None and (d.get("error") or d.get("message")):
                    return "bad", f"HTTP 200 with error: {str(d.get('error') or d.get('message'))[:200]}"
                else:
                    data = d["data"] or []                              # null data without an error: no trade
                    cnt = d.get("count")
                    if len(data) >= CAP or len(data) in KNOWN_CAPS or (isinstance(cnt, int) and cnt > len(data)):
                        return "cap", f"{len(data)} records returned (cap {CAP})"
                    return "ok", data
            elif status == 429 and ("quota" in msg.lower() or retry_after(h, msg, 10.0) > 300):
                st["quota"] = True
                raise Stop(f"daily call quota used up (HTTP 429: {msg})")
            elif status == 429 or (status == 403 and "quota" not in msg.lower()
                                   and retry_after(h, "", 1e9) <= 300):
                rate_waits += 1
                if rate_waits > 30:
                    return "transient", f"HTTP {status} rate limit does not clear: {msg}"
                pause = min(retry_after(h, msg, 10.0), 300.0) + 1.0
                st["interval"] = min(max(st["interval"] * 1.5, 2.0), 15.0)
                print(f"    rate limit (HTTP {status}); waiting {pause:.0f} s, pause between calls now "
                      f"{st['interval']:.1f} s")
                time.sleep(pause)
                continue
            elif status == 403:
                st["quota"] = "quota" in msg.lower()
                raise Stop(f"daily call quota used up (HTTP 403: {msg})" if st["quota"] else f"HTTP 403: {msg}")
            elif status == 401:
                st["badkey"] = True
                raise Stop(f"the key was rejected (HTTP 401: {msg})")
            elif status in (400, 404, 413, 414, 422):
                return "bad", f"HTTP {status}: {msg}"
            else:
                transient += 1
                why = f"HTTP {status}: {msg}"
        if transient >= (2 if len(cmds) > 1 else 4):
            return "transient", why
        time.sleep(min(60.0, 5.0 * 2 ** transient))


def rows_of(data):
    return [[row.get("cmdCode"), row.get("period"), row.get("partnerCode"), row.get("netWgt") or row.get("qty"),
             row.get("primaryValue")] for row in data if row.get("partnerCode") not in (None, 0, "0")]


def save(rep, y, cmds, months, rows):
    f, w, _ = writers[rep]
    for row in rows:
        w.writerow(row)
    f.flush()
    os.fsync(f.fileno())
    if len(months) == 12:
        keys = [f"{rep}|{c}|{y}" for c in cmds]
        done.update(f"{rep}|{c}|{m}" for c in cmds for m in months)
    else:
        keys = [f"{rep}|{c}|{m}" for c in cmds for m in months]
        done.update(keys)
        keys += [f"{rep}|{c}|{y}" for c in cmds if all(f"{rep}|{c}|{m}" in done for m in months_of(y))]
    with open(done_path, "a", encoding="utf-8") as fd:
        fd.write("".join(k + "\n" for k in keys))
        fd.flush()
        os.fsync(fd.fileno())
    done.update(keys)


def label(y, cmds, months):
    what = cmds[0] if len(cmds) == 1 else f"{len(cmds)} items {cmds[0]}..{cmds[-1]}"
    return f"{what} {y if len(months) == 12 else months[0]}"


def fail(rep, y, cmds, months, why):
    st["fails"] += 1
    st["consec_fail"] += 1
    for c in cmds:
        log(f"FAIL {rep}|{c}|{y if len(months) == 12 else months[0]} {why}")
    print(f"    FAIL {label(y, cmds, months)}: {why}")


def compact(path):
    """Remove exact duplicate rows (an interrupted run can write a batch twice)."""
    with open(path, newline="", encoding="utf-8") as fr:
        rd = csv.reader(fr)
        header = next(rd, None)
        seen, keep, dup = set(), [], 0
        for row in rd:
            t = tuple(row)
            if t in seen:
                dup += 1
                continue
            seen.add(t)
            keep.append(row)
    if dup:
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as fw:
            ww = csv.writer(fw)
            ww.writerow(header)
            ww.writerows(keep)
        os.replace(tmp, path)
    return dup, len(keep)


stopped = ""
try:
    while queue:
        if time.time() - START > a.budget:
            raise Stop(f"time budget of {a.budget:.0f} s used")
        rep, y, cmds, months = queue.popleft()
        skip = [c for c in cmds if c in bad_codes]
        if skip:
            for c in skip:
                log(f"FAIL {rep}|{c}|{y if len(months) == 12 else months[0]} skipped: rejected earlier in this run")
            st["fails"] += len(skip)
            print(f"    skipped {', '.join(skip)} {y}: the API rejected this code earlier in this run")
            cmds = [c for c in cmds if c not in bad_codes]
            if not cmds:
                continue
        t0 = time.time()
        kind, res = fetch(rep, cmds, months)
        if kind == "ok":
            rows = rows_of(res)
            save(rep, y, cmds, months, rows)
            st["ok"] += 1
            st["consec_fail"] = 0
            print(f"[call {st['calls']:>3}] {rep} {label(y, cmds, months)}: {len(res)} records, {len(rows)} rows "
                  f"kept ({time.time() - t0:.1f} s); {len(queue)} units left")
        elif kind == "bad" and KEYED and st["send_max"] and st["ok"] == 0:
            st["send_max"] = False                                     # first call rejected: try without maxRecords
            log(f"NOTE first call rejected ({res}); retrying without maxRecords")
            print(f"    first call rejected ({res}); retrying without maxRecords")
            queue.appendleft((rep, y, cmds, months))
        elif len(cmds) > 1:
            k = (len(cmds) + 1) // 2
            queue.appendleft((rep, y, cmds[k:], months))
            queue.appendleft((rep, y, cmds[:k], months))
            log(f"SPLIT {rep}|{y} {len(cmds)} items ({res})")
            print(f"    split {label(y, cmds, months)} into {k} + {len(cmds) - k} items: {res}")
        elif kind == "cap" and len(months) > 1:
            for m in reversed(months):
                queue.appendleft((rep, y, cmds, [m]))
            log(f"MONTHLY {rep}|{cmds[0]}|{y} ({res})")
            print(f"    {label(y, cmds, months)}: {res}; pulling it month by month")
        else:
            fail(rep, y, cmds, months, res)
            if kind == "bad":
                bad_codes.add(cmds[0])
            if st["ok"] == 0 and kind == "transient":
                raise Stop("the first call failed (network or server); check the connection and re-run")
            if st["ok"] == 0 and st["consec_fail"] >= 2:
                raise Stop("the first calls failed; see the messages above and log.txt")
            if st["consec_fail"] >= 5:
                raise Stop("5 failures in a row; see log.txt")
except Stop as e:
    stopped = str(e)
except KeyboardInterrupt:
    stopped = "interrupted (Ctrl+C)"
finally:
    for f, _, _ in writers.values():
        f.close()
    for rep, (_, _, p) in writers.items():
        dup, n = compact(p)
        print(f"{p}: {n} rows" + (f" ({dup} duplicate rows removed)" if dup else ""))
    if stopped:
        log(f"STOP {stopped}")

left = sum(1 for rep in reporters for c in items for y in range(y0, y1 + 1) if f"{rep}|{c}|{y}" not in done)
print(f"calls this run: {st['calls']}; failures this run: {st['fails']}" + (f"; stopped: {stopped}" if stopped else ""))
if left == 0:
    print("all item-years done.")
elif st["badkey"]:
    print("Copy the Primary key again from My API Portal (comtradeplus.un.org), set COMTRADE_KEY and run again.")
elif st["quota"]:
    print("The daily quota is used up: run the same command tomorrow and it continues where it stopped.")
else:
    print("Run the same command again to continue" + (" (failed item-years are retried)." if st["fails"] else "."))
if st["fails"] and bad_codes:
    print("Codes the API rejected: " + ", ".join(sorted(bad_codes)) + ". If one keeps failing, its HS6 code may not "
          "exist in Comtrade; set include to 0 for it in the items file.")
print("remaining:", left)
