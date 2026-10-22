#!/usr/bin/env python3

import os as _os, sys as _sys
_f = _os.path.abspath(__file__)
_here = _os.path.dirname(_f)
_pkg = _os.path.dirname(_here) if _os.path.basename(_here) in ('preprocessing','scoring','figures') else _here
_os.environ.setdefault('ASSIMCAST_ROOT', _pkg)
for _p in (_pkg, _os.path.join(_pkg,'model'), _os.path.join(_pkg,'preprocessing'),
           _os.path.join(_pkg,'scoring'), _os.path.join(_pkg,'figures')):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)

"""
fetch_hafs.py — Batch download NOAA HAFS-A (hfsa) f000 analysis 10-m winds + ATCF trak
for NA(L)/EP(E)/WP(W) tropical cyclones, via S3 byte-range subsetting.

Only the 10u/10v GRIB2 messages (~0.7 MB) are pulled per (storm,init) instead of the
~0.2-0.8 GB full file. Invest systems (ATCF number >= 90) are skipped.

Storage:
  <out>/<NA|EP|WP>/<atcfid>/<initYYYYMMDDHH>/trak.atcfunix
  <out>/<NA|EP|WP>/<atcfid>/<initYYYYMMDDHH>/uv_f000.grib2
A manifest.csv records every target and status; existing files are skipped (resumable).

Examples:
  python fetch_hafs.py --dates 20241108            # smoke test, one day
  python fetch_hafs.py --start 2024-05-01 --end 2024-12-15 --cycles 00 --workers 6
"""
import argparse, csv, os, re, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

BUCKET = "noaa-nws-hafs-pds.s3.amazonaws.com"
BASE = f"https://{BUCKET}"
# ATCF basin letters: l=NATL e=EPAC c=CPAC w=WPAC a=NIO(Arabian) b=NIO(Bay)
# s=South Indian  p=South Pacific
BASIN_MAP = {"l": "NA", "e": "EP", "c": "CP", "w": "WP",
             "a": "NI", "b": "NI", "s": "SI", "p": "SP"}
KEY_RE = re.compile(r"<Key>([^<]+)</Key>")
NEXT_RE = re.compile(r"<NextContinuationToken>([^<]+)</NextContinuationToken>")
TRUNC_RE = re.compile(r"<IsTruncated>true</IsTruncated>")
STORM_RE = re.compile(r"hfsa/(\d{8})/(\d{2})/(\d{2})([a-z])\.")


def http_get(url, max_retries=3, timeout=40, rng=None):
    last = None
    for k in range(max_retries):
        try:
            req = urllib.request.Request(url)
            if rng:
                req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa
            last = e
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def list_prefix(prefix):
    """Return all keys under a prefix (handles pagination)."""
    keys = []
    token = None
    while True:
        url = f"{BASE}/?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token, safe="")
        body = http_get(url, timeout=40).decode("utf-8", "ignore")
        keys.extend(KEY_RE.findall(body))
        if TRUNC_RE.search(body):
            m = NEXT_RE.search(body)
            if not m:
                break
            token = m.group(1)
        else:
            break
    return keys


def parse_uv_range(idx_text):
    """Find byte span covering 10-m UGRD+VGRD (anl / 0 hour fcst). Offset is the first
    integer field > 1e6 (idx layouts differ between f000 and fNNN)."""
    rows = []
    for ln in idx_text.splitlines():
        if not (("UGRD" in ln or "VGRD" in ln) and "10 m above ground" in ln):
            continue
        if not (":anl:" in ln or "0 hour fcst" in ln):
            continue
        off = None
        for p in ln.split(":"):
            p = p.strip()
            if p.isdigit() and int(p) > 1_000_000:
                off = int(p)
                break
        if off is None:
            continue
        var = "UGRD" if "UGRD" in ln else "VGRD"
        rows.append((off, var, ln))
    rows.sort(key=lambda x: x[0])
    offs = [r[0] for r in rows]
    u = next((i for i, r in enumerate(rows) if r[1] == "UGRD"), None)
    v = next((i for i, r in enumerate(rows) if r[1] == "VGRD"), None)
    if u is None or v is None or v != u + 1:
        return None
    start = rows[u][0]
    # end = one byte before the record following VGRD; need the global next offset
    return [start, offs, u, v]


def full_uv_span(idx_text):
    """Return (start,end) bytes for 10m U then V, using next-record offset from full idx."""
    alloff = []
    for ln in idx_text.splitlines():
        for p in ln.split(":"):
            p = p.strip()
            if p.isdigit() and int(p) > 1_000_000:
                alloff.append(int(p))
                break
    alloff = sorted(set(alloff))
    rows = []
    for ln in idx_text.splitlines():
        if (("UGRD" in ln or "VGRD" in ln) and "10 m above ground" in ln
                and (":anl:" in ln or "0 hour fcst" in ln)):
            off = next((int(p) for p in ln.split(":") if p.strip().isdigit()
                        and int(p) > 1_000_000), None)
            if off:
                rows.append((off, "UGRD" if "UGRD" in ln else "VGRD"))
    rows.sort()
    if len(rows) < 2 or rows[0][1] != "UGRD" or rows[1][1] != "VGRD":
        return None
    start = rows[0][0]
    vend = rows[1][0]
    nxt = [o for o in alloff if o > vend]
    end = (min(nxt) - 1) if nxt else None  # None => till EOF
    return start, end


def daterange(a, b):
    d0 = datetime.strptime(a, "%Y-%m-%d")
    d1 = datetime.strptime(b, "%Y-%m-%d")
    cur = d0
    while cur <= d1:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def discover(date, cycle):
    """Return list of (basin,atcfid,init,trak_key,grb_key,idx_key)."""
    pref = f"hfsa/{date}/{cycle}/"
    try:
        keys = list_prefix(pref)
    except Exception as e:
        print(f"[LIST-FAIL] {pref} {e}", file=sys.stderr)
        return []
    out = {}
    for k in keys:
        m = STORM_RE.search(k)
        if not m:
            continue
        d, cy, num, bl = m.groups()
        if int(num) >= 90 or bl not in BASIN_MAP:  # invest or non-TC letter
            continue
        sid = num + bl
        init = d + cy
        rec = out.setdefault(sid, dict(basin=BASIN_MAP[bl], sid=sid, init=init))
        if k.endswith("trak.atcfunix"):
            rec["trak"] = k
        if re.search(r"storm\.atm\.f000\.grb2$", k):
            rec["grb"] = k
        if re.search(r"storm\.atm\.f000\.grb2\.idx$", k):
            rec["idx"] = k
        if re.search(r"parent\.atm\.f000\.grb2$", k):
            rec["pgrb"] = k
        if re.search(r"parent\.atm\.f000\.grb2\.idx$", k):
            rec["pidx"] = k
    return list(out.values())


def fetch_one(rec, out):
    basin, sid, init = rec["basin"], rec["sid"], rec["init"]
    ddir = os.path.join(out, basin, sid, init)
    os.makedirs(ddir, exist_ok=True)
    trak_p = os.path.join(ddir, "trak.atcfunix")
    uv_p = os.path.join(ddir, "uv_f000.grib2")
    status, msg, nbytes = "ok", "", 0
    try:
        if not (os.path.exists(trak_p) and os.path.getsize(trak_p) > 0):
            if "trak" in rec:
                open(trak_p, "wb").write(http_get(BASE + "/" + rec["trak"]))
            else:
                msg += "no-trak;"
        if not (os.path.exists(uv_p) and os.path.getsize(uv_p) > 1000):
            if "idx" not in rec:
                status, msg = "skip", msg + "no-idx;"
            else:
                idx = http_get(BASE + "/" + rec["idx"]).decode("utf-8", "ignore")
                span = full_uv_span(idx)
                if span is None:
                    status, msg = "skip", msg + "uv-not-parsed;"
                else:
                    start, end = span
                    b = http_get(BASE + "/" + rec["grb"], rng=(start, end if end else start + 2_000_000))
                    if b.count(b"GRIB") < 2:
                        status, msg = "bad", msg + f"grib-count={b.count(b'GRIB')};"
                    else:
                        open(uv_p, "wb").write(b)
                        nbytes = len(b)
        else:
            nbytes = os.path.getsize(uv_p)
    except Exception as e:  # noqa
        status, msg = "fail", msg + repr(e)[:160]
    return dict(basin=basin, sid=sid, init=init,
                trak=os.path.exists(trak_p), uv=os.path.exists(uv_p),
                nbytes=nbytes, status=status, msg=msg)


def fetch_parent(rec, out):
    """Download parent.atm (6km outer-domain) 10-m U/V span -> parent_uv_f000.grib2."""
    basin, sid, init = rec["basin"], rec["sid"], rec["init"]
    ddir = os.path.join(out, basin, sid, init)
    os.makedirs(ddir, exist_ok=True)
    p = os.path.join(ddir, "parent_uv_f000.grib2")
    status, msg, nbytes = "ok", "", 0
    try:
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            nbytes = os.path.getsize(p)
        elif "pidx" not in rec:
            status, msg = "skip", "no-pidx;"
        else:
            idx = http_get(BASE + "/" + rec["pidx"]).decode("utf-8", "ignore")
            span = full_uv_span(idx)
            if span is None:
                status, msg = "skip", "puv-not-parsed;"
            else:
                start, end = span
                b = http_get(BASE + "/" + rec["pgrb"],
                             rng=(start, end if end else start + 8_000_000))
                if b.count(b"GRIB") < 2:
                    status, msg = "bad", msg + f"grib-count={b.count(b'GRIB')};"
                else:
                    open(p, "wb").write(b)
                    nbytes = len(b)
    except Exception as e:  # noqa
        status, msg = "fail", msg + repr(e)[:160]
    return dict(basin=basin, sid=sid, init=init, puv=os.path.exists(p),
                nbytes=nbytes, status=status, msg=msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/noaa/hafs")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--dates", default="", help="comma YYYYMMDD to override range")
    ap.add_argument("--cycles", default="00")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="cap #targets (debug)")
    ap.add_argument("--product", choices=["storm", "parent", "both"], default="storm",
                    help="storm=2km nest uv_f000; parent=6km parent_uv_f000; both")
    ap.add_argument("--parent-cycles", default="00",
                    help="comma cycles for parent when product=both (storm uses --cycles)")
    args = ap.parse_args()

    dates = args.dates.split(",") if args.dates else list(daterange(args.start, args.end))
    cycles = args.cycles.split(",")

    print(f"[DISCOVER] {len(dates)} dates x {cycles} ...", flush=True)
    tasks = [(date, cy) for date in dates for cy in cycles]
    discover_workers = max(args.workers * 3, 24)
    print(f"  discovering {len(tasks)} date-cycles with {discover_workers} threads",
          flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=discover_workers) as ex:
        futs = [ex.submit(discover, d, c) for d, c in tasks]
        for ni, fut in enumerate(as_completed(futs), 1):
            recs.extend(fut.result())
            if ni % 80 == 0 or ni == len(futs):
                print(f"  discovered {ni}/{len(futs)} date-cycles: "
                      f"{len(recs)} storm-inits", flush=True)
    # de-dup (sid,init)
    uniq = {(r["sid"], r["init"]): r for r in recs}
    recs = list(uniq.values())
    if args.limit:
        recs = recs[:args.limit]
    print(f"[DISCOVER] {len(recs)} unique storm-inits across "
          f"{len(set(r['sid'] for r in recs))} storms", flush=True)

    os.makedirs(args.out, exist_ok=True)

    def run_pool(fn, rows, manifest_name, header, row_fmt, tag):
        if not rows:
            print(f"[{tag}] no rows", flush=True)
            return
        path = os.path.join(args.out, manifest_name)
        done = 0
        with open(path, "a", newline="") as f:
            wr = csv.writer(f)
            if f.tell() == 0:
                wr.writerow(header)
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(fn, r, args.out): r for r in rows}
                for fut in as_completed(futs):
                    res = fut.result()
                    wr.writerow(row_fmt(res))
                    f.flush()
                    done += 1
                    if done % 25 == 0 or res["status"] != "ok":
                        print(f"  [{tag} {done}/{len(rows)}] {res['basin']} {res['sid']} "
                              f"{res['init']} {res['status']} {res['msg']}".rstrip(), flush=True)
        print(f"[{tag} DONE] {done} targets -> {path}", flush=True)

    if args.product in ("storm", "both"):
        run_pool(fetch_one, recs, "manifest.csv",
                 ["basin", "sid", "init", "trak", "uv", "nbytes", "status", "msg"],
                 lambda r: [r["basin"], r["sid"], r["init"], r["trak"], r["uv"],
                            r["nbytes"], r["status"], r["msg"]], "STORM")
    if args.product in ("parent", "both"):
        pcyc = set(args.parent_cycles.split(","))
        prows = [r for r in recs if r["init"][-2:] in pcyc]
        print(f"[PARENT] {len(prows)} inits for cycles {sorted(pcyc)}", flush=True)
        run_pool(fetch_parent, prows, "manifest_parent.csv",
                 ["basin", "sid", "init", "puv", "nbytes", "status", "msg"],
                 lambda r: [r["basin"], r["sid"], r["init"], r["puv"],
                            r["nbytes"], r["status"], r["msg"]], "PARENT")
    print(f"[DONE] product={args.product}; out={args.out}", flush=True)


if __name__ == "__main__":
    main()
