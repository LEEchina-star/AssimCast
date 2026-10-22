#!/usr/bin/env python

import os as _os, sys as _sys
_f = _os.path.abspath(__file__)
_here = _os.path.dirname(_f)
_pkg = _os.path.dirname(_here) if _os.path.basename(_here) in ('preprocessing','scoring','figures') else _here
_os.environ.setdefault('ASSIMCAST_ROOT', _pkg)
for _p in (_pkg, _os.path.join(_pkg,'model'), _os.path.join(_pkg,'preprocessing'),
           _os.path.join(_pkg,'scoring'), _os.path.join(_pkg,'figures')):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)

# fetch_parent_lead.py - TARGETED parent.atm 10-m u/v download for an explicit
# (basin,sid,init) list and a given lead (f000 analysis / f006 6-h forecast), WITHOUT
# listing S3 (keys are constructed from the known storm id and init). Uses byte-range
# subsetting exactly like the storm fetch. Output:
#   <out>/<basin>/<sid>/<init>/parent_uv_f<lead>.grib2
# f000 -> analysis span (anl/0-hour); f006 -> lead_uv_span (6 hour fcst). Resumable.
import os, sys, argparse, time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_hafs as fh
from fetch_hafs_lead import lead_uv_span


def key_for(sid, init, lead):
    return (f"hfsa/{init[:8]}/{init[8:10]}/{sid}.{init}.hfsa."
            f"parent.atm.f{lead}.grb2")


def fetch_one(basin, sid, init, lead, out):
    ddir = os.path.join(out, basin, sid, init)
    os.makedirs(ddir, exist_ok=True)
    p = os.path.join(ddir, f"parent_uv_f{lead}.grib2")
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return dict(basin=basin, sid=sid, init=init, lead=lead,
                    status="exists", nbytes=os.path.getsize(p))
    grb = key_for(sid, init, lead)
    try:
        itxt = fh.http_get(fh.BASE + "/" + grb + ".idx").decode("utf-8", "ignore")
        span = fh.full_uv_span(itxt) if lead == "000" else lead_uv_span(itxt)
        if span is None:
            return dict(basin=basin, sid=sid, init=init, lead=lead,
                        status="skip", nbytes=0, msg="uv-not-parsed")
        start, end = span
        b = fh.http_get(fh.BASE + "/" + grb,
                        rng=(start, end if end else start + 8_000_000))
        if b.count(b"GRIB") < 2:
            return dict(basin=basin, sid=sid, init=init, lead=lead,
                        status="bad", nbytes=len(b), msg=f"grib={b.count(b'GRIB')}")
        with open(p, "wb") as f:
            f.write(b)
        return dict(basin=basin, sid=sid, init=init, lead=lead,
                    status="ok", nbytes=len(b))
    except Exception as e:  # noqa
        return dict(basin=basin, sid=sid, init=init, lead=lead,
                    status="fail", nbytes=0, msg=repr(e)[:140])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/noaa/hafs")
    ap.add_argument("--listfile", required=True, help="csv lines basin,sid,init")
    ap.add_argument("--lead", default="006", choices=["000", "006"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = []
    with open(args.listfile) as f:
        for ln in f:
            parts = ln.strip().split(",")
            if len(parts) == 3:
                rows.append(tuple(parts))
    if args.limit:
        rows = rows[:args.limit]
    print(f"[PARENT f{args.lead}] {len(rows)} targets, {args.workers} workers", flush=True)
    counts = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, b, s, d, args.lead, args.out): (b, s, d)
                for b, s, d in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            if r["status"] in ("fail", "bad", "skip"):
                print(f"  [{r['status']}] {r['basin']}/{r['sid']}/{r['init']} "
                      f"{r.get('msg','')}", flush=True)
            if i % 100 == 0:
                print(f"  {i}/{len(rows)} {counts} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[DONE parent f{args.lead}] {counts} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
