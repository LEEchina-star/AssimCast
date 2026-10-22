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

# fetch_hafs_lead.py - download HAFS-A storm-nest 10-m U/V for a forecast LEAD (default
# f006), reusing fetch_hafs primitives. f006 initialised at tau and valid at tau+6 cannot
# have assimilated tau+6 observations, so it provides a forecast (non-circular) teacher
# for the gate against IBTrACS/SFMR. Files land beside f000 as uv_f<lead>.grib2.
import os, sys, re, argparse
from concurrent.futures import ThreadPoolExecutor
import fetch_hafs as fh


def lead_uv_span(idx_text):
    """Byte span of 10-m UGRD then VGRD in a single-lead forecast file (e.g. f006 =
    '6 hour fcst'). Unlike full_uv_span (which targets f000 anl/0-hour), do not filter
    on forecast hour, since each storm.atm.fNNN file carries only one lead."""
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
        if ("UGRD" in ln or "VGRD" in ln) and "10 m above ground" in ln:
            off = next((int(p) for p in ln.split(":") if p.strip().isdigit()
                        and int(p) > 1_000_000), None)
            if off:
                rows.append((off, "UGRD" if "UGRD" in ln else "VGRD"))
    rows.sort()
    for i in range(len(rows) - 1):
        if rows[i][1] == "UGRD" and rows[i + 1][1] == "VGRD":
            vend = rows[i + 1][0]
            nxt = [o for o in alloff if o > vend]
            return rows[i][0], (min(nxt) - 1) if nxt else None
    return None


def discover_lead(date, cycle, lead):
    pref = f"hfsa/{date}/{cycle}/"
    try:
        keys = fh.list_prefix(pref)
    except Exception as e:
        print(f"[LIST-FAIL] {pref} {e}", file=sys.stderr)
        return []
    gre = re.compile(rf"storm\.atm\.f{lead}\.grb2$")
    ire = re.compile(rf"storm\.atm\.f{lead}\.grb2\.idx$")
    out = {}
    for k in keys:
        m = fh.STORM_RE.search(k)
        if not m:
            continue
        d, cy, num, bl = m.groups()
        if int(num) >= 90 or bl not in fh.BASIN_MAP:
            continue
        sid = num + bl
        rec = out.setdefault(sid, dict(basin=fh.BASIN_MAP[bl], sid=sid, init=d + cy))
        if gre.search(k):
            rec["grb"] = k
        if ire.search(k):
            rec["idx"] = k
    return [r for r in out.values() if "grb" in r]


def fetch_lead(rec, out, lead):
    ddir = os.path.join(out, rec["basin"], rec["sid"], rec["init"])
    os.makedirs(ddir, exist_ok=True)
    p = os.path.join(ddir, f"uv_f{lead}.grib2")
    status, msg, nbytes = "ok", "", 0
    try:
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            nbytes = os.path.getsize(p)
            status = "exists"
        elif "idx" not in rec:
            status, msg = "skip", "no-idx;"
        else:
            idx = fh.http_get(fh.BASE + "/" + rec["idx"]).decode("utf-8", "ignore")
            span = lead_uv_span(idx)
            if span is None:
                status, msg = "skip", "uv-not-parsed;"
            else:
                start, end = span
                b = fh.http_get(fh.BASE + "/" + rec["grb"],
                                rng=(start, end if end else start + 2_000_000))
                if b.count(b"GRIB") < 2:
                    status, msg = "bad", f"grib-count={b.count(b'GRIB')};"
                else:
                    open(p, "wb").write(b)
                    nbytes = len(b)
    except Exception as e:  # noqa
        status, msg = "fail", repr(e)[:160]
    return dict(basin=rec["basin"], sid=rec["sid"], init=rec["init"], lead=lead,
                uv=os.path.exists(p), nbytes=nbytes, status=status, msg=msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/noaa/hafs")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--dates", default="")
    ap.add_argument("--cycles", default="00,06,12,18")
    ap.add_argument("--lead", default="006")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dates = args.dates.split(",") if args.dates else list(fh.daterange(args.start, args.end))
    cycles = args.cycles.split(",")
    tasks = [(d, c) for d in dates for c in cycles]
    print(f"[DISCOVER f{args.lead}] {len(tasks)} date-cycles ...", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=max(args.workers * 3, 24)) as ex:
        for rl in ex.map(lambda t: discover_lead(t[0], t[1], args.lead), tasks):
            recs.extend(rl)
    if args.limit:
        recs = recs[:args.limit]
    print(f"[TARGETS] {len(recs)} storm inits with f{args.lead}", flush=True)
    n_ok = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(lambda r: fetch_lead(r, args.out, args.lead), recs)):
            if res["status"] in ("ok", "exists"):
                n_ok += 1
            elif res["status"] == "skip":
                n_skip += 1
            else:
                n_fail += 1
                print(f"  [{res['status']}] {res['basin']}/{res['sid']}/{res['init']} {res['msg']}",
                      file=sys.stderr)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(recs)} ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)
    print(f"[DONE f{args.lead}] ok/exists={n_ok} skip={n_skip} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
