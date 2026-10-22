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

# fetch_gfs_10m.py - download GFS 0.25 deg f000 10-m u/v for the 2024 WP cat3+ inits
# (operational deployment surrogate for the Form-1 model). Uses byte-range requests on
# the NOAA AWS open-data bucket (noaa-gfs-bdp-pds), pulling only the two 10-m wind
# records (~2 MB/init) instead of the whole global file (~500 MB).
#
# Output: data/gfs_wp10m/gfs_<YYYYMMDD_HH>.npz with u10,v10 (global 0.25 arrays),
# lat, lon and the centre lat/lon; a second csv holds the sample list.
import os, sys, subprocess, argparse, io
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
OUT = os.path.join(ROOT, "data/gfs_wp10m")
os.makedirs(OUT, exist_ok=True)


def fetch_idx(date, cyc):
    url = f"{BUCKET}/gfs.{date}/{cyc}/atmos/gfs.t{cyc}z.pgrb2.0p25.f000.idx"
    txt = subprocess.run(["curl", "-s", "--max-time", "40", url],
                         capture_output=True, text=True).stdout
    recs = []
    for line in txt.splitlines():
        parts = line.split(":")
        # num:startbyte:d=date:VAR:level:fcst:
        if len(parts) >= 7:
            recs.append((int(parts[0]), int(parts[1]), parts[3], parts[4]))
    return recs


def fetch_range(date, cyc, start, end):
    url = f"{BUCKET}/gfs.{date}/{cyc}/atmos/gfs.t{cyc}z.pgrb2.0p25.f000"
    r = subprocess.run(["curl", "-s", "--max-time", "120", "-r", f"{start}-{end}", url],
                       capture_output=True)
    return r.stdout


def find_ranges(recs):
    """Return (start,end) byte ranges for UGRD and VGRD 10-m above ground."""
    out = {}
    for i, (num, off, var, lev) in enumerate(recs):
        if lev.startswith("10 m above ground") and var in ("UGRD", "VGRD"):
            end = recs[i + 1][1] - 1 if i + 1 < len(recs) else off + 2_000_000
            out[var] = (off, end)
    return out


def read_grib(buf):
    import tempfile, xarray as xr
    tf = tempfile.NamedTemporaryFile(suffix=".grb2", delete=False)
    tf.write(buf); tf.close()
    try:
        ds = xr.open_dataset(tf.name, engine="cfgrib", backend_kwargs={"indexpath": ""})
        ds.load()  # force read before the temp file is removed
    finally:
        os.remove(tf.name)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_storm", type=int, default=3)
    ap.add_argument("--max_inits", type=int, default=30)
    ap.add_argument("--lags", default="0,6,12,18",
                    help="history analysis lags (h) to fetch for each target init")
    args = ap.parse_args()
    lags = [int(x) for x in args.lags.split(",")]
    m = pd.read_csv(os.path.join(ROOT, "data/hafs_parent_v6/meta_all.csv"),
                    keep_default_na=False)
    w = m[(m.basin == "WP") & (m.set == "test") & (m.vmax_kt >= 96)].copy()
    # pick per_storm inits around each storm's peak
    picks = []
    for sid, g in w.groupby("sid"):
        g = g.sort_values("vmax_kt", ascending=False).head(1)  # peak init
        pk = g.iloc[0]
        allg = w[w.sid == sid].sort_values("tau")
        pi = list(allg.tau).index(pk.tau)
        for j in range(max(0, pi - 1), min(len(allg), pi + 2)):
            picks.append(allg.iloc[j])
    picks = picks[:args.max_inits]
    pd.DataFrame(picks).to_csv(os.path.join(OUT, "gfs_inits.csv"), index=False)
    # collect all required analysis times (target tau plus history lags), deduped
    need = {}
    for r in picks:
        t0 = pd.to_datetime(str(r.tau), format="%Y%m%d%H")
        for lag in lags:
            t = t0 - pd.Timedelta(hours=lag)
            need[t.strftime("%Y%m%d%H")] = True
    print(f"[GFS] {len(picks)} target inits, {len(need)} analysis times to fetch")
    for tau in sorted(need):
        date, cyc = tau[:8], tau[8:10]
        fn = os.path.join(OUT, f"gfs_{date}_{cyc}.npz")
        if os.path.exists(fn):
            print("  cached", tau); continue
        try:
            recs = fetch_idx(date, cyc)
            rg = find_ranges(recs)
            if "UGRD" not in rg or "VGRD" not in rg:
                print("  MISSING 10m records", tau); continue
            ub = fetch_range(date, cyc, *rg["UGRD"])
            vb = fetch_range(date, cyc, *rg["VGRD"])
            du = read_grib(ub); dv = read_grib(vb)
            u = np.asarray(du.u10.values, dtype=np.float32)
            v = np.asarray(dv.v10.values, dtype=np.float32)
            lat = np.asarray(du.latitude.values, dtype=np.float32)
            lon = np.asarray(du.longitude.values, dtype=np.float32)
            np.savez_compressed(fn, u=u, v=v, lat=lat, lon=lon)
            print(f"  ok {tau} u{u.shape} range[{u.min():.1f},{u.max():.1f}]", flush=True)
        except Exception as ex:
            print("  FAIL", tau, type(ex).__name__, str(ex)[:120])
    print("GFS fetch done:", len(need), "analysis times")


if __name__ == "__main__":
    main()
