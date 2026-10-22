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
make_rss_global.py — global RSS SMAP / AMSR2 TC-wind daily L3 builder for 2024-2026,
replacing the west-Pacific-only make_smap.py / make_amsr2.py that relied on the RSS
NRT storm lists.

Each daily file is a 0.25 deg global grid with two nodes (1=ascending, 2=descending);
per-pixel time is 'minute of day' (SMAP: minute, AMSR2: time). Pixels are matched to
IBTrACS fixes in time (±3 h) and space (within 500 km), with no hard-coded domain and
no NRT list. Native resolution (~25-36 km) is retained; the footprint-aware operator
handles the coarse footprint.

Output columns datetime/lat/lon/wind_speed/source/satellite_id; satellite_id encodes
source+node+date so ascending/descending passes are distinct, and build_da_causal
splits separate orbits with the >2 h time gap.
    python make_rss_global.py --src smap --years 2024 2025 2026
"""
import os, glob, re, argparse, sys
import numpy as np
import pandas as pd
import xarray as xr

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from ib_global import load_tracks, StormMatcher

ROOT = os.path.join(HERE, "data/observations")
RADIUS_KM = 500.0
MAX_DT_H = 3.0
NODE_NAME = {1: "A", 2: "D"}

CFG = {
    "amsr2": dict(dir="amsr2_daily", wind="wind_speed_TC", minute="time",
                  pat=r"amsr2_TC-wind_daily_(\d{4})_(\d{2})_(\d{2})"),
    "smap": dict(dir="smap_daily", wind="wind", minute="minute",
                 pat=r"smap_wind_daily_(\d{4})_(\d{2})_(\d{2})"),
}


def file_date(fname, pat):
    m = re.search(pat, fname)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def process_file(path, src, cfg, matcher, storm_dates):
    date = file_date(os.path.basename(path), cfg["pat"])
    if date is None:
        return None
    # cheap skip: a storm fix within ±1 day
    d0 = pd.Timestamp(date)
    nearby = { (d0 + pd.Timedelta(days=dd)).strftime("%Y-%m-%d") for dd in (-1, 0, 1) }
    if not nearby & storm_dates:
        return None
    try:
        ds = xr.open_dataset(path)
        lat = ds.lat.values.astype(float)
        lon = ds.lon.values.astype(float)
        wind = ds[cfg["wind"]].values            # (lat,lon,node)
        minute = ds[cfg["minute"]].values        # (lat,lon,node) minutes
        nodes = ds.node.values.astype(int)
        ds.close()
    except Exception as e:
        print(f"  OPEN-FAIL {os.path.basename(path)}: {repr(e)[:80]}")
        return None

    frames = []
    for ni, node in enumerate(nodes):
        w = wind[:, :, ni]
        mn = minute[:, :, ni]
        valid = np.isfinite(w) & (w >= 0) & (w < 100) & (mn >= 0) & (mn <= 1440)
        if not valid.any():
            continue
        ii, jj = np.where(valid)
        plat = lat[ii]
        plon = lon[jj]
        pws = w[ii, jj].astype(float)
        # per-pixel datetime (minute of day)
        pt = pd.Timestamp(date) + pd.to_timedelta(mn[ii, jj].astype("int64"), unit="m")
        keep, sid, vk = matcher.match_points(plat, plon, pt)
        if not keep.any():
            continue
        k = np.where(keep)[0]
        frames.append(pd.DataFrame({
            "datetime": pt[k],
            "lat": plat[k].astype("float64"),
            "lon": plon[k].astype("float64"),
            "wind_speed": pws[k].astype("float64"),
            "source": src,
            "satellite_id": f"{src}_{NODE_NAME.get(int(node), int(node))}_{date}",
            "sid": sid[k],
            "node": int(node),
        }))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=["smap", "amsr2", "both"], default="both")
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--radius", type=float, default=RADIUS_KM)
    args = ap.parse_args()

    srcs = ["smap", "amsr2"] if args.src == "both" else [args.src]
    tracks = load_tracks(2024, use_cache=True)
    matcher = StormMatcher(tracks, max_dt_h=MAX_DT_H, radius_km=args.radius)
    storm_dates = set(tracks["time"].dt.strftime("%Y-%m-%d"))
    print(f"IBTrACS tracks: {tracks.sid.nunique()} storms")

    for src in srcs:
        cfg = CFG[src]
        out_dir = os.path.join(ROOT, f"{src}_global")
        os.makedirs(out_dir, exist_ok=True)
        for year in args.years:
            files = sorted(glob.glob(os.path.join(ROOT, cfg["dir"], str(year), "*.nc")))
            if not files:
                print(f"[{src} {year}] no files")
                continue
            frames = []
            for k, f in enumerate(files):
                df = process_file(f, src, cfg, matcher, storm_dates)
                if df is not None:
                    frames.append(df)
                if (k + 1) % 50 == 0:
                    print(f"  [{src} {year}] {k+1}/{len(files)} scanned, {len(frames)} days matched")
            if not frames:
                print(f"[{src} {year}] NO matched overpasses")
                continue
            all_df = pd.concat(frames, ignore_index=True)
            outp = os.path.join(out_dir, f"{src}_{year}_typhoon.pkl")
            all_df.to_pickle(outp)
            print(f"[{src} {year}] files={len(files)} days={len(frames)} points={len(all_df)}")
            print(f"  wind {all_df.wind_speed.min():.1f}-{all_df.wind_speed.max():.1f} m/s, "
                  f"n>=33={int((all_df.wind_speed>=33).sum())}  -> {outp}")


if __name__ == "__main__":
    main()
