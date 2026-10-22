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
make_sar_global.py — global CyclObs C-Band SAR L2 OVW (Sentinel-1/RCM/RADARSAT-2)
builder for 2024-2026, replacing the west-Pacific-only make_sar.py.

Per downloaded L2 OVW netCDF (one instantaneous scene per file):
  * lat/lon are 1-D coordinates; wind_speed/quality_level/rejection_flag are (time,lat,lon);
  * keep only high-quality retrievals: quality_level == 4 (good) AND rejection_flag == 0;
  * each scene is a single overpass at its time variable (measurementDate as fallback);
  * keep pixels within RADIUS km of a real IBTrACS centre (±3h), all six basins;
  * the file name stem (unique timestamp+orbit) is the overpass id, so build_da_causal
    never splits one scene or merges two scenes.

Output (one pkl per year, columns datetime/lat/lon/wind_speed/source/satellite_id):
    data/observations/sar_global/sar_{year}_typhoon.pkl
Run with the numpy2 env:
    python make_sar_global.py [--years 2024 2025 2026]
"""
import os, glob, argparse, sys
import numpy as np
import pandas as pd
import xarray as xr

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from ib_global import load_tracks, StormMatcher

RAW_ROOT = os.path.join(HERE, "data/observations", "sar_raw")
OUT_DIR = os.path.join(HERE, "data/observations", "sar_global")
RADIUS_KM = 500.0
MAX_DT_H = 3.0
# Native SAR is ~3 km; aggregate to the model grid (0.06 deg ~ 6 km) so one scene is
# not tens of thousands of redundant points. The eye-wall/RMW (typically >=20 km) is
# fully resolved at 6 km; a cell stores the median of the native retrievals it covers.
BIN_DEG = 0.06


def scene_id(path):
    stem = os.path.basename(path).replace(".nc", "")
    # e.g. rcm1-sc50ma-owi-cm-20240529t090534-...-00003-xxxxx_ll_gd  -> keep through orbit
    return stem


def process_file(path, matcher):
    """Return DataFrame of QC-passed, storm-near pixels for one scene, or None."""
    try:
        ds = xr.open_dataset(path)
    except Exception as e:
        print(f"  OPEN-FAIL {os.path.basename(path)}: {repr(e)[:80]}")
        return None
    try:
        lat1 = ds.lat.values.astype(float)
        lon1 = ds.lon.values.astype(float)
        if "time" in ds.variables and ds.time.values.size:
            t = pd.Timestamp(ds.time.values.flatten()[0])
        else:
            t = pd.Timestamp(ds.attrs.get("measurementDate"))
        ws = ds.wind_speed.values[0]
        ql = ds.quality_level.values[0]
        rf = ds.rejection_flag.values[0]
        mission = str(ds.attrs.get("missionName", "SAR"))
    except Exception as e:
        print(f"  VAR-FAIL {os.path.basename(path)}: {repr(e)[:80]}")
        ds.close()
        return None
    ds.close()

    LON, LAT = np.meshgrid(lon1, lat1)
    # high quality mask (good retrieval, no rejection bits set)
    good = np.isfinite(ws) & (ql == 4) & (rf == 0)
    if not good.any():
        return None
    # storm proximity on the QC-good pixels only
    glat, glon = LAT[good], LON[good]
    gws = ws[good]
    keep = matcher.match_scene(glat, glon, t)
    if not keep.any():
        return None
    glat, glon, gws = glat[keep], glon[keep], gws[keep]
    # aggregate native ~3 km retrievals onto the model-resolution grid (bin median)
    blat = np.round(glat / BIN_DEG) * BIN_DEG
    blon = np.round(glon / BIN_DEG) * BIN_DEG
    df = pd.DataFrame({"blat": blat, "blon": blon, "ws": gws})
    agg = df.groupby(["blat", "blon"], sort=False)["ws"].agg(["mean", "max"]).reset_index()
    out = pd.DataFrame({
        "datetime": [t] * len(agg),
        "lat": agg["blat"].astype("float64"),
        "lon": agg["blon"].astype("float64"),
        # wind_speed = areal mean for the footprint operator; wind_max preserves the
        # native eye-wall peak (a 6 km cell can otherwise lower a cat-5 Vmax).
        "wind_speed": agg["mean"].astype("float64"),
        "wind_max": agg["max"].astype("float64"),
        "source": "sar",
        "satellite_id": scene_id(path),
        "mission": mission,
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--radius", type=float, default=RADIUS_KM)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    tracks = load_tracks(2024, use_cache=True)
    matcher = StormMatcher(tracks, max_dt_h=MAX_DT_H, radius_km=args.radius)
    print(f"IBTrACS tracks: {tracks.sid.nunique()} storms, {len(tracks)} fixes")

    for year in args.years:
        files = sorted(glob.glob(os.path.join(RAW_ROOT, str(year), "*.nc")))
        if not files:
            print(f"[year {year}] no raw files under {RAW_ROOT}/{year}")
            continue
        frames, n_scene, n_point = [], 0, 0
        per_mission = {}
        for k, f in enumerate(files):
            df = process_file(f, matcher)
            if df is not None:
                frames.append(df)
                n_scene += 1
                n_point += len(df)
                m = df["mission"].iloc[0]
                per_mission[m] = per_mission.get(m, 0) + 1
            if (k + 1) % 100 == 0:
                print(f"  [{year}] {k+1}/{len(files)} scanned, {n_scene} matched")
        if not frames:
            print(f"[year {year}] NO matched scenes")
            continue
        all_df = pd.concat(frames, ignore_index=True)
        outp = os.path.join(OUT_DIR, f"sar_{year}_typhoon.pkl")
        all_df.to_pickle(outp)
        print(f"[year {year}] files={len(files)} matched_scenes={n_scene} points={n_point}")
        print(f"  missions={per_mission}")
        print(f"  wind {all_df.wind_speed.min():.1f}-{all_df.wind_speed.max():.1f} m/s, "
              f"n>=33 m/s={int((all_df.wind_speed>=33).sum())}  -> {outp}")


if __name__ == "__main__":
    main()
