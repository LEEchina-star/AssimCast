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
convert_obs_global.py — merge the global multi-year (2024-2026) SAR/SMAP/AMSR2
observation pickles into the slim npz format consumed by build_da_causal.py
(overpass reconstruction + causal, leakage-free DA manifest).

Writes data/observations/compat/{src}_global_30min.npz (does NOT overwrite the
existing west-Pacific 2024 {src}_30min.npz used by the current v7 run).
    python convert_obs_global.py
"""
import os, glob, argparse
import numpy as np
import pandas as pd

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
OBS = os.path.join(HERE, "data/observations")
OUT = os.path.join(OBS, "compat")
os.makedirs(OUT, exist_ok=True)

# src -> directory holding {src}_{year}_typhoon.pkl
JOBS = {
    "sar": "sar_global",
    "smap": "smap_global",
    "amsr2": "amsr2_global",
}


def convert(src, folder, years=None):
    pkls = sorted(glob.glob(os.path.join(OBS, folder, f"{src}_*_typhoon.pkl")))
    if years is not None:
        pkls = [p for p in pkls if any(f"_{y}_typhoon" in os.path.basename(p) for y in years)]
    if not pkls:
        print(f"{src}: no pickles under {folder} for years={years}")
        return
    frames = []
    for p in pkls:
        x = pd.read_pickle(p)
        x["datetime"] = pd.to_datetime(x["datetime"])
        frames.append(x)
        print(f"  {os.path.basename(p)}: n={len(x)}")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["lat", "lon", "wind_speed"]).copy()
    t = df["datetime"].values.astype("datetime64[ns]").astype("int64")
    sat = df["satellite_id"].astype(str).values
    if "sid" in df.columns:
        track = df["sid"].astype(str).values
    else:
        track = np.array([""] * len(df), dtype=object)
    order = np.argsort(t)
    outp = os.path.join(OUT, f"{src}_global_30min.npz")
    np.savez_compressed(
        outp,
        time_ns=t[order].astype("int64"),
        lat=df["lat"].values[order].astype("float64"),
        lon=df["lon"].values[order].astype("float64"),
        ws=df["wind_speed"].values[order].astype("float32"),
        sat=sat[order].astype(object),
        track=track[order].astype(object))
    n_ov = pd.Series(sat).nunique()
    print(f"{src:6s} total n={len(df):8d} overpasses={n_ov:4d} "
          f"{df['datetime'].min()} -> {df['datetime'].max()}\n        -> {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=None)
    ap.add_argument("--tag", default="global", help="output {src}_{tag}_30min.npz")
    args = ap.parse_args()
    for src, folder in JOBS.items():
        convert(src, folder, args.years)
    print("done ->", OUT)
