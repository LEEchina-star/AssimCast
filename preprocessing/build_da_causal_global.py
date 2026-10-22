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
build_da_causal_global.py — global, multi-year version of build_da_causal.py.

Reads the global SAR/SMAP/AMSR2 overpass files ({src}_global_30min.npz) and links each
whole overpass to a v6/v7 test sample (meta_all.csv) under the same causal, leakage-free
rules as the original:
  * overpass rebuilt from satellite id + time gap (no within-swath interleaving);
  * causal when its median time lies in [tau-3h, tau];
  * matched to one sample by highest-wind point within R_MAX deg of the centre;
  * source/overpass id retained for zero-overlap L1/L2 evaluations.

Output: data/da_v6/causal_manifest_global.pkl (does NOT overwrite the old manifest).
Once meta_all is extended to 2025/2026 (GFS deployment samples), the same builder covers
the multi-year validation automatically.
"""
import numpy as np, pandas as pd, os, pickle, glob, argparse

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
COMP = os.path.join(HERE, "data/observations/compat")
OUT = os.path.join(HERE, "data/da_v6")


def load_test_meta(meta_dir):
    """Combined test meta. Prefer a single meta_all.csv; otherwise concatenate the
    per-basin-year meta_{B}_{Y}.csv written by the deploy builder (2025/2026)."""
    all_csv = os.path.join(meta_dir, "meta_all.csv")
    if os.path.exists(all_csv):
        return pd.read_csv(all_csv, keep_default_na=False)
    frames = []
    for f in sorted(glob.glob(os.path.join(meta_dir, "meta_*.csv"))):
        if os.path.basename(f).startswith("centers"):
            continue
        d = pd.read_csv(f, keep_default_na=False)
        if "year" not in d.columns:
            d["year"] = d["tau"].astype(str).str[:4].astype(int)
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no meta_all.csv or meta_*.csv in {meta_dir}")
    m = pd.concat(frames, ignore_index=True)
    # deterministic order: basin, year, tau, so the cumcount row matches dataset rows
    m = m.sort_values(["basin", "year", "tau"]).reset_index(drop=True)
    return m

GAPH = {"sar": 1.0, "smap": 2.0, "amsr2": 2.0}
SRC = {"sar": 0, "smap": 1, "amsr2": 2}
# SAR is binned to the 0.06 deg model grid; footprint sigma = half a cell (~3 km).
FOOT_SIG = {"sar": 0.03, "smap": 0.14, "amsr2": 0.10}
R_MAX = 8.0
CAUSAL_H = 3.0


def lon180(x):
    """Normalize longitude to [-180, 180). Source swaths use mixed conventions
    (SAR -180..180, SMAP/AMSR2 0..360) and deploy meta stores clon mod 360; use
    wrapped longitude everywhere so Atlantic / antimeridian storms match."""
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def wrap_dlon(d):
    """Wrap a longitude difference to [-180, 180)."""
    return (np.asarray(d, dtype=float) + 180.0) % 360.0 - 180.0


def load_overpasses(src):
    z = np.load(os.path.join(COMP, f"{src}_global_30min.npz"), allow_pickle=True)
    df = pd.DataFrame({"time_ns": z["time_ns"].astype("int64"), "lat": z["lat"].astype(float),
                       "lon": lon180(z["lon"].astype(float)), "ws": z["ws"].astype(float),
                       "sat": z["sat"].astype(str)})
    df["time"] = pd.to_datetime(df["time_ns"])
    df = df.sort_values(["sat", "time"]).reset_index(drop=True)
    gap = df["time"].diff().dt.total_seconds() / 3600
    df["oid"] = ((df["sat"] != df["sat"].shift()) | (gap > GAPH[src])).cumsum()
    df["src"] = SRC[src]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", default=os.path.join(HERE, "data/hafs_parent_v6"))
    ap.add_argument("--out", default=os.path.join(OUT, "causal_manifest_global.pkl"))
    args = ap.parse_args()
    m = load_test_meta(args.meta_dir)
    test = m[m.set == "test"].copy().reset_index(drop=True)
    test["tau_dt"] = pd.to_datetime(test["tau"], format="%Y%m%d%H")
    test["year"] = test["tau"].astype(str).str[:4].astype(int)
    test["row"] = test.groupby(["basin", "year"]).cumcount()
    test["hour_key"] = test["tau_dt"].dt.strftime("%Y%m%d%H")
    test["clon180"] = lon180(test["clon"].values)
    by_hour = {k: g for k, g in test.groupby("hour_key")}

    entries = []
    counts = {"sar": 0, "smap": 0, "amsr2": 0}
    for src in ("sar", "smap", "amsr2"):
        npz = os.path.join(COMP, f"{src}_global_30min.npz")
        if not os.path.exists(npz):
            print(f"[skip] {src} global npz not found: {npz}")
            continue
        df = load_overpasses(src)
        print(f"{src}: points={len(df)} overpasses={df.oid.nunique()}")
        for oid, g in df.groupby("oid"):
            tmed = g["time"].median()
            hour = tmed.ceil("6h")
            hkey = hour.strftime("%Y%m%d%H")
            if hkey not in by_hour:
                continue
            if (hour - tmed).total_seconds() > CAUSAL_H * 3600 or (hour - tmed).total_seconds() < 0:
                continue
            cand = by_hour[hkey]
            k = g["ws"].idxmax()
            plat, plon = g.loc[k, "lat"], g.loc[k, "lon"]
            dlat = cand["clat"].values - plat
            dlon = wrap_dlon(cand["clon180"].values - plon) * np.cos(np.radians(plat))
            dist = np.hypot(dlat, dlon)
            j = int(np.argmin(dist))
            if dist[j] > R_MAX:
                continue
            row = cand.iloc[j]
            ddlat = g["lat"].values - row["clat"]
            ddlon = wrap_dlon(g["lon"].values - row["clon180"]) * np.cos(np.radians(row["clat"]))
            inside = np.hypot(ddlat, ddlon) <= 10.0
            gg = g[inside]
            if len(gg) < 5:
                continue
            entries.append(dict(
                basin=row["basin"], sid=row["sid"], year=int(row["year"]),
                row=int(row["row"]), tau=row["tau"], hour_key=row["hour_key"],
                clat=float(row["clat"]), clon=float(row["clon"]),
                vmax_kt=float(row["vmax_kt"]), src=src, overpass=int(oid),
                lat=gg["lat"].values.astype("float32"),
                lon=gg["lon"].values.astype("float32"),
                ws=gg["ws"].values.astype("float32"),
                foot_sig=float(FOOT_SIG[src])))
            counts[src] += 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(entries, fh)
    print(f"\nGLOBAL causal entries: {len(entries)}  {counts}")
    skey = set((e["basin"], e["sid"], e["tau"]) for e in entries)
    cat3 = set((e["basin"], e["sid"], e["tau"]) for e in entries if e["vmax_kt"] >= 64)
    cat4 = set((e["basin"], e["sid"], e["tau"]) for e in entries if e["vmax_kt"] >= 96)
    print(f"distinct samples with >=1 causal overpass: {len(skey)}  (cat3+ {len(cat3)}, cat4+ {len(cat4)})")
    bybasin = {}
    for e in entries:
        bybasin[e["basin"]] = bybasin.get(e["basin"], 0) + 1
    print("entries by basin:", bybasin)
    by_sample = {}
    for e in entries:
        by_sample.setdefault((e["basin"], e["sid"], e["tau"]), set()).add(e["src"])
    multi = {k: v for k, v in by_sample.items() if len(v) >= 2}
    print(f"samples with >=2 sources (L2 cross-source pairs): {len(multi)}")


if __name__ == "__main__":
    main()
