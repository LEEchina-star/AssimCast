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

# score_external_baselines.py
# Fair comparison of pre-computed global AI weather models against IBTrACS at
# the SAME 6-h lead, SAME best-track fixes, and SAME 150-km Vmax window used for
# AssimCast, but on each model's NATIVE 0.25-deg grid (no fake upsampling).
#
# Data: NOAA AIWP Model Reforecast archive (Radford et al. 2025, BAMS,
#   doi:10.1175/BAMS-D-24-0057.1), public S3 bucket noaa-oar-mlwp-data.
#   Models are initialised from the operational GFS analysis at 00/12 UTC and
#   archived at 6-h steps. We therefore score the strict 6-h-lead subset of our
#   2024 held-out cases whose initialisation tau is in {00,12} UTC.
#
# u10/v10 are read on demand through the kerchunk combined-parquet reference
# (one global 0.25-deg field per zarr chunk), so no multi-GB download is needed.
#
# Vmax diagnostic is copied verbatim from hafs_gate_ibtracs.core_metrics
# (max wind within 150 km of the IBTrACS centre) to guarantee identical metrics;
# match_ib_t1 is copied as well, so this script needs no eccodes/grib stack.
import os, sys, glob, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
IBDIR = os.path.join(HERE, "data", "observations",
                     "ibtracs_v04r01_netcdf_s18421025_e20260827_c20260827")
PRED_DIR = os.path.join(HERE, "data", "hafs_pred_v7_final")
OURS_CSV = os.path.join(HERE, "results", "ibtracs_v7ep12.csv")
KT = 0.514444
CAT3 = 96.0 * KT
R_SEARCH = 150.0
IB0 = datetime(1858, 11, 17)
IB_MAP = {"CP": "EP"}
BUCKET = "https://noaa-oar-mlwp-data.s3.amazonaws.com"

# tag -> (parquet combined reference, display name)
MODELS = {
    "pangu":       ("PANG_v100_GFS", "Pangu-Weather"),
    "graphcast":   ("GRAP_v100_GFS", "GraphCast"),
    "fourcastnet": ("FOUR_v200_GFS", "FourCastNet"),
}
_ibcache = {}
_dscache = {}

# ---------------------------------------------------------------------------
# metrics copied verbatim from hafs_gate_ibtracs.py (identical scoring window)
# ---------------------------------------------------------------------------
def core_vmax(U, V, LAT, LON, clat, clon):
    clon_u = clon + 360.0 if (np.nanmax(LON) > 180.1 and clon < 0) else clon
    X = (LON - clon_u) * 111.32 * np.cos(np.deg2rad(clat))
    Y = (LAT - clat) * 111.32
    R = np.hypot(X, Y)
    WS = np.hypot(U, V)
    m = (R <= R_SEARCH) & np.isfinite(WS)
    return float(np.nanmax(WS[m])) if m.any() else np.nan


def load_ib(basin):
    if basin in _ibcache:
        return _ibcache[basin]
    ds = xr.open_dataset(os.path.join(IBDIR, f"IBTrACS.{basin}.v04r01.nc"),
                         decode_times=False)
    _ibcache[basin] = ds
    return ds


def match_ib_t1(basin, t0, clat, clon, dt_max=3.0, dist_max=300.0):
    try:
        ds = load_ib(basin)
    except Exception:
        return None
    storms = np.where(ds.season.values.astype(float) == t0.year)[0]
    if len(storms) == 0:
        return None
    days = ds.time.values[storms].astype(float)
    lat = ds.usa_lat.values[storms].astype(float)
    lon = ds.usa_lon.values[storms].astype(float)
    wnd = ds.usa_wind.values[storms].astype(float)
    t0d = (t0 - IB0).total_seconds() / 86400.0
    dt_d = np.abs(days - t0d) * 24.0
    dphi = np.radians(lat - clat)
    a = (np.sin(dphi / 2) ** 2 +
         np.cos(np.radians(clat)) * np.cos(np.radians(lat)) *
         np.sin(np.radians(lon - clon) / 2) ** 2)
    dist = 6371.0 * 2 * np.arcsin(np.sqrt(a))
    valid = (dt_d <= dt_max) & (dist <= dist_max) & (wnd > 0) & np.isfinite(wnd)
    if not valid.any():
        return None
    s, t = np.unravel_index(np.argmin(np.where(valid, dist, 1e9)), dist.shape)
    return dict(wind_kt=float(wnd[s, t]), lat=float(lat[s, t]),
                lon=float(lon[s, t]), dt_h=float(dt_d[s, t]))


def open_model(tag, ref):
    if tag in _dscache:
        return _dscache[tag]
    import fsspec
    fs = fsspec.filesystem("reference",
                           fo=f"{BUCKET}/parquet/{ref}_combined_all.parq",
                           remote_protocol="s3",
                           remote_options={"anon": True},
                           target_protocol="https")
    ds = xr.open_zarr(fs.get_mapper(""), chunks=None, consolidated=False)
    _dscache[tag] = ds
    return ds


def build_cases():
    """Reconstruct the 2024 held-out case list exactly as score_ibtracs_v5 does."""
    rows = []
    for mp in sorted(glob.glob(os.path.join(PRED_DIR, "meta_*_test.csv"))):
        tag = os.path.basename(mp)[len("meta_"):-len("_test.csv")]
        basin = tag.split("_")[0]
        ib_basin = IB_MAP.get(basin, basin)
        meta = pd.read_csv(mp, keep_default_na=False)
        for i in range(len(meta)):
            mt = meta.iloc[i]
            tau = datetime.strptime(str(mt["tau"]).zfill(10), "%Y%m%d%H")
            if tau.hour not in (0, 12):
                continue  # strict 6-h lead: archive only has 00/12 UTC inits
            t1 = tau + timedelta(hours=6)
            ib = match_ib_t1(ib_basin, t1, float(mt["clat"]), float(mt["clon"]))
            if ib is None:
                continue
            rows.append(dict(basin=basin, ib_basin=ib_basin, sid=str(mt["sid"]),
                             tau=tau.strftime("%Y%m%d%H"), t1=t1.strftime("%Y%m%d%H"),
                             tau_iso=np.datetime64(tau),
                             ib_lat=ib["lat"], ib_lon=ib["lon"],
                             ib_wind_mps=ib["wind_kt"] * KT))
    return pd.DataFrame(rows)


def model_vmax_at(ds, init, lat, lon, lead_idx=1):
    """u10/v10 f006 global field, then 150-km Vmax at IBTrACS centre."""
    it = ds.init_time.values
    init = np.datetime64(init)
    ii = int(np.argmin(np.abs(it - init)))
    if abs(it[ii] - init) > np.timedelta64(1, "h"):
        return np.nan
    u = ds.u10.isel(init_time=ii, time=lead_idx).values
    v = ds.v10.isel(init_time=ii, time=lead_idx).values
    la = ds.latitude.values.astype(float)
    lo = ds.longitude.values.astype(float)
    LAT, LON = np.meshgrid(la, lo, indexing="ij")
    return core_vmax(u, v, LAT, LON, lat, lon)


def bloc(d, y, pred_col, catmask=None):
    mm = np.ones(len(d), bool) if catmask is None else catmask
    yv = d[y].values[mm]; pv = d[pred_col].values[mm]
    ok = np.isfinite(yv) & np.isfinite(pv)
    yv, pv = yv[ok], pv[ok]
    if len(yv) < 5:
        return None
    e = pv - yv
    r = float(np.corrcoef(yv, pv)[0, 1]) if np.std(pv) > 1e-6 else np.nan
    return {"n": int(len(yv)), "bias": float(np.mean(e)),
            "mae": float(np.mean(np.abs(e))), "r": r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat3only", action="store_true",
                    help="score only major-hurricane (cat3+, >=96 kt) cases")
    ap.add_argument("--models", default="pangu,graphcast,fourcastnet")
    args = ap.parse_args()
    sel = [m for m in args.models.split(",") if m in MODELS]

    cases = build_cases()
    print(f"strict 6-h subset: {len(cases)} cases, "
          f"cat3+={int((cases.ib_wind_mps>=CAT3).sum())}", flush=True)
    if args.cat3only:
        cases = cases[cases.ib_wind_mps >= CAT3].reset_index(drop=True)
        print(f"scoring cat3+ only: {len(cases)} cases", flush=True)

    # join our full-model/teacher/ERA5 results from the master per-case CSV
    ours = pd.read_csv(OURS_CSV, keep_default_na=False, dtype={"basin": str})
    ours["key"] = ours["basin"].astype(str) + "|" + ours["sid"].astype(str) + "|" + ours["t1"].astype(str)
    cases["key"] = cases["basin"] + "|" + cases["sid"] + "|" + cases["t1"]
    keep = {"key": "key", "pred_vmax": "assimcast_vmax", "tgt_vmax": "teacher_vmax",
            "adv_vmax": "stageA_vmax", "era_vmax": "era5_vmax",
            "pers_vmax": "pers_vmax"}
    cases = cases.merge(ours[list(keep)].rename(columns=keep), on="key", how="left")

    for tag in sel:
        ref, name = MODELS[tag]
        ds = open_model(tag, ref)
        res = {}
        # group by unique init so one fetched field serves all storms that cycle.
        # Values are keyed by DataFrame index and assigned back by index, because
        # groupby(sort=True) orders groups by time whereas the frame is ordered by
        # basin; appending a positional list would misattribute each field.
        for init, grp in cases.groupby("tau_iso", sort=True):
            init = np.datetime64(init)
            try:
                it = ds.init_time.values
                ii = int(np.argmin(np.abs(it - init)))
                u = ds.u10.isel(init_time=ii, time=1).values
                v = ds.v10.isel(init_time=ii, time=1).values
                la = ds.latitude.values.astype(float)
                lo = ds.longitude.values.astype(float)
                LAT, LON = np.meshgrid(la, lo, indexing="ij")
                for idx, r in grp.iterrows():
                    res[idx] = core_vmax(u, v, LAT, LON, r.ib_lat, r.ib_lon)
            except Exception as e:
                for idx in grp.index:
                    res[idx] = np.nan
                print(f"  {name} init {init} fail: {repr(e)[:100]}", flush=True)
        cases[f"{tag}_vmax"] = pd.Series(res)
        ok = int(np.isfinite(cases[f"{tag}_vmax"]).sum())
        print(f"{name:14s} scored {ok}/{len(cases)}", flush=True)

    out_csv = os.path.join(HERE, "results", "external_baselines_2024.csv")
    cases.to_csv(out_csv, index=False)
    print("wrote", out_csv, flush=True)

    cat3 = cases["ib_wind_mps"].values >= CAT3
    cols = [("era5_vmax", "ERA5 (0.25 deg)"), ("stageA_vmax", "Stage A advection"),
            ("pangu_vmax", "Pangu-Weather"), ("graphcast_vmax", "GraphCast"),
            ("fourcastnet_vmax", "FourCastNet"),
            ("assimcast_vmax", "AssimCast (full)"), ("teacher_vmax", "HAFS teacher")]
    summary = {"n": int(len(cases)), "n_cat3": int(cat3.sum()), "models": {}}
    for subset, mm in [("all", None), ("cat3+", cat3)]:
        print(f"\n===== Vmax bias vs IBTrACS [{subset}] (m/s, positive=over) =====")
        for col, lab in cols:
            if col not in cases:
                continue
            r = bloc(cases, "ib_wind_mps", col, mm)
            if r:
                summary["models"].setdefault(col, {"label": lab})[subset] = r
                print(f"  {lab:22s} n={r['n']:4d}  bias={r['bias']:+6.2f}  "
                      f"MAE={r['mae']:5.2f}  r={r['r']:.3f}")
    with open(os.path.join(HERE, "results", "external_baselines_2024.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nsaved results/external_baselines_2024.json")


if __name__ == "__main__":
    main()
