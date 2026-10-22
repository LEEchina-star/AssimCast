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

# HAFS inner-core resolution Gate vs IBTrACS best track (Vmax / RMW), with ERA5.
# For each downloaded 00Z storm f000:
#   - parse HAFS trak.atcfunix TAU=0 (lat/lon/Vmax[kt]/MSLP/RMW[nm])
#   - match the co-located IBTrACS v04r01 fix (|dt|<=6 h, nearest within 150 km)
#   - score EVERY field at the SAME IBTrACS centre (fair): storm nest native
#     0.02 deg, area-box-mean coarsened 0.04/0.06, and ERA5 0.25 deg
#   - Vmax = max wind within 150 km; RMW = peak of azimuthal-mean tangential
#     wind (hemisphere-aware, filled-eye skipped)
# Writes results/hafs_gate_ibtracs.csv; prints overall + intensity-binned stats.
import os, glob, csv, json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
import xarray as xr
try:
    from eccodes import codes_grib_multi_support_on
    codes_grib_multi_support_on()
    _ECCODES = True
except Exception:
    # GRIB reading requires the native ecCodes library; allow non-GRIB paths
    # (ERA5 multibasin / coarse-only deploy construction) to import without it.
    _ECCODES = False
from datetime import datetime

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
HAFS = os.path.join(HERE, "data", "noaa", "hafs")
IBDIR = os.path.join(HERE, "data", "observations",
                     "ibtracs_v04r01_netcdf_s18421025_e20260827_c20260827")
ERADIR = os.path.join(HERE, "data", "multibasin")
OUT_CSV = os.path.join(HERE, "results", "hafs_gate_ibtracs.csv")
KT, NM = 0.514444, 1.852
R_SEARCH = 150.0
DT_MAX, DIST_MAX = 6.0, 150.0
IB0 = datetime(1858, 11, 17)
LET2IB = {"l": "NA", "e": "EP", "c": "EP", "w": "WP",
          "a": "NI", "b": "NI", "s": "SI", "p": "SP"}
ERA_BASIN = {"NA": "NA", "EP": "EP", "CP": "EP", "WP": "WP",
             "NI": "NI", "SI": "SI", "SP": "SP"}
_ibcache, _eracache = {}, {}


def parse_trak(path):
    if not os.path.exists(path):
        return None
    with open(path, errors="ignore") as fh:
        for line in fh:
            f = [x.strip() for x in line.split(",")]
            if "HFSA" not in f or len(f) < 9:
                continue
            hi = f.index("HFSA")
            try:
                if f[hi + 1] != "000":
                    continue
                la, lo = f[hi + 2], f[hi + 3]

                def ll(s):
                    v = float(s[:-1]) / 10.0
                    return -v if s[-1] in "SW" else v

                lat, lon = ll(la), ll(lo)
                vmax = float(f[hi + 4]); mslp = float(f[hi + 5])
                rmw = None
                if "ARMW" in f:
                    j = f.index("ARMW")
                    try:
                        rmw = float(f[j + 1])
                    except Exception:
                        rmw = None
                return dict(lat=lat, lon=lon, vmax_kt=vmax, mslp=mslp, rmw_nm=rmw)
            except Exception:
                continue
    return None


def core_metrics(U, V, LAT, LON, clat, clon):
    clon_u = clon + 360.0 if (np.nanmax(LON) > 180.1 and clon < 0) else clon
    X = (LON - clon_u) * 111.32 * np.cos(np.deg2rad(clat))
    Y = (LAT - clat) * 111.32
    R = np.hypot(X, Y)
    WS = np.hypot(U, V)
    gx = abs(float(LAT[1, 0] - LAT[0, 0])) * 111.32
    m = (R <= R_SEARCH) & np.isfinite(WS)
    vmax = float(np.nanmax(WS[m]))
    # Axisymmetric tangential-wind peak (standard RMW diagnostic; kept for
    # profiles/visualisation). Quantitative RMW validation uses HAFS trak ARMW.
    sgn = 1.0 if clat >= 0 else -1.0
    az = np.arctan2(X, Y)
    Vt = (-U * np.sin(az) + V * np.cos(az)) * sgn
    gxc = gx
    edges = np.arange(0, 120.0 + gxc, gxc)
    cc = (R <= 120.0) & np.isfinite(Vt)
    idx = np.clip((R[cc] / gxc).astype(int), 0, len(edges) - 2)
    prof = np.full(len(edges) - 1, np.nan)
    vts = Vt[cc]
    for k in range(prof.shape[0]):
        sel = idx == k
        if sel.sum() > 4:
            prof[k] = np.nanmean(vts[sel])
    rr = 0.5 * (edges[:-1] + edges[1:])
    w = max(3, int(round(15.0 / gxc)))
    fill = np.nanmean(prof[np.isfinite(prof)]) if np.isfinite(prof).any() else 0.0
    prof_s = (np.convolve(np.nan_to_num(prof, nan=fill), np.ones(w) / w, mode="same")
              if len(prof) >= w else prof)
    rmin = max(10.0, 2.0 * gxc)
    valid = (rr >= rmin) & (rr <= 120) & np.isfinite(prof_s)
    rmw = float(rr[valid][np.argmax(prof_s[valid])]) if valid.sum() > 2 else np.nan
    return vmax, rmw


def load_ib(basin):
    if basin in _ibcache:
        return _ibcache[basin]
    ds = xr.open_dataset(os.path.join(IBDIR, f"IBTrACS.{basin}.v04r01.nc"),
                         decode_times=False)
    _ibcache[basin] = ds
    return ds


def match_ib(basin, t0, clat, clon):
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
    rmw = (ds.usa_rmw.values[storms].astype(float)
           if "usa_rmw" in ds else np.full_like(wnd, np.nan))
    sid = ds.sid.values[storms]; name = ds.name.values[storms]
    t0d = (t0 - IB0).total_seconds() / 86400.0
    dt_d = np.abs(days - t0d) * 24.0
    dphi = np.radians(lat - clat)
    a = (np.sin(dphi / 2) ** 2 +
         np.cos(np.radians(clat)) * np.cos(np.radians(lat)) *
         np.sin(np.radians(lon - clon) / 2) ** 2)
    dist = 6371.0 * 2 * np.arcsin(np.sqrt(a))
    valid = (dt_d <= DT_MAX) & (dist <= DIST_MAX) & (wnd > 0) & np.isfinite(wnd)
    if not valid.any():
        return None
    s, t = np.unravel_index(np.argmin(np.where(valid, dist, 1e9)), dist.shape)

    def dec(x):
        try:
            a = np.asarray(x)
            if a.dtype.kind == "S":
                return b"".join(a.flatten()).decode("utf-8", "ignore").strip()
            if a.dtype.kind == "U":
                return "".join(a.flatten()).strip()
        except Exception:
            pass
        return ""

    return dict(wind_kt=float(wnd[s, t]),
                rmw_km=(float(rmw[s, t]) * NM if np.isfinite(rmw[s, t]) and rmw[s, t] > 0 else np.nan),
                lat=float(lat[s, t]), lon=float(lon[s, t]),
                dt_h=float(dt_d[s, t]), dist_km=float(dist[s, t]),
                sid=dec(sid[s]), name=dec(name[s]))


def match_ib_t1(basin, t0, clat, clon, dt_max=3.0, dist_max=300.0):
    """Nearest-in-TIME best-track fix at exactly t0 (a 6-h analysis time).

    Unlike match_ib (which minimises distance over a +-6 h window and therefore snaps a
    t1=tau+6 query back to the tau fix, whose distance to the anchor is zero), this first
    restricts to fixes within dt_max h (the t1 synoptic fix has dt=0), then picks the
    nearest storm. dist_max is widened because the t1 centre legitimately moved ~100-200 km
    from the tau anchor. Returns the true t1 centre/intensity, required for valid-time
    verification of a 6-h forecast."""
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
    rmw = (ds.usa_rmw.values[storms].astype(float)
           if "usa_rmw" in ds else np.full_like(wnd, np.nan))
    sid = ds.sid.values[storms]; name = ds.name.values[storms]
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

    def dec(x):
        try:
            a = np.asarray(x)
            if a.dtype.kind == "S":
                return b"".join(a.flatten()).decode("utf-8", "ignore").strip()
            if a.dtype.kind == "U":
                return "".join(a.flatten()).strip()
        except Exception:
            pass
        return ""

    return dict(wind_kt=float(wnd[s, t]),
                rmw_km=(float(rmw[s, t]) * NM if np.isfinite(rmw[s, t]) and rmw[s, t] > 0 else np.nan),
                lat=float(lat[s, t]), lon=float(lon[s, t]),
                dt_h=float(dt_d[s, t]), dist_km=float(dist[s, t]),
                sid=dec(sid[s]), name=dec(name[s]))



def load_era(basin):
    if basin in _eracache:
        return _eracache[basin]
    eb = ERA_BASIN.get(basin, basin)
    z = np.load(os.path.join(ERADIR, f"era5_wind_{eb}_2024.npz"), allow_pickle=True)
    _eracache[basin] = (z["data"], z["time"], z["lat"].astype(float), z["lon"].astype(float))
    return _eracache[basin]


def era5_vmax(basin, t0, clat, clon):
    try:
        data, t, la, lo = load_era(basin)
    except Exception:
        return np.nan
    tt = np.asarray(t, dtype="datetime64[ns]")
    tq = np.datetime64(t0)
    j = int(np.argmin(np.abs(tt - tq)))
    if abs(int((tt[j] - tq) / np.timedelta64(1, "s"))) > 3 * 3600:
        return np.nan
    u = data[j, 0].astype(np.float32); v = data[j, 1].astype(np.float32)
    clon_u = clon + 360.0 if (lo[-1] > 180.1 and clon < 0) else clon
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    X = (LO - clon_u) * 111.32 * np.cos(np.deg2rad(clat))
    Y = (LA - clat) * 111.32
    R = np.hypot(X, Y)
    ws = np.hypot(u, v)
    mm = (R <= R_SEARCH) & np.isfinite(ws)
    return float(np.nanmax(ws[mm])) if mm.any() else np.nan


def main():
    rows = []
    gribs = sorted(glob.glob(os.path.join(HAFS, "*", "*", "*", "uv_f000.grib2")))
    gribs = [g for g in gribs if os.path.basename(os.path.dirname(g))[-2:] == "00"]
    print(f"[GATE] {len(gribs)} 00Z storm f000 files", flush=True)
    for ni, g in enumerate(gribs):
        ddir = os.path.dirname(g)
        init = os.path.basename(ddir)
        sid = os.path.basename(os.path.dirname(ddir))
        basin_dir = os.path.basename(os.path.dirname(os.path.dirname(ddir)))
        ibb = LET2IB.get(sid[-1])
        if ibb is None:
            continue
        t0 = datetime.strptime(init, "%Y%m%d%H")
        trak = parse_trak(os.path.join(ddir, "trak.atcfunix"))
        if trak is None:
            continue
        ib = match_ib(ibb, t0, trak["lat"], trak["lon"])
        cla = ib["lat"] if ib else trak["lat"]
        clo = ib["lon"] if ib else trak["lon"]
        try:
            ds = xr.open_dataset(g, engine="cfgrib",
                                 backend_kwargs={"indexpath": ""}, errors="ignore").sortby(
                ["latitude", "longitude"])
            un = [v for v in ds.data_vars if v.lower().startswith("u")][0]
            vn = [v for v in ds.data_vars if v.lower().startswith("v")][0]
            du, dv = ds[un], ds[vn]
            lat = du.latitude.values.astype(float); lon = du.longitude.values.astype(float)
            LAT, LON = np.meshgrid(lat, lon, indexing="ij")
            v_n, rmw_n = core_metrics(du.values, dv.values, LAT, LON, cla, clo)
            u4 = du.coarsen(latitude=2, longitude=2, boundary="trim").mean()
            v4 = dv.coarsen(latitude=2, longitude=2, boundary="trim").mean()
            LA4, LO4 = np.meshgrid(u4.latitude.values, u4.longitude.values, indexing="ij")
            v_4, rmw_4 = core_metrics(u4.values, v4.values, LA4, LO4, cla, clo)
            u6 = du.coarsen(latitude=3, longitude=3, boundary="trim").mean()
            v6 = dv.coarsen(latitude=3, longitude=3, boundary="trim").mean()
            LA6, LO6 = np.meshgrid(u6.latitude.values, u6.longitude.values, indexing="ij")
            v_6, rmw_6 = core_metrics(u6.values, v6.values, LA6, LO6, cla, clo)
        except Exception as e:
            print("  grib fail", basin_dir, sid, init, repr(e)[:80], flush=True)
            continue
        v_era = era5_vmax(basin_dir, t0, cla, clo) if ib else np.nan
        rows.append(dict(basin=basin_dir, sid=sid, init=init, clat=cla, clon=clo,
                         trak_vmax_mps=trak["vmax_kt"] * KT,
                         trak_rmw_km=(trak["rmw_nm"] or np.nan) * NM,
                         nest_vmax_native=v_n, nest_rmw_native=rmw_n,
                         nest_vmax_004=v_4, nest_rmw_004=rmw_4,
                         nest_vmax_006=v_6, nest_rmw_006=rmw_6,
                         era5_vmax=v_era,
                         ib_wind_mps=(ib["wind_kt"] * KT if ib else np.nan),
                         ib_wind_kt=(ib["wind_kt"] if ib else np.nan),
                         ib_rmw_km=(ib["rmw_km"] if ib else np.nan),
                         ib_dt_h=(ib["dt_h"] if ib else np.nan),
                         ib_dist_km=(ib["dist_km"] if ib else np.nan),
                         ib_name=(ib["name"] if ib else "")))
        if (ni + 1) % 50 == 0:
            print(f"  processed {ni+1}/{len(gribs)}", flush=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[GATE] wrote {len(rows)} rows -> {OUT_CSV}")
    summarize(rows)


def stats(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5:
        return None
    e = x - y
    return dict(n=int(len(x)), bias=float(np.mean(e)), mae=float(np.mean(np.abs(e))),
                rmse=float(np.sqrt(np.mean(e ** 2))),
                corr=(float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else np.nan))


def boot(xa, xb, y, B=10000, seed=0):
    m = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(y)
    xa, xb, y = xa[m], xb[m], y[m]
    if len(xa) < 8:
        return None
    da, db = np.abs(xa - y), np.abs(xb - y)
    rng = np.random.default_rng(seed)
    bs = np.empty(B)
    for b in range(B):
        ix = rng.integers(0, len(xa), len(xa))
        bs[b] = da[ix].mean() - db[ix].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(diff=float(da.mean() - db.mean()), lo=float(lo), hi=float(hi),
                sig=bool((lo > 0) or (hi < 0)))


SRCS = [("HAFS trak", "trak_vmax_mps"), ("nest native 2km", "nest_vmax_native"),
        ("nest agg 0.04", "nest_vmax_004"), ("nest agg 0.06", "nest_vmax_006"),
        ("ERA5 0.25deg", "era5_vmax")]


def print_vmax(d, title, ycol="ib_wind_mps"):
    print(f"\n--- {title} (n={len(d)}) ---")
    y = d[ycol].values
    for nm, col in SRCS:
        s = stats(d[col].values, y)
        if s:
            print(f"  {nm:17s} bias={s['bias']:+6.2f} MAE={s['mae']:5.2f} "
                  f"RMSE={s['rmse']:5.2f} corr={s['corr']:.3f} m/s")


def summarize(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    d = df[df["ib_wind_mps"].notna()].copy()
    print(f"\n===== Vmax vs IBTrACS at IB centre, matched {len(d)} fixes "
          f"(median dt {d['ib_dt_h'].median():.2f} h, dist {d['ib_dist_km'].median():.1f} km) =====")
    print_vmax(d, "ALL")

    bins = [(-1, 17.5, "<34 kt tropical"), (17.5, 33, "34-63 kt TS/storm"),
            (33, 49, "64-95 kt cat1-2"), (49, 58, "96-112 kt cat3"),
            (58, 70, "113-136 kt cat4"), (70, 200, ">=137 kt cat5")]
    for lo, hi, lab in bins:
        dd = d[(d["ib_wind_mps"] >= lo) & (d["ib_wind_mps"] < hi)]
        if len(dd) >= 5:
            print_vmax(dd, lab)

    # paired MAE: nest native vs ERA5 and nest-agg06 vs ERA5 (all + major hurricanes)
    for lab, dd in [("ALL", d), (">=50 m/s (cat3+)", d[d["ib_wind_mps"] >= 50])]:
        if len(dd) >= 8:
            for nm, col in [("native", "nest_vmax_native"), ("agg06", "nest_vmax_006")]:
                r = boot(dd[col].values, dd["era5_vmax"].values, dd["ib_wind_mps"].values)
                if r:
                    print(f"  [paired {lab}] MAE({nm})-MAE(ERA5)={r['diff']:+.2f} m/s "
                          f"95%CI[{r['lo']:+.2f},{r['hi']:+.2f}] sig={r['sig']}")

    print("\n===== RMW vs IBTrACS usa_rmw, NA/EP recon basins, hurricanes (usa_wind>=64 kt) =====")
    dr = d[(d["basin"].isin(["NA", "EP", "CP"])) & (d["ib_wind_kt"] >= 64) & d["ib_rmw_km"].notna()]
    yr = dr["ib_rmw_km"].values
    for nm, col in [("HAFS trak", "trak_rmw_km"), ("nest native 2km", "nest_rmw_native"),
                    ("nest agg 0.04", "nest_rmw_004"), ("nest agg 0.06", "nest_rmw_006")]:
        s = stats(dr[col].values, yr)
        if s:
            print(f"  {nm:17s} n={s['n']:3d} bias={s['bias']:+6.2f} MAE={s['mae']:5.2f} "
                  f"RMSE={s['rmse']:5.2f} corr={s['corr']:.3f} km")

    print("\n===== Vmax MAE by basin (nest native vs ERA5, m/s) =====")
    for b in sorted(d["basin"].unique()):
        db = d[d["basin"] == b]
        s1 = stats(db["nest_vmax_native"].values, db["ib_wind_mps"].values)
        s2 = stats(db["era5_vmax"].values, db["ib_wind_mps"].values)
        if s1 and s2:
            print(f"  {b:3s} n={s1['n']:3d}  nest MAE={s1['mae']:5.2f} bias={s1['bias']:+5.2f} | "
                  f"ERA5 MAE={s2['mae']:5.2f} bias={s2['bias']:+6.2f}")


if __name__ == "__main__":
    main()
