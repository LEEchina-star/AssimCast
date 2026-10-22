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

# Build the Z1 HAFS inner-core finetuning dataset (run with np2: cfgrib/xarray, no scipy).
#
# Each sample (tau = anchor, target = tau+6h):
#   fine  (8,128,128) ERA5 0.25 bilinear-upsampled to the tau-anchored FIXED 0.05 deg
#                     grid, frames tau-18,-12,-6,tau (u,v interleaved)  [deployment input]
#   target(2,128,128) HAFS storm nest 0.02 -> same fixed 0.05 grid, area-conservative,
#                     tau+6 analysis (the high-res teacher); edges backfilled with anchor
#   coarse(2,128,128) ERA5 0.25 native 128 window (32 deg) at tau          [environment]
#   mask  (1,128,128) 1 = native HAFS target, 0 = backfilled with coarse anchor
# Stored float16 as (N,13,128,128): channels [fine*8, target*2, coarse*2, mask*1].
# The vortex is anchored on the tau centre (IBTrACS best-track preferred, HAFS trak
# fallback) and DRIFTS inside the fixed frame - identical to the ERA5 pretraining set.
import os, sys, glob, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
try:
    from eccodes import codes_grib_multi_support_on
    codes_grib_multi_support_on()
except Exception:
    pass

ROOT = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
HAFS = os.path.join(ROOT, "data/noaa/hafs")
ERADIR = os.path.join(ROOT, "data/multibasin")
OUT = os.path.join(ROOT, "data/hafs_core")
sys.path.insert(0, ROOT)
from hafs_gate_ibtracs import parse_trak, match_ib, ERA_BASIN

N, DFINE, DCOARSE = 128, 0.05, 0.25
HALF = (N - 1) / 2.0
BASINS = ["NA", "EP", "CP", "WP", "NI", "SI", "SP"]
_era = {}


# ----------------------------- geometry (validated) -----------------------------
def cell_edges(c):
    c = np.asarray(c, float); d = np.diff(c)
    e = np.empty(len(c) + 1)
    e[1:-1] = (c[:-1] + c[1:]) / 2
    e[0] = c[0] - d[0] / 2; e[-1] = c[-1] + d[-1] / 2
    return e


def ov_weights(se, de, coslat=None):
    lo = np.maximum(se[:-1][None, :], de[:-1][:, None])
    hi = np.minimum(se[1:][None, :], de[1:][:, None])
    ov = np.clip(hi - lo, 0.0, None)
    if coslat is not None:
        ov = ov * coslat[None, :]
    return ov


def read_hafs(path):
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""},
                         errors="ignore").sortby(["latitude", "longitude"])
    un = [x for x in ds.data_vars if x.lower().startswith("u")][0]
    vn = [x for x in ds.data_vars if x.lower().startswith("v")][0]
    la = ds.latitude.values.astype(float); lo = ds.longitude.values.astype(float)
    u = ds[un].values.astype(np.float32); v = ds[vn].values.astype(np.float32)
    u[u > 500] = np.nan; v[v > 500] = np.nan
    return u, v, la, lo


def remap_conservative(u, v, sla, slo, clat, clon, size=N, d=DFINE, thr=0.5):
    tla = clat + (np.arange(size) - HALF) * d
    tlo = clon + (np.arange(size) - HALF) * d
    ela, elo = cell_edges(tla), cell_edges(tlo)
    sel_a, sel_o = cell_edges(sla), cell_edges(slo)
    ia = np.where((sel_a[1:] > ela[0]) & (sel_a[:-1] < ela[-1]))[0]
    io = np.where((sel_o[1:] > elo[0]) & (sel_o[:-1] < elo[-1]))[0]
    ia = np.arange(max(ia[0] - 1, 0), min(ia[-1] + 2, len(sla)))
    io = np.arange(max(io[0] - 1, 0), min(io[-1] + 2, len(slo)))
    sla_s, slo_s = sla[ia], slo[io]
    Wa = ov_weights(cell_edges(sla_s), ela, coslat=np.cos(np.deg2rad(sla_s)))
    Wo = ov_weights(cell_edges(slo_s), elo)
    Wa_s, Wo_s = Wa.sum(1, keepdims=True), Wo.sum(1, keepdims=True)
    zu = u[np.ix_(ia, io)].astype(np.float32)
    zv = v[np.ix_(ia, io)].astype(np.float32)
    valid = (np.isfinite(zu) & np.isfinite(zv)).astype(np.float32)
    den = Wa @ valid @ Wo.T
    cov = den / (Wa_s @ Wo_s.T)
    ou = np.full((size, size), np.nan, np.float32)
    ov_ = np.full((size, size), np.nan, np.float32)
    for z, o in ((zu, ou), (zv, ov_)):
        num = Wa @ np.where(valid > 0, z, 0.0) @ Wo.T
        m = cov > thr
        o[m] = num[m] / den[m]
    return ou, ov_, cov, tla, tlo


def load_era(basin, year):
    eb = ERA_BASIN.get(basin, basin)
    key = (eb, year)
    if key not in _era:
        p = os.path.join(ERADIR, f"era5_wind_{eb}_{year}.npz")
        if not os.path.exists(p):
            return None
        z = np.load(p, allow_pickle=True)
        _era[key] = (z["data"], np.asarray(z["time"], "datetime64[ns]"),
                     z["lat"].astype(float), z["lon"].astype(float))
    return _era[key]


def era_at(basin, dt):
    e = load_era(basin, dt.year)
    if e is None:
        return None
    data, t, la, lo = e
    j = int(np.argmin(np.abs(t - np.datetime64(dt))))
    if abs((t[j] - np.datetime64(dt)) / np.timedelta64(1, "h")) > 3.0:
        return None
    u = data[j, 0].astype(np.float32)[::-1]
    v = data[j, 1].astype(np.float32)[::-1]
    return u, v, la[::-1], lo


def bilinear(u, v, sla, slo, tla, tlo):
    def one(z):
        fi = np.clip(np.interp(tla, sla, np.arange(len(sla))), 0, len(sla) - 1.0001)
        fj = np.clip(np.interp(tlo, slo, np.arange(len(slo))), 0, len(slo) - 1.0001)
        i0 = np.floor(fi).astype(int); i1 = np.minimum(i0 + 1, len(sla) - 1)
        j0 = np.floor(fj).astype(int); j1 = np.minimum(j0 + 1, len(slo) - 1)
        wy = (fi - i0)[:, None]; wx = (fj - j0)[None, :]
        return ((1 - wy) * (1 - wx) * z[np.ix_(i0, j0)] + wy * (1 - wx) * z[np.ix_(i1, j0)]
                + (1 - wy) * wx * z[np.ix_(i0, j1)] + wy * wx * z[np.ix_(i1, j1)])
    return one(u), one(v)


def coarse_crop(u, v, la, lo, clat, clon, size=N):
    ci = int(np.argmin(np.abs(la - clat))); cj = int(np.argmin(np.abs(lo - clon))); h = size // 2
    return (u[ci - h:ci + h, cj - h:cj + h], v[ci - h:ci + h, cj - h:cj + h])


# ----------------------------- sample assembly -----------------------------
def build_sample(basin, sid, d_tau, d_t1, stats):
    fmt = "%Y%m%d%H"
    tau = datetime.strptime(d_tau, fmt); t1 = datetime.strptime(d_t1, fmt)
    sdir = os.path.join(HAFS, basin, sid)
    # anchor centre: IBTrACS preferred, HAFS trak fallback
    tr0 = parse_trak(os.path.join(sdir, d_tau, "trak.atcfunix"))
    clat, clon = tr0["lat"], tr0["lon"]
    clon_u = clon + 360.0 if clon < 0 else clon
    csrc = "hafs"
    m = match_ib(basin, tau, clat, clon)
    if m is not None:
        clat, clon = m["lat"], m["lon"]
        clon_u = clon + 360.0 if clon < 0 else clon
        csrc = "ibtracs"
    # target HAFS t+6
    u1n, v1n, la1, lo1 = read_hafs(os.path.join(sdir, d_t1, "uv_f000.grib2"))
    tu, tv, cov, tla, tlo = remap_conservative(u1n, v1n, la1, lo1, clat, clon_u)
    # ERA5 frames tau-18,-12,-6,tau on the fixed fine grid
    fine = []
    anchor_uv = None
    for k in (-18, -12, -6, 0):
        fr = era_at(basin, tau + timedelta(hours=k))
        if fr is None:
            return None, "era5_missing"
        eu, ev, ela, elo = fr
        fu, fv = bilinear(eu, ev, ela, elo, tla, tlo)
        fine += [fu, fv]
        if k == 0:
            anchor_uv = (fu, fv)
            au025, av025, ala, alo = eu, ev, ela, elo
    fine = np.stack(fine, 0)                       # (8,128,128)
    # backfill target edges with the coarse anchor, mask native HAFS cells
    msk = (cov > 0.5).astype(np.float32)[None]
    tu = np.where(msk[0] > 0.5, tu, anchor_uv[0])
    tv = np.where(msk[0] > 0.5, tv, anchor_uv[1])
    target = np.stack([tu, tv], 0).astype(np.float32)
    # coarse 32-deg environment at tau
    cu, cv = coarse_crop(au025, av025, ala, alo, clat, clon_u)
    if cu.shape != (N, N):
        return None, "coarse_oob"
    coarse = np.stack([cu, cv], 0).astype(np.float32)
    arr = np.concatenate([fine, target, coarse, msk], 0).astype(np.float16)  # (13,N,N)
    # normalizer accumulation on native-HAFS target cells (train years only)
    if stats is not None:
        mm = msk[0] > 0.5
        for z, k in ((tu, "u"), (tv, "v")):
            z = z[mm].astype(np.float64)
            stats[k]["n"] += z.size
            stats[k]["s"] += float(z.sum())
            stats[k]["s2"] += float((z * z).sum())
    tr1 = parse_trak(os.path.join(sdir, d_t1, "trak.atcfunix"))
    meta = dict(basin=basin, sid=sid, tau=d_tau, t1=d_t1, clat=clat, clon=clon,
                center_src=csrc, vmax_kt=tr1["vmax_kt"], cov_min=float(cov.min()))
    return arr, meta


def split_for(year, sid):
    if year == 2024:
        return "test"
    h = int(sid[:-1]) if sid[:-1].isdigit() else (hash(sid) % 1000)
    return "val" if (h % 10 == 0) else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basins", default=",".join(BASINS))
    ap.add_argument("--years", default="2023,2024,2025")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sid", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    basins = args.basins.split(","); years = [int(y) for y in args.years.split(",")]
    stats = {c: {"n": 0, "s": 0.0, "s2": 0.0} for c in ("u", "v")}
    all_meta = []
    n_ok = n_skip = 0
    for basin in basins:
        sids = sorted(glob.glob(os.path.join(HAFS, basin, "*")))
        if args.sid:
            sids = [p for p in sids if os.path.basename(p) == args.sid]
        # gather (sid,tau,t1) grouped by year ACROSS all storms in the basin, so each
        # basin-year is written once (no per-storm overwrite)
        year_samples = {}
        for sd in sids:
            sid = os.path.basename(sd)
            inits = sorted(os.path.basename(p) for p in glob.glob(os.path.join(sd, "20*")))
            inits = [x for x in inits if len(x) == 10 and x.isdigit()]
            for d1 in inits:
                t1 = datetime.strptime(d1, "%Y%m%d%H")
                d0 = (t1 - timedelta(hours=6)).strftime("%Y%m%d%H")
                if (d0 in inits and t1.year in years
                        and os.path.exists(os.path.join(sd, d1, "uv_f000.grib2"))
                        and os.path.exists(os.path.join(sd, d0, "trak.atcfunix"))):
                    year_samples.setdefault(t1.year, []).append((sid, d0, d1))
        for yr in sorted(year_samples):
            plist = year_samples[yr]
            if args.limit:
                plist = plist[:args.limit]
            arrs, metas = [], []
            for sid, d0, d1 in plist:
                try:
                    st = stats if yr != 2024 else None
                    arr, meta = build_sample(basin, sid, d0, d1, st)
                    if arr is None:
                        n_skip += 1
                        print(f"  skip {basin}/{sid}/{d1}: {meta}")
                        continue
                    meta["set"] = split_for(yr, sid)
                    arrs.append(arr); metas.append(meta); n_ok += 1
                except Exception as ex:
                    n_skip += 1
                    print(f"  ERR {basin}/{sid}/{d1}: {type(ex).__name__} {ex}")
            if arrs:
                A = np.stack(arrs, 0)
                fp = os.path.join(args.out, f"hafs_core_{basin}_{yr}.npy")
                np.save(fp, A)
                pd.DataFrame(metas).to_csv(
                    os.path.join(args.out, f"meta_{basin}_{yr}.csv"), index=False)
                all_meta += metas
                print(f"{basin} {yr}: {len(arrs)} samples "
                      f"({len(plist) - len(arrs)} skipped) -> {fp} "
                      f"({A.nbytes/1e6:.0f} MB)", flush=True)
        _era.clear()
    pd.DataFrame(all_meta).to_csv(os.path.join(args.out, "meta_all.csv"), index=False)
    norm = {}
    for k in ("u", "v"):
        n, s, s2 = stats[k]["n"], stats[k]["s"], stats[k]["s2"]
        mean = s / max(n, 1); var = max(s2 / max(n, 1) - mean ** 2, 1e-6)
        norm[k] = {"mean": float(mean), "std": float(np.sqrt(var)), "n": int(n)}
    with open(os.path.join(args.out, "normalizer_hafs.json"), "w") as f:
        json.dump(norm, f, indent=2)
    print(f"\nDONE ok={n_ok} skip={n_skip}")
    print("normalizer_hafs:", json.dumps(norm))


if __name__ == "__main__":
    main()
