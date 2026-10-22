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

# score_structure_radii.py - IBTrACS wind-radii (R34/R50/R64) structural verification.
#
# Independent "structure" referee for the held-out 2024 season, complementary to Vmax.
# For every matched 6-h forecast we derive, in the four geographic quadrants
# (NE, SE, SW, NW relative to true north), the outermost closed-contour radius at
# which point winds exceed the operational thresholds
#   R34 = 34 kt  = 17.5 m/s,  R50 = 25.7 m/s,  R64 = 32.9 m/s,
# and compare with the IBTrACS usa_r34 / usa_r50 / usa_r64 quadrant radii.
#
# Method: wind speed is sampled along azimuth rays (every 5 degrees) from the
# best-track centre; on each ray we locate the eyewall peak and the first outward
# crossing of the threshold (excluding environmental wind disconnected from the
# vortex). A quadrant radius is the robust maximum (90th percentile) of the ray
# closed-contour radii inside that quadrant. The same IBTrACS centre is used for
# every field (fair); outer radii are insensitive to small centre offsets.
#
# Fields are identical to the main Table-1 run: pred (full), teacher (f006),
# Stage-A advection, persistence and the coarse ERA5 input, all in the same npz.
#
# Outputs:
#   results/structure_radii_2024.csv   (per storm-fix x quadrant)
#   results/structure_radii_2024.json  (bias/MAE/corr + coverage-hit stats)
import os, sys, glob, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from hafs_gate_ibtracs import load_ib, IB0, KT, NM

IB_MAP = {"CP": "EP"}
IMSK = 10
# IBTrACS usa_r* quadrant order is NE, SE, SW, NW (geographic, true north).
QNAMES = ["NE", "SE", "SW", "NW"]
QAZ = [45.0, 135.0, 225.0, 315.0]   # quadrant centre azimuth (deg from true N)
THR = {"R34": 17.5, "R50": 25.7, "R64": 32.9}
RMAX = 400.0
RBIN = 5.0
AZ_STEP = 5.0
RAY_PCT = 90.0
FIELDS = [("pred", 0), ("tgt", 2), ("adv", 4), ("pers", 6), ("era", 8)]


def gridv(clat, clon, size, res):
    cu = clon + 360.0 if clon < 0 else clon
    C = (size - 1) / 2.0
    lat1 = clat + (np.arange(size) - C) * res
    lon1 = cu + (np.arange(size) - C) * res
    return np.meshgrid(lat1, lon1, indexing="ij")


def eye_center(U, V, LAT, LON, cla, clo, r_search=150.0):
    """Storm-relative eye position = the wind-speed maximum within r_search of the
    best-track centre. Structural radii are computed about each field's own eye so
    that track displacement is not mistaken for outer-core structure; track error
    itself is scored separately (track_err_km in the prediction metadata)."""
    clon_u = clo + 360.0 if (np.nanmax(LON) > 180.1 and clo < 0) else clo
    X = (LON - clon_u) * 111.32 * np.cos(np.deg2rad(cla))
    Y = (LAT - cla) * 111.32
    R = np.hypot(X, Y)
    WS = np.hypot(U, V)
    m = (R <= r_search) & np.isfinite(WS)
    if not m.any():
        return cla, clo
    ia, io = np.unravel_index(np.argmax(np.where(m, WS, -np.inf)), WS.shape)
    return float(LAT[ia, io]), float(LON[ia, io])


def ray_profiles(WS, LAT, LON, cla, clo):
    """Sample wind speed along azimuth rays from the centre. Returns (rr, prof)."""
    clon_u = clo + 360.0 if (np.nanmax(LON) > 180.1 and clo < 0) else clo
    lat0 = LAT[:, 0]; lon0 = LON[0, :]
    dlat = lat0[1] - lat0[0]; dlon = lon0[1] - lon0[0]
    n_azi = int(round(360.0 / AZ_STEP))
    rr = np.arange(RBIN, RMAX + RBIN, RBIN)
    prof = np.full((n_azi, len(rr)), np.nan)
    for ai in range(n_azi):
        az = np.deg2rad(ai * AZ_STEP)
        plat = cla + rr * np.cos(az) / 111.32
        plon = clon_u + rr * np.sin(az) / (111.32 * np.cos(np.deg2rad(cla)))
        ilat = np.rint((plat - lat0[0]) / dlat).astype(int)
        ilon = np.rint((plon - lon0[0]) / dlon).astype(int)
        ok = (ilat >= 0) & (ilat < WS.shape[0]) & (ilon >= 0) & (ilon < WS.shape[1])
        vals = np.full(len(rr), np.nan)
        vals[ok] = WS[ilat[ok], ilon[ok]]
        prof[ai] = vals
    return rr, prof


def quadrant_radii(U, V, LAT, LON, cla, clo):
    """dict {R34/R50/R64 -> np.array([rNE,rSE,rSW,rNW])} in km (closed contours)."""
    WS = np.hypot(U, V)
    rr, prof = ray_profiles(WS, LAT, LON, cla, clo)
    n_azi = prof.shape[0]
    half = int(round(45.0 / AZ_STEP))    # rays each side of quadrant centre
    nray_q = 2 * half
    out = {}
    for name, thr in THR.items():
        rad = np.full(4, np.nan)
        for qi in range(4):
            c0 = int(round(QAZ[qi] / AZ_STEP))
            idxs = [(c0 + k) % n_azi for k in range(-half, half)]
            closed = []
            for ai in idxs:
                p = prof[ai]
                if np.isfinite(p).sum() < 10:
                    continue
                p = pd.Series(p).interpolate(limit_direction="both").rolling(
                    3, center=True, min_periods=1).mean().values
                inner = rr <= 150.0
                if not np.isfinite(p[inner]).any():
                    continue
                im = int(np.argmax(np.where(inner, p, -np.inf)))
                if p[im] < thr:
                    continue  # this ray never reaches the threshold -> no closed contour
                cross = np.nan
                for b in range(im + 1, len(rr) - 1):
                    if p[b] < thr and p[b + 1] < thr:
                        cross = rr[b]
                        break
                if np.isfinite(cross):
                    closed.append(cross)
            if len(closed) >= max(3, nray_q // 3):
                rad[qi] = float(np.nanpercentile(closed, RAY_PCT))
        out[name] = rad
    return out


def locate_fix(ds, t1, cla, clo, dt_max=3.0, dist_max=300.0):
    """Reproduce the t1 match and return (storm,time) indices for radius lookup."""
    storms = np.where(ds.season.values.astype(float) == t1.year)[0]
    if len(storms) == 0:
        return None
    days = ds.time.values[storms].astype(float)
    lat = ds.usa_lat.values[storms].astype(float)
    lon = ds.usa_lon.values[storms].astype(float)
    wnd = ds.usa_wind.values[storms].astype(float)
    t0d = (t1 - IB0).total_seconds() / 86400.0
    dt_d = np.abs(days - t0d) * 24.0
    dphi = np.radians(lat - cla)
    a = (np.sin(dphi / 2) ** 2 +
         np.cos(np.radians(cla)) * np.cos(np.radians(lat)) *
         np.sin(np.radians(lon - clo) / 2) ** 2)
    dist = 6371.0 * 2 * np.arcsin(np.sqrt(a))
    valid = (dt_d <= dt_max) & (dist <= dist_max) & (wnd > 0) & np.isfinite(wnd)
    if not valid.any():
        return None
    s, t = np.unravel_index(np.argmin(np.where(valid, dist, 1e9)), dist.shape)
    return int(storms[s]), int(t)


def ib_radii(ds, s, t):
    """Best-track quadrant radii (km) and max wind (kt) at fix (s,t)."""
    res = {}
    for name, var in [("R34", "usa_r34"), ("R50", "usa_r50"), ("R64", "usa_r64")]:
        if var in ds:
            a = ds[var].values[s, t].astype(float)
            a = np.where((a > 0) & np.isfinite(a), a * NM, np.nan)
            res[name] = a
        else:
            res[name] = np.full(4, np.nan)
    w = float(ds.usa_wind.values[s, t])
    return res, w


def stats_pair(obs, pred):
    m = np.isfinite(obs) & np.isfinite(pred)
    n = int(m.sum())
    if n < 5:
        return None
    e = pred[m] - obs[m]
    corr = float(np.corrcoef(obs[m], pred[m])[0, 1]) if obs[m].std() > 1e-6 and pred[m].std() > 1e-6 else np.nan
    return {"n": n, "bias": float(np.mean(e)), "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))), "corr": corr}


def main():
    pred = os.path.join(HERE, "data", "hafs_pred_v7_final")
    res = 0.06
    rows = []
    for fp in sorted(glob.glob(os.path.join(pred, "pred_*_test.npz"))):
        tag = os.path.basename(fp)[len("pred_"):-len("_test.npz")]
        basin = tag.split("_")[0]
        ib_basin = IB_MAP.get(basin, basin)
        mp = os.path.join(pred, f"meta_{tag}_test.csv")
        if not os.path.exists(mp):
            continue
        F = np.load(fp)["f"].astype(float)
        meta = pd.read_csv(mp, keep_default_na=False)
        size = F.shape[-1]
        try:
            ds = load_ib(ib_basin)
        except Exception:
            ds = None
        print(f"{basin}: {F.shape[0]} fields", flush=True)
        for i in range(F.shape[0]):
            mt = meta.iloc[i]
            if "t1" in mt and str(mt["t1"]) not in ("", "nan"):
                t1 = datetime.strptime(str(mt["t1"]), "%Y%m%d%H")
            else:
                t1 = datetime.strptime(str(mt["tau"]).zfill(10), "%Y%m%d%H") + timedelta(hours=6)
            if ds is None:
                continue
            loc = locate_fix(ds, t1, float(mt["clat"]), float(mt["clon"]))
            if loc is None:
                continue
            s, t = loc
            ibq, wkt = ib_radii(ds, s, t)
            if not any(np.isfinite(ibq[k]).any() for k in THR):
                continue
            LAT, LON = gridv(float(mt["clat"]), float(mt["clon"]), size, res)
            mask = F[i][IMSK]
            cla = float(ds.usa_lat.values[s, t]); clo = float(ds.usa_lon.values[s, t])
            frad = {}
            for fld, c in FIELDS:
                U = np.where(mask > 0.5, F[i][c], np.nan)
                V = np.where(mask > 0.5, F[i][c + 1], np.nan)
                # radii about each field's own eye (storm-relative; track scored separately)
                cla_f, clo_f = eye_center(U, V, LAT, LON, cla, clo)
                frad[fld] = quadrant_radii(U, V, LAT, LON, cla_f, clo_f)
            for qi in range(4):
                row = dict(basin=basin, sid=str(mt["sid"]),
                           t1=t1.strftime("%Y%m%d%H"), quad=QNAMES[qi],
                           ib_wind_mps=wkt * KT)
                for k in THR:
                    row[f"ib_{k}"] = ibq[k][qi]
                    for fld, _ in FIELDS:
                        row[f"{fld}_{k}"] = frad[fld][k][qi]
                rows.append(row)
    d = pd.DataFrame(rows)
    if len(d) == 0:
        print("No radius matches.")
        return
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out_csv = os.path.join(HERE, "results", "structure_radii_2024.csv")
    d.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv} ({len(d)} fix-quadrant rows, {len(d)//4} fixes)")

    cat3 = d["ib_wind_mps"].values >= 49.4
    summary = {}
    for k in THR:
        summary[k] = {}
        for grp, mm in [("all", np.ones(len(d), bool)), ("cat3+", cat3)]:
            obs = d[f"ib_{k}"].values[mm]
            block = {"n_fix_quad": int(np.isfinite(obs).sum())}
            for fld, _ in FIELDS:
                stt = stats_pair(d[f"ib_{k}"].values[mm], d[f"{fld}_{k}"].values[mm])
                if stt:
                    both = np.isfinite(obs) & np.isfinite(d[f"{fld}_{k}"].values[mm])
                    stt["hit_rate"] = float(both.sum() / max(np.isfinite(obs).sum(), 1))
                    block[fld] = stt
            summary[k][grp] = block
    with open(os.path.join(HERE, "results", "structure_radii_2024.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n===== Wind-radius verification (km): bias / MAE / corr / hit-rate =====")
    for k in THR:
        for grp in ["all", "cat3+"]:
            blk = summary[k][grp]
            print(f"\n{k} [{grp}]  IB-reporting quadrants={blk['n_fix_quad']}")
            for fld, lab in [("era", "ERA5 coarse"), ("pers", "persistence"),
                             ("adv", "Stage A adv"), ("tgt", "HAFS teacher"),
                             ("pred", "full model")]:
                s = blk.get(fld)
                if s:
                    print(f"  {lab:13s} n={s['n']:4d}  bias={s['bias']:+6.1f}  "
                          f"MAE={s['mae']:5.1f}  corr={s['corr']:.3f}  hit={s['hit_rate']:.2f}")
    print("\nsaved -> results/structure_radii_2024.json")


if __name__ == "__main__":
    main()
