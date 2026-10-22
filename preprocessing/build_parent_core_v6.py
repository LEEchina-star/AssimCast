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

# build_parent_core_v6.py
# Form-1 (operational downscaling) dataset:
#   "ERA5/GFS coarse input learns motion  ->  HAFS high-resolution teacher provides
#    structure super-resolution"
#
# Same fixed 0.06 deg (~6 km), storm-anchored 20 deg, 336x336 geometry as v4, so the
# storm is free to move across the window over the 18 h history and the 6 h target.
# The ONLY structural change vs v4 is in the input frames:
#   * v4 fine[0:8] = HAFS parent f000 analysis (high-res) -> inference needs HAFS.
#   * v6 fine[0:8] = ERA5 0.25 deg bilinear-upsampled to the 0.06 window (COARSE),
#                    i.e. exactly what an operational GFS 0.25 analysis would give.
# The teacher target stays HAFS parent f006 (valid tau+6, pure 6-h dynamics).
#
# Stage A (CentreHead + whole-state advection) therefore moves the COARSE vortex;
# Stage B diffusion learns the motion-compensated coarse->fine increment
#   x0 = f006 - ERA5_adv  (eyewall / RMW / high-wind tail / 0.06 sharpness),
# which is genuine super-resolution rather than v5's fine->fine residual.
#
# Stored float16 (N,15,336,336) (identical layout/indexing to v4 so every downstream
# dataset / centre table / scoring script works unchanged):
#   [0:8]   fine    = ERA5 upsampled frames tau-18,-12,-6,tau (u,v)  (deployment input)
#   [8:10]  target  = HAFS parent f006 (valid tau+6, high-res teacher)
#   [10:12] era_ref = ERA5 tau upsampled (== fine tau frame; kept for v4 layout parity)
#   [12:14] coarse  = ERA5 0.25 native 32-deg wide environment (steering)
#   [14]    mask    = 1 where parent f006 is finite (native high-res teacher cell)
#
# Sample selection is kept identical to v4 (parent f000 history + f006 + trak must exist)
# so the row index aligns 1:1 with centers_trak_v5.csv and meta. Run with np2.
import os, sys, glob, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# package bootstrap places sibling modules on sys.path
from build_hafs_core import (ROOT, HAFS, BASINS, read_hafs, era_at, bilinear,
                             parse_trak, match_ib, split_for)

OUT6 = os.path.join(ROOT, "data/hafs_parent_v6")
FMT = "%Y%m%d%H"
SIZE = 336
D_FINE = 0.06                       # parent native / output resolution
HALF = (SIZE - 1) / 2.0             # 167.5
D_COARSE = 32.0 / (SIZE - 1)        # 32 deg wide environment mapped onto 336 px


def fine_grid(clat, clon_u):
    lat = clat + (np.arange(SIZE) - HALF) * D_FINE
    lon = clon_u + (np.arange(SIZE) - HALF) * D_FINE
    return lat, lon


def coarse_grid(clat, clon_u):
    lat = clat + (np.arange(SIZE) - HALF) * D_COARSE
    lon = clon_u + (np.arange(SIZE) - HALF) * D_COARSE
    return lat, lon


def read_parent(path):
    """parent.atm u10/v10 on a regular ascending 0.06 grid, longitude in 0-360."""
    u, v, la, lo = read_hafs(path)
    lo = lo.copy()
    if lo.min() < 0:
        lo = np.where(lo < 0, lo + 360.0, lo)
    jo = np.argsort(lo)
    return u[:, jo].astype(np.float32), v[:, jo].astype(np.float32), \
        la.astype(np.float32), lo[jo].astype(np.float32)


def sample_regular(u, v, la, lo, tlat, tlon):
    if tlon.min() < lo.min() - 1e-4 or tlon.max() > lo.max() + 1e-4 \
            or tlat.min() < la.min() - 1e-4 or tlat.max() > la.max() + 1e-4:
        return None, None, None
    su, sv = bilinear(u, v, la, lo, tlat, tlon)
    fu, fv = bilinear(np.isfinite(u).astype(np.float32),
                      np.isfinite(v).astype(np.float32), la, lo, tlat, tlon)
    cov = (fu > 0.5) & (fv > 0.5)
    return su.astype(np.float32), sv.astype(np.float32), cov


def build_sample_v6(basin, sid, d_tau, stats):
    tau = datetime.strptime(d_tau, FMT)
    sdir = os.path.join(HAFS, basin, sid)
    # tau centre (IBTrACS preferred, HAFS trak fallback)
    tr0 = parse_trak(os.path.join(sdir, d_tau, "trak.atcfunix"))
    clat, clon = tr0["lat"], tr0["lon"]
    clon_u = clon + 360.0 if clon < 0 else clon
    csrc = "hafs"
    mm = match_ib(basin, tau, clat, clon)
    if mm is not None:
        clat, clon = mm["lat"], mm["lon"]
        clon_u = clon + 360.0 if clon < 0 else clon
        csrc = "ibtracs"

    flat, flon = fine_grid(clat, clon_u)
    clat_g, clon_g = coarse_grid(clat, clon_u)

    # ---- COARSE INPUT: ERA5 frames tau-18..tau upsampled to the 0.06 fine window ----
    era_fine = {}
    au = av = ala = alo = None
    for k in (-18, -12, -6, 0):
        fr = era_at(basin, tau + timedelta(hours=k))
        if fr is None:
            return None, "era5_missing"
        eu, ev, ela, elo = fr
        eu = np.ascontiguousarray(eu); ev = np.ascontiguousarray(ev)
        era_fine[k] = bilinear(eu, ev, ela, elo, flat, flon)
        if k == 0:
            au, av, ala, alo = eu, ev, ela, elo
    eru, erv = era_fine[0]
    # coarse 32-deg wide environment
    cu, cv = bilinear(au, av, ala, alo, clat_g, clon_g)

    # fine history is PURE ERA5 (coarse, operational GFS-equivalent input); no HAFS f000
    fine = np.concatenate([era_fine[k] for k in (-18, -12, -6, 0)], 0)  # (8,336,336)

    # ---- HIGH-RES TEACHER: parent f006 (valid tau+6) ----
    p6 = os.path.join(sdir, d_tau, "parent_uv_f006.grib2")
    if not os.path.exists(p6):
        return None, "no_parent_f006"
    u6, v6, la6, lo6 = read_parent(p6)
    tu, tv, cov6 = sample_regular(u6, v6, la6, lo6, flat, flon)
    if tu is None:
        return None, "target_oob"
    if float(cov6.mean()) < 0.5:
        return None, f"target_cov_{float(cov6.mean()):.2f}"
    tu = np.where(cov6, tu, eru).astype(np.float32)
    tv = np.where(cov6, tv, erv).astype(np.float32)
    target = np.stack([tu, tv], 0)
    era_ref = np.stack([eru, erv], 0).astype(np.float32)
    coarse = np.stack([cu, cv], 0).astype(np.float32)
    msk = cov6.astype(np.float32)[None]

    arr = np.concatenate([fine, target, era_ref, coarse, msk], 0).astype(np.float16)  # 15

    if stats is not None:
        for z, c in ((tu, "u"), (tv, "v")):
            z = z[cov6].astype(np.float64)
            stats[c]["n"] += z.size
            stats[c]["s"] += float(z.sum())
            stats[c]["s2"] += float((z * z).sum())

    meta = dict(basin=basin, sid=sid, tau=d_tau,
                t1=(tau + timedelta(hours=6)).strftime(FMT),
                clat=clat, clon=clon, center_src=csrc, vmax_kt=tr0["vmax_kt"],
                cov_target=float(cov6.mean()), cov_hist=1.0)
    return arr, meta


def _worker(args):
    basin, sid, d0 = args
    try:
        arr, meta = build_sample_v6(basin, sid, d0, None)
        return arr, basin, sid, d0, meta
    except Exception as ex:
        return None, basin, sid, d0, f"{type(ex).__name__} {ex}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basins", default=",".join(BASINS))
    ap.add_argument("--years", default="2023,2024")
    ap.add_argument("--out", default=OUT6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sid", default="")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    basins = args.basins.split(",")
    years = set(int(y) for y in args.years.split(","))

    # gather (basin, valid_year, sid, init) -- identical selection to v4 for row alignment
    tasks = []
    for basin in basins:
        sids = sorted(glob.glob(os.path.join(HAFS, basin, "*")))
        if args.sid:
            sids = [p for p in sids if os.path.basename(p) == args.sid]
        for sd in sids:
            sid = os.path.basename(sd)
            inits = sorted(os.path.basename(p) for p in glob.glob(os.path.join(sd, "20*")))
            inits = [x for x in inits if len(x) == 10 and x.isdigit()]
            for d0 in inits:
                tau = datetime.strptime(d0, FMT)
                t1 = tau + timedelta(hours=6)
                if t1.year not in years:
                    continue
                hist = [d0] + [(tau + timedelta(hours=k)).strftime(FMT)
                               for k in (-6, -12, -18)]
                if not all(os.path.exists(os.path.join(sd, d, "parent_uv_f000.grib2"))
                           for d in hist):
                    continue
                if not os.path.exists(os.path.join(sd, d0, "parent_uv_f006.grib2")):
                    continue
                if not os.path.exists(os.path.join(sd, d0, "trak.atcfunix")):
                    continue
                tasks.append((basin, t1.year, sid, d0))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"[BUILD v6] {len(tasks)} ready samples, workers={args.workers}", flush=True)

    if args.workers > 1:
        import multiprocessing as mp
        t0 = __import__("time").time()
        with mp.get_context("fork").Pool(args.workers) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(
                    _worker, [(b, s, d) for b, y, s, d in tasks], chunksize=4), 1):
                results.append(r)
                if i % 100 == 0:
                    print(f"  built {i}/{len(tasks)} ({__import__('time').time()-t0:.0f}s)",
                          flush=True)
    else:
        results = [_worker((b, s, d)) for b, y, s, d in tasks]

    groups, all_meta = {}, []
    n_ok = n_skip = 0
    for arr, basin, sid, d0, meta in results:
        if arr is None:
            n_skip += 1
            print(f"  skip {basin}/{sid}/{d0}: {meta}", flush=True)
            continue
        yr = (datetime.strptime(d0, FMT) + timedelta(hours=6)).year
        groups.setdefault((basin, yr), []).append((arr, meta))
        n_ok += 1

    for (basin, yr), items in sorted(groups.items()):
        items.sort(key=lambda z: z[1]["tau"])
        A = np.stack([x[0] for x in items], 0)
        metas = []
        for arr, meta in items:
            meta["set"] = split_for(yr, meta["sid"])
            metas.append(meta)
        fp = os.path.join(args.out, f"hafs_parent_v6_{basin}_{yr}.npy")
        np.save(fp, A)
        pd.DataFrame(metas).to_csv(
            os.path.join(args.out, f"meta_{basin}_{yr}.csv"), index=False)
        all_meta += metas
        print(f"{basin} {yr}: {len(items)} -> {fp} ({A.nbytes/1e6:.0f} MB)", flush=True)

    pd.DataFrame(all_meta).to_csv(os.path.join(args.out, "meta_all.csv"), index=False)
    print(f"\nDONE parent-v6 ok={n_ok} skip={n_skip} (inputs=ERA5 coarse, teacher=HAFS f006)")


if __name__ == "__main__":
    main()
