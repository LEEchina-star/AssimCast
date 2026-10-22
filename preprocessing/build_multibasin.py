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
build_multibasin.py — build storm-centred multi-basin training data from global ERA5.

The model is trained on 256x256 pixel-space wind fields, so instead of a global grid we store
one regional domain per ocean basin and crop a 256x256 window centred on the storm at load
time. This gives global applicability at regional compute cost.

Basins (longitudes kept in ERA5's 0-360 convention so every box is a contiguous slice):
    NA  North Atlantic      260-360 E,   0-55 N
    EP  East Pacific        180-290 E,   0-45 N
    WP  West Pacific        100-180 E,  -6-60 N
    NI  North Indian         40-105 E,   0-30 N
    SI  South Indian         20-125 E, -46-0  N
    SP  South Pacific       140-260 E, -48-0  N

Sources of global ERA5 u10/v10 (hourly, 0.25 deg):
    /Volumes/EXTERNAL_USB/ERA5_WIND/<YYYYMM>.nc          (1993-2009)
    /Volumes/EXTERNAL_USB/ERA5_WIND/<YYYY>/<YYYYMM>.nc   (2023-2024)

Outputs per basin-year:
    data/multibasin/era5_wind_<basin>_<year>.npz   data (T,2,H,W) float16, time, lat, lon
    data/multibasin/center_seq_<basin>_<year>.npz  time, center_lat, center_lon,
                                                   center_iy, center_ix, center_wind_ms, lat, lon

Usage:
    python build_multibasin.py --basins NA EP --years 2023 2024     # a quick subset
    python build_multibasin.py --all                                # everything available
"""
import os, glob, argparse, warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, xarray as xr
warnings.filterwarnings('ignore')

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
ERA5 = os.environ.get('ERA5_ROOT', os.path.join(os.environ.get('ASSIMCAST_ROOT',''), 'data/ERA5'))
# Extra staging roots for recent years (2025/2026), searched after the primary root.
ERA5_EXTRA = [os.path.join(HERE, 'data', 'era5_monthly')]
IBT = os.environ.get('IBTRACS_OLD',
                     os.environ.get("IBTRACS_PATH", "IBTrACS.since1980.v04r01.nc"))
OUT = f'{HERE}/data/multibasin'
MPI_WORKERS = 6
RES = 0.25

BASINS = {
    # Boxes are sized from the 0.2-99.8 percentile of the 1993-2024 track extents plus a
    # 16-degree margin, which is what a 128x128 crop (half-width 64 cells = 16 deg) needs so
    # that the storm stays centred throughout the history and target. Storms are assigned to a
    # basin by their best-track label, not geometrically, so the boxes may overlap freely.
    'NA': dict(lon=(234, 360), lat=(-7, 78)),    # North Atlantic   505 x 341
    'EP': dict(lon=(113, 287), lat=(-10, 59)),   # East Pacific     697 x 277
    'WP': dict(lon=(71, 196), lat=(-12, 65)),    # West Pacific     501 x 309
    'NI': dict(lon=(28, 113), lat=(-15, 42)),    # North Indian     341 x 229
    'SI': dict(lon=(17, 158), lat=(-53, 12)),    # South Indian     565 x 261
    'SP': dict(lon=(63, 241), lat=(-57, 10)),    # South Pacific    713 x 269
}
CROP_SIZE = 128          # model input window; the full architecture is unchanged


# Centre source per basin: each ocean basin keeps its own regional official best track, which
# is standard practice and keeps the West Pacific identical to the single-basin model (CMA).
AGENCY = {
    'WP': ['cma', 'tokyo', 'usa'],        # China Meteorological Administration
    'NA': ['usa'],                        # NHC
    'EP': ['usa'],                        # NHC
    'NI': ['newdelhi', 'usa'],            # IMD
    'SI': ['reunion', 'usa'],             # RSMC La Reunion
    'SP': ['usa', 'nadi', 'bom'],         # JTWC, with Nadi/BoM fallback
}


def era5_path(year, month):
    ym = f'{year}{month:02d}'
    candidates = [
        f'{ERA5}/{year}/{ym}.nc',
        f'{ERA5}/{ym}.nc',
    ]
    for root in ERA5_EXTRA:
        candidates.append(f'{root}/{year}/{ym}.nc')
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def available_years():
    """Probe years explicitly: os.listdir/scandir raise EINVAL on the NTFS volume, while
    os.path.exists and open() work normally, so we test each candidate year directly."""
    ys = []
    for y in range(1993, 2027):
        if any(os.path.exists(era5_path(y, m)) for m in (1, 6, 12)):
            ys.append(y)
    return ys


def load_tracks():
    """Global IBTrACS storm tracks, 6-hourly, TS intensity or stronger.

    The centre comes from the basin's own regional agency (see AGENCY), falling back to the
    US/JTWC record and then to the generic IBTrACS position, so every basin uses the best
    track its scientific community expects while the West Pacific stays on CMA (consistent
    with the single-basin model).
    """
    ds = xr.open_dataset(IBT)
    def dec(x): return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
    iso = ds['iso_time'].values
    bas = ds['basin'].values; sid = ds['sid'].values
    glat = ds['lat'].values; glon = ds['lon'].values
    n_storm = iso.shape[0]
    # pre-read every candidate agency field
    fields = {}
    for ag in ['cma', 'usa', 'wmo', 'tokyo', 'newdelhi', 'reunion', 'bom', 'nadi']:
        if f'{ag}_lat' in ds and f'{ag}_lon' in ds:
            fields[ag] = (ds[f'{ag}_lat'].values, ds[f'{ag}_lon'].values,
                          ds[f'{ag}_wind'].values if f'{ag}_wind' in ds else None)
    recs = []
    used = {}
    for i in range(n_storm):
        basin = dec(bas[i, 0])
        ts = pd.to_datetime([dec(x) for x in iso[i]], errors='coerce')
        got = None
        for ag in AGENCY.get(basin, ['usa']):
            if ag not in fields:
                continue
            la, lo, w = fields[ag]
            if w is None:
                continue
            v = (np.isfinite(la[i]) & np.isfinite(lo[i]) & np.isfinite(w[i]) & (w[i] >= 34.0)
                 & ~pd.isna(ts))
            if v.sum() >= 4:
                got = (ag, np.asarray(la[i], float)[v], np.asarray(lo[i], float)[v],
                       np.asarray(w[i], float)[v], ts[v])
                break
        if got is None:
            v = (np.isfinite(glat[i]) & np.isfinite(glon[i]) & ~pd.isna(ts))
            if v.sum() < 4:
                continue
            got = ('generic', np.asarray(glat[i], float)[v], np.asarray(glon[i], float)[v],
                   np.full(int(v.sum()), 34.0), ts[v])
        ag, la_, lo_, w_, t_ = got
        used[ag] = used.get(ag, 0) + 1
        recs.append(dict(sid=dec(sid[i]), basin=basin, agency=ag, time=t_,
                         lat=la_, lon=(lo_ + 360.0) % 360.0, wind=w_))
    ds.close()
    print('[IBTrACS] centre source used per storm:', used, flush=True)
    return recs


def _read_month(args):
    """Worker: read one monthly global file once and return every basin box as float16."""
    year, m, basins, boxes = args
    p = era5_path(year, m)
    if p is None:
        return None
    out = {}
    try:
        ds = xr.open_dataset(p)
    except Exception as e:
        return None
    tname = 'valid_time' if 'valid_time' in ds else 'time'
    t = pd.to_datetime(ds[tname].values)
    keep = np.where(np.isin(t.hour, [0, 6, 12, 18]) & (t.minute == 0))[0]
    if keep.size == 0:
        ds.close(); return None
    lat_all = ds['latitude'].values
    ascending = lat_all[0] < lat_all[-1]
    lats = lons = None
    for b in basins:
        lon0, lon1 = boxes[b]['lon']; lat0, lat1 = boxes[b]['lat']
        ls = slice(lat0, lat1) if ascending else slice(lat1, lat0)
        try:
            sub = (ds[['u10', 'v10']].isel({tname: keep})
                     .sel(latitude=ls, longitude=slice(lon0, lon1)))
            if sub.sizes.get('latitude', 0) < 2 or sub.sizes.get('longitude', 0) < 2:
                continue
            arr = np.stack([sub['u10'].values, sub['v10'].values], axis=1).astype(np.float16)
        except Exception:
            continue
        out[b] = arr
        if lats is None:
            lats = sub['latitude'].values.astype(np.float32)
            lons = sub['longitude'].values.astype(np.float32)
    ds.close()
    return dict(year=year, month=m, data=out, times=t[keep], lat=lats, lon=lons)


def _read_month_star(a):
    return _read_month(a)


def build_year(year, basins, tracks, overwrite=False):
    """Read each monthly global file ONCE and slice every basin box out of it."""
    months = [m for m in range(1, 13) if era5_path(year, m)]
    if not months:
        print(f'  [skip] {year}: no ERA5'); return
    # months in which each basin actually has a storm (skips dead months)
    active = {}
    for b in basins:
        mm = set()
        for r in tracks:
            if r['basin'] != b:
                continue
            lon0, lon1 = BASINS[b]['lon']; lat0, lat1 = BASINS[b]['lat']
            for t_, la_, lo_ in zip(r['time'], r['lat'], r['lon']):
                if lon0 <= lo_ <= lon1 and lat0 <= la_ <= lat1:
                    mm.add(pd.Timestamp(t_).month)
        active[b] = mm
    months = [m for m in months if any(m in active[b] for b in basins)]
    print(f'    active months: {months}', flush=True)
    acc = {b: dict(chunks=[], times=[]) for b in basins}
    lats = lons = None
    jobs = [(year, m, [b for b in basins if m in active[b]], BASINS) for m in months]
    jobs = [j for j in jobs if j[2]]
    # netCDF4/HDF5 is NOT thread safe (concurrent reads abort with SIGABRT), so months are
    # read in separate PROCESSES. 'fork' avoids re-importing this module in every worker.
    ctx = mp.get_context('fork')
    with ProcessPoolExecutor(max_workers=min(len(jobs), MPI_WORKERS) or 1, mp_context=ctx) as pool:
        for res in pool.map(_read_month_star, jobs, chunksize=1):
            if res is None:
                continue
            for b, arr in res['data'].items():
                acc[b]['chunks'].append(arr)
                acc[b]['times'].append(res['times'])
                if lats is None:
                    lats = res['lat']; lons = res['lon']
            print(f"    {year}-{res['month']:02d}: {len(res['times'])} steps x "
                  f"{len(res['data'])} basins", flush=True)

    for b in basins:
        out_f = f'{OUT}/era5_wind_{b}_{year}.npz'
        trk_f = f'{OUT}/tracks_{b}_{year}.npz'
        if os.path.exists(out_f) and os.path.exists(trk_f) and not overwrite:
            print(f'  [skip] {b} {year} (exists)'); continue
        if not acc[b]['chunks']:
            continue
        data = np.concatenate(acc[b]['chunks'], axis=0)
        tt = np.concatenate(acc[b]['times'])
        if lats is None or data.shape[2] != lats.size:
            # basin-specific coordinates: re-derive from the box
            box = BASINS[b]
            nlat = data.shape[2]; nlon = data.shape[3]
            lats_b = np.linspace(box['lat'][1], box['lat'][0], nlat).astype(np.float32)
            lons_b = np.linspace(box['lon'][0], box['lon'][1], nlon).astype(np.float32)
        else:
            lats_b, lons_b = lats, lons
        np.savez_compressed(out_f, data=data, time=tt.astype('datetime64[ns]'),
                            lat=lats_b, lon=lons_b, variables=np.array(['u10', 'v10']))
        # ---- per-storm tracks (ragged) --------------------------------------------
        # One entry per storm, holding that storm's own 6-hourly sequence. The dataset then
        # makes one sample per storm per time step, which removes the "strongest storm at each
        # time" switching artefact of the previous per-timestep centre list.
        lon0, lon1 = BASINS[b]['lon']; lat0, lat1 = BASINS[b]['lat']
        storms = []
        for r in tracks:
            if r['basin'] != b:
                continue
            # restrict to this build year (a storm may straddle a year boundary, so mask by
            # the timestamp of each track point rather than by the storm's start year)
            inyear = np.asarray(pd.DatetimeIndex(r['time']).year == year)
            keep = ((r['lon'] >= lon0) & (r['lon'] <= lon1)
                    & (r['lat'] >= lat0) & (r['lat'] <= lat1) & inyear)
            if keep.sum() < 5:
                continue
            # IBTrACS reports positions at 3-hourly (or irregular) intervals for recent
            # storms, while ERA5 is 6-hourly. Snap every track point to the nearest 6-hourly
            # analysis time and keep, for each analysis time, the closest original point.
            raw_t = pd.DatetimeIndex(r['time'])[keep]
            raw_la = r['lat'][keep]; raw_lo = r['lon'][keep]; raw_w = r['wind'][keep]
            snapped = raw_t.round('6h')
            delta = np.abs((snapped - raw_t).total_seconds())
            best = {}
            for q in range(len(snapped)):
                key = snapped[q]
                if key not in best or delta[q] < best[key][0]:
                    best[key] = (delta[q], raw_la[q], raw_lo[q], raw_w[q])
            # a track point further than 2 h from any analysis time is not usable
            keys = sorted(k for k, v in best.items() if v[0] <= 2 * 3600)
            if len(keys) < 5:
                continue
            tt = pd.DatetimeIndex(keys)
            la_s = np.array([best[k][1] for k in keys], np.float32)
            lo_s = np.array([best[k][2] for k in keys], np.float32)
            w_s = np.array([best[k][3] for k in keys], np.float32)
            # keep only steps whose predecessor and successor are present (the dataset needs a
            # contiguous 4-step history and a 1-step target)
            idx = np.where(keep)[0]
            ok = np.zeros(len(tt), bool)
            tset = set(tt)
            for i, t in enumerate(tt):
                need = [t + pd.Timedelta(hours=6 * d) for d in range(-3, 2)]
                if all(n in tset for n in need):
                    ok[i] = True
            if ok.sum() < 1:
                continue
            storms.append((r['sid'], tt[ok], la_s[ok], lo_s[ok], w_s[ok]))
        if not storms:
            print(f'  [warn] {b} {year}: no usable storms'); continue
        Tmax = max(len(x[1]) for x in storms)
        S = len(storms)
        sid_arr = np.array([x[0] for x in storms], dtype=object)
        tk_time = np.full((S, Tmax), np.datetime64('NaT'), 'datetime64[ns]')
        tk_lat = np.full((S, Tmax), np.nan, np.float32)
        tk_lon = np.full((S, Tmax), np.nan, np.float32)
        tk_wind = np.full((S, Tmax), np.nan, np.float32)
        for i, (sid, tt, la_, lo_, w_) in enumerate(storms):
            n = len(tt)
            tk_time[i, :n] = tt.values.astype('datetime64[ns]')
            tk_lat[i, :n] = la_; tk_lon[i, :n] = lo_; tk_wind[i, :n] = w_
        np.savez_compressed(
            f'{OUT}/tracks_{b}_{year}.npz', storm_ids=sid_arr,
            n_times=np.array([len(x[1]) for x in storms], np.int32),
            time=tk_time, lat=tk_lat, lon=tk_lon, wind_kt=tk_wind,
            lat_grid=lats_b, lon_grid=lons_b)
        n_samples = sum(1 for _ in range(len(tk_time)))  # placeholder, dataset recounts
        tot_steps = int(np.isfinite(tk_lat).sum())
        print(f'  [done] {b} {year}: array {data.shape} ({data.nbytes/1e6:.0f} MB), '
              f'{S} storms, {tot_steps} storm steps', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--basins', nargs='*', default=None)
    ap.add_argument('--years', nargs='*', type=int, default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    avail = available_years()
    print(f'[ERA5] global years available: {avail}')
    years = args.years if args.years else avail
    basins = args.basins if args.basins else list(BASINS)
    print(f'[PLAN] basins={basins}', flush=True)
    print(f'[PLAN] years={years}', flush=True)
    tracks = load_tracks()
    print(f'[IBTrACS] {len(tracks)} storms', flush=True)
    for y in years:
        print(f'[YEAR] {y}', flush=True)
        build_year(y, basins, tracks, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
