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
parse_sfmr.py — turn the NOAA/AOML HRD SFMR archive into flat tables.

Source : data/noaa/sfmr/<year>/<storm>/AFRC_SFMR*.nc
         Stepped Frequency Microwave Radiometer 10-m surface wind retrievals
         (Uhlhorn et al. 2007, Mon. Wea. Rev. 135, 3070-3085)

Only FLAG == 0 (good) points with a valid SWS are kept.  Output schema matches the
SAR / SMAP files already in data/observations so the same collocation code works:

    data/observations/sfmr_<year>.pkl
        datetime (ns), lat, lon, wind_speed (m/s), storm, satellite_id ('SFMR')

Usage:  python3 parse_sfmr.py [--years 2008 2009 ...]
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import netCDF4 as nc

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
SRC = f'{HERE}/data/noaa/sfmr'
DST = f'{HERE}/data/observations'


def one(path, storm):
    try:
        d = nc.Dataset(path)
    except Exception:
        return None
    for v in ('DATE', 'TIME', 'LAT', 'LON', 'SWS', 'FLAG'):
        if v not in d.variables:
            d.close()
            return None
    date = np.asarray(d['DATE'][:], np.int64)
    time = np.asarray(d['TIME'][:], np.int64)
    lat = np.asarray(d['LAT'][:], np.float32)
    lon = np.asarray(d['LON'][:], np.float32)
    sws = np.asarray(d['SWS'][:], np.float32)
    flg = np.asarray(d['FLAG'][:], np.int32)
    d.close()

    ok = (flg == 0) & (sws > -900) & np.isfinite(sws) & np.isfinite(lat) & np.isfinite(lon)
    if ok.sum() == 0:
        return None
    date, time, lat, lon, sws = date[ok], time[ok], lat[ok], lon[ok], sws[ok]

    ds = pd.to_datetime(date.astype(str), format='%Y%m%d', errors='coerce')
    hh = (time // 10000).astype(int)
    mm = ((time // 100) % 100).astype(int)
    ss = (time % 100).astype(int)
    good = (hh < 24) & (mm < 60) & (ss < 60) & np.asarray(ds.notna())
    if good.sum() == 0:
        return None
    dd = (ds[good] + pd.to_timedelta(hh[good], unit='h')
          + pd.to_timedelta(mm[good], unit='m') + pd.to_timedelta(ss[good], unit='s'))
    return pd.DataFrame(dict(datetime=dd.values, lat=lat[good].astype(np.float32),
                             lon=(lon[good] % 360.0).astype(np.float32),
                             wind_speed=sws[good].astype(np.float32),
                             storm=storm, satellite_id='SFMR'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='*', type=int, default=None)
    ap.add_argument('--stride', type=int, default=10,
                    help='keep every Nth 1-Hz sample (10 -> 10 s resolution)')
    a = ap.parse_args()
    os.makedirs(DST, exist_ok=True)

    years = a.years or sorted(int(x) for x in os.listdir(SRC)
                              if x.isdigit())
    grand = 0
    for y in years:
        files = sorted(glob.glob(f'{SRC}/{y}/*/*.nc'))
        if not files:
            continue
        parts = []
        for p in files:
            storm = os.path.basename(os.path.dirname(p))
            df = one(p, storm)
            if df is not None and len(df):
                parts.append(df)
        if not parts:
            print(f'{y}: no valid points', flush=True)
            continue
        out = pd.concat(parts, ignore_index=True)
        out['datetime'] = pd.to_datetime(out['datetime'])
        if a.stride > 1:
            out = out.iloc[::a.stride].reset_index(drop=True)
        out = out.sort_values('datetime').reset_index(drop=True)
        out.to_pickle(f'{DST}/sfmr_{y}.pkl')
        strong = int((out.wind_speed > 20).sum())
        grand += len(out)
        print(f'{y}: {len(files):4d} flights  {len(out):7d} pts  '
              f'({strong:6d} > 20 m/s, {out.storm.nunique()} storms)  '
              f'{out.wind_speed.min():.1f}-{out.wind_speed.max():.1f} m/s', flush=True)
    print(f'TOTAL {grand} SFMR points written to {DST}/sfmr_<year>.pkl')


if __name__ == '__main__':
    main()
