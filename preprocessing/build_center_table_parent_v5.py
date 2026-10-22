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

# build_center_table_parent_v5.py - operational HAFS trak.atcfunix vortex centres for
# Stage A on the parent 0.06 deg (336, 20.1 deg) fixed grid. History centres at tau-18,-12,
# -6,tau come from the f000 fix of each init cycle; the target centre is the tau-init
# f006 fix. All remapped to the tau-anchored grid (absolute pixel coords). This is what the
# CenterHead consumes at inference (HAFS initialisation centres are operationally known),
# so train and inference inputs are consistent.
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
HAFS = os.path.join(HERE, "data/noaa/hafs")
META_GLOB = os.path.join(HERE, "data/hafs_parent_v4/meta_*_*.csv")
OUT = os.path.join(HERE, "data/hafs_parent_v4/centers_trak_v5.csv")
D = 0.06
G = 336
C = (G - 1) / 2.0
KM = 111.32 * D


def ll(s):
    v = float(s[:-1]) / 10.0
    return -v if s[-1] in "SW" else v


def parse_hours(path):
    h = {}
    if not os.path.exists(path):
        return h
    with open(path, errors="ignore") as fh:
        for line in fh:
            f = [x.strip() for x in line.split(",")]
            if "HFSA" not in f:
                continue
            hi = f.index("HFSA")
            try:
                hr = int(f[hi + 1]); la = ll(f[hi + 2]); lo = ll(f[hi + 3])
            except Exception:
                continue
            if hr not in h:
                h[hr] = (la, lo)
    return h


_cache = {}
def hours_for(basin, sid, init):
    key = (basin, sid, init)
    if key not in _cache:
        _cache[key] = parse_hours(os.path.join(HAFS, basin, sid, init, "trak.atcfunix"))
    return _cache[key]


def wrap_dlon(a, b):
    return ((a - b + 180.0) % 360.0) - 180.0


def gpt(lat, lon, clat, clon):
    return C + (lat - clat) / D, C + wrap_dlon(lon, clon) / D


def km(dy, dx, clat):
    return np.sqrt((dy * KM) ** 2 + (dx * KM * np.cos(np.deg2rad(clat))) ** 2)


def main():
    import glob
    rows = []
    for mp in sorted(glob.glob(META_GLOB)):
        tag = os.path.basename(mp)[len("meta_"):-4]
        b, yr = tag.rsplit("_", 1)
        m = pd.read_csv(mp, keep_default_na=False)
        for row, r in m.iterrows():
            basin, sid, tau = b, str(r["sid"]), str(r["tau"]).zfill(10)
            clat, clon = float(r["clat"]), float(r["clon"]) % 360.0
            dt = datetime.strptime(tau, "%Y%m%d%H")
            t1 = (dt + timedelta(hours=6)).strftime("%Y%m%d%H")
            rec = dict(basin=basin, year=int(yr), row=int(row), sid=sid,
                       tau=tau, t1=t1, set=r["set"],
                       clat=clat, clon=clon, avail=0)
            hist = []
            for dh in (-18, -12, -6, 0):
                di = (dt + timedelta(hours=dh)).strftime("%Y%m%d%H")
                h = hours_for(basin, sid, di)
                if 0 not in h:
                    rows.append(rec); break
                la, lo = h[0]
                hist.append((la, lo % 360.0))
            else:
                h0 = hours_for(basin, sid, tau)
                if 6 not in h0:
                    rows.append(rec); continue
                tla, tlo = h0[6]; tlo %= 360.0
                for k, (la, lo) in enumerate(hist):
                    iy, ix = gpt(la, lo, clat, clon)
                    rec[f"hy{k}"] = iy; rec[f"hx{k}"] = ix
                ty, tx = gpt(tla, tlo, clat, clon)
                rec["ty"] = ty; rec["tx"] = tx
                rec["anchor_resid_km"] = float(
                    km(hist[3][0] - clat, wrap_dlon(hist[3][1], clon) / D, clat))
                rec["avail"] = 1
                rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    ok = df[df.avail == 1].copy()
    print(f"total={len(df)} trak-centre available={len(ok)} missing={len(df)-len(ok)}")
    hy = ok[[f"hy{k}" for k in range(4)]].values
    hx = ok[[f"hx{k}" for k in range(4)]].values
    ty, tx = ok["ty"].values, ok["tx"].values
    cl = ok["clat"].values
    targ = km(ty - hy[:, 3], tx - hx[:, 3], cl)
    print("tau->f006 displacement (km): median %.0f mean %.0f p90 %.0f" % (
        np.median(targ), np.mean(targ), np.percentile(targ, 90)))
    print("anchor f000 vs window-centre residual: median %.0f p90 %.0f km" % (
        ok.anchor_resid_km.median(), ok.anchor_resid_km.quantile(.9)))
    print("available by set:\n", ok.groupby("set").size())
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
