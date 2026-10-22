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

# build_env_flow.py - Precompute the outer environmental advection field (env_y/x) for
# every parent-v4 sample, on CPU (does not touch MPS training). The 6-h environmental
# motion is a large-scale smooth optical flow from the f000 tau-6 -> tau frames; the
# inner-core displacement is supplied separately by the CenterHead and blended with the
# tanh weight in v5_warp. Flow is computed on a gaussian-blurred wind-speed field so it
# represents the large-scale environmental current rather than eyewall convection.
import os, glob, json
import numpy as np, pandas as pd, cv2

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data/hafs_parent_v4")
CEN = os.path.join(DATA, "centers_trak_v5.csv")
G = 336
BLUR = 31        # ~1.9 deg large-scale smoothing


def flow_one(prev_z, tau_z):
    def mag(z):
        m = np.hypot(z[0], z[1]).astype(np.float32)
        m = cv2.GaussianBlur(m, (BLUR | 1, BLUR | 1), 0)
        return cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    fl = cv2.calcOpticalFlowFarneback(mag(prev_z), mag(tau_z), None,
                                      0.5, 4, 31, 3, 5, 1.2, 0)
    # smooth the raw flow (large-scale environmental current), pixels / 6 h
    fy = cv2.GaussianBlur(fl[:, :, 1], (BLUR | 1, BLUR | 1), 0)
    fx = cv2.GaussianBlur(fl[:, :, 0], (BLUR | 1, BLUR | 1), 0)
    return np.stack([fy, fx], 0).astype(np.float16)


def main():
    cen = pd.read_csv(CEN, keep_default_na=False)
    cen = cen[cen.avail == 1]
    for npy in sorted(glob.glob(os.path.join(DATA, "hafs_parent_v4_*.npy"))):
        base = os.path.basename(npy)[:-4]
        tag = base[len("hafs_parent_v4_"):]
        b, yr = tag.rsplit("_", 1)
        out = os.path.join(DATA, f"envflow_{b}_{yr}.npy")
        if os.path.exists(out):
            print("skip", out); continue
        arr = np.load(npy, mmap_mode="r")
        rows = cen[(cen.basin == b) & (cen.year == int(yr))]
        flows = np.zeros((arr.shape[0], 2, G, G), np.float16)
        k = 0
        for _, r in rows.iterrows():
            i = int(r.row)
            if i >= arr.shape[0]:
                continue
            z = np.array(arr[i]).astype(np.float32)
            flows[i] = flow_one(z[4:6], z[6:8])
            k += 1
        np.save(out, flows)
        print(f"{b}_{yr}: {k} flows -> {out} ({flows.nbytes/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
