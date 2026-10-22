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

# plot_structure_radii.py - composite radial structure (cat3+) + wind-radius bias.
# Two stages:
#   np2  python plot_structure_radii.py compute   (eccodes; reads netCDF/npz)
#   base python plot_structure_radii.py plot      (matplotlib; reads cached npz)
import os, sys, glob, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

CAT3 = 49.4
THR_KT = {"R34": 17.5, "R50": 25.7, "R64": 32.9}
OUT = os.path.join(HERE, "docs", "NC_manuscript", "figures")
CACHE = "/tmp/radial_profiles.npz"


def compute():
    import score_structure_radii as S
    from hafs_gate_ibtracs import load_ib
    prof = {f: [] for f in ["era", "tgt", "pred"]}
    rr0 = None
    for fp in sorted(glob.glob(os.path.join(HERE, "data", "hafs_pred_v7_final", "pred_*_test.npz"))):
        tag = os.path.basename(fp)[len("pred_"):-len("_test.npz")]
        basin = tag.split("_")[0]
        mp = os.path.join(HERE, "data", "hafs_pred_v7_final", f"meta_{tag}_test.csv")
        if not os.path.exists(mp):
            continue
        F = np.load(fp)["f"].astype(float)
        meta = pd.read_csv(mp, keep_default_na=False)
        size = F.shape[-1]
        try:
            ds = load_ib(S.IB_MAP.get(basin, basin))
        except Exception:
            continue
        for i in range(F.shape[0]):
            mt = meta.iloc[i]
            t1 = datetime.strptime(str(mt["tau"]).zfill(10), "%Y%m%d%H") + timedelta(hours=6)
            loc = S.locate_fix(ds, t1, float(mt["clat"]), float(mt["clon"]))
            if loc is None:
                continue
            s, t = loc
            w = float(ds.usa_wind.values[s, t]) * S.KT
            if w < CAT3:
                continue
            LAT, LON = S.gridv(float(mt["clat"]), float(mt["clon"]), size, 0.06)
            mask = F[i][10]
            cla = float(ds.usa_lat.values[s, t]); clo = float(ds.usa_lon.values[s, t])
            for fld, c in [("era", 8), ("tgt", 2), ("pred", 0)]:
                U = np.where(mask > 0.5, F[i][c], np.nan)
                V = np.where(mask > 0.5, F[i][c + 1], np.nan)
                ef, eo = S.eye_center(U, V, LAT, LON, cla, clo)
                rr, p = S.ray_profiles(np.hypot(U, V), LAT, LON, ef, eo)
                rr0 = rr
                prof[fld].append(p)
        print(f"{basin}: accumulated", flush=True)
    rmax = 300.0
    keep = rr0 <= rmax
    out = {"rr": rr0[keep]}
    for f in prof:
        A = np.vstack([np.nanmean(p[:, keep], axis=0) for p in prof[f]])
        out[f"{f}_med"] = np.nanmedian(A, 0)
        out[f"{f}_q1"] = np.nanpercentile(A, 25, 0)
        out[f"{f}_q3"] = np.nanpercentile(A, 75, 0)
        out[f"{f}_n"] = np.array([A.shape[0]])
    d = pd.read_csv(os.path.join(HERE, "results", "structure_radii_2024.csv"))
    d3 = d[d["ib_wind_mps"] >= CAT3]
    for k in THR_KT:
        out[f"ib_{k}"] = np.array([np.nanmedian(d3[f"ib_{k}"].values)])
    np.savez(CACHE, **out)
    print("saved", CACHE, "n cat3 profiles per field:", {f: len(prof[f]) for f in prof})


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(CACHE)
    rr = z["rr"]
    ib_med = {k: float(z[f"ib_{k}"][0]) for k in THR_KT}
    stats = json.load(open(os.path.join(HERE, "results", "structure_radii_2024.json")))

    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans",
                         "axes.linewidth": 1.0, "xtick.direction": "out", "ytick.direction": "out"})
    col = {"era": "#7f7f7f", "tgt": "#2166ac", "pred": "#d6604d"}
    lab = {"era": "ERA5", "tgt": "HAFS reference (0.06$^\\circ$)", "pred": "AssimCast"}

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.2))
    a = ax[0]
    for f in ["era", "tgt", "pred"]:
        ls = "--" if f == "era" else "-"
        a.plot(rr, z[f"{f}_med"], ls, color=col[f], lw=2.4, label=lab[f], zorder=3)
        a.fill_between(rr, z[f"{f}_q1"], z[f"{f}_q3"], color=col[f], alpha=0.13, zorder=1)
    for k, v in THR_KT.items():
        a.axhline(v, color="0.75", lw=0.9, ls=":", zorder=0)
        a.text(rr[-1] - 4, v + 0.8, k, ha="right", va="bottom", color="0.45", fontsize=9)
    for k, rr_ib in ib_med.items():
        a.axvline(rr_ib, color="0.3", lw=1.1, ls=(0, (4, 2)), alpha=0.8)
        a.text(rr_ib, 68.5, k, ha="center", va="top", color="0.2", fontsize=9)
    a.set_xlim(0, rr[-1]); a.set_ylim(0, 72)
    a.set_xlabel("Radius from storm centre (km)")
    a.set_ylabel("Axisymmetric wind speed (m s$^{-1}$)")
    a.set_title("Radial wind structure, 2024 cat-3+ storms")
    a.legend(loc="upper right", frameon=False, fontsize=9.5)
    a.text(0.03, 0.04, "Dashed verticals: IBTrACS median R34/R50/R64",
            transform=a.transAxes, fontsize=8.5, color="0.35")

    b = ax[1]
    names = ["R34", "R50", "R64"]
    x = np.arange(len(names)); w = 0.25
    for j, f in enumerate(["era", "tgt", "pred"]):
        biases = [stats[k]["cat3+"][f]["bias"] for k in names]
        hits = [stats[k]["cat3+"][f]["hit_rate"] for k in names]
        bars = b.bar(x + (j - 1) * w, biases, w, color=col[f], label=lab[f],
                     edgecolor="k", linewidth=0.5)
        for bb, h, bv in zip(bars, hits, biases):
            cx = bb.get_x() + bb.get_width() / 2
            if abs(bv) >= 20:  # tall bar: horizontal label inside the bar
                yt = bv - 6 if bv > 0 else bv + 6
                b.text(cx, yt, f"{h:.2f}", ha="center",
                       va="top" if bv > 0 else "bottom",
                       fontsize=8, color="white", fontweight="bold")
            else:  # short bar: label just outside
                yt = bv + 3 if bv >= 0 else bv - 3
                b.text(cx, yt, f"hit {h:.2f}", ha="center",
                       va="bottom" if bv >= 0 else "top", fontsize=7.5)
    b.axhline(0, color="k", lw=1.0)
    b.set_xticks(x); b.set_xticklabels(names)
    b.set_ylabel("Wind-radius bias (km)")
    b.set_ylim(-118, 118)
    b.set_title("Quadrant wind-radius bias, 2024 cat-3+")
    b.text(0.01, 0.02, "Labels: quadrant coverage (hit rate)",
            transform=b.transAxes, fontsize=8.5, color="0.35")
    b.legend(loc="upper right", frameon=False, fontsize=9.5)

    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ["pdf", "png"]:
        fo = os.path.join(OUT, f"fig_structure_radii.{ext}")
        fig.savefig(fo, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print("wrote", fo)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "compute"
    if mode == "compute":
        compute()
    else:
        plot()
