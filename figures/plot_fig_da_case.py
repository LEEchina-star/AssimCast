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

# plot_fig_da_case.py — training-free satellite assimilation, inner-core case study.
#
# Leakage-free design: the six-hour forecast issued at tau is guided ONLY by
# satellite overpasses inside the causal window (t <= tau); it is then scored
# against a *future*, independent overpass acquired after the issue time
# (|t - t1| <= 3 h), which is never assimilated.
#
# Showcase: Super Typhoon Yagi (12W) in the western North Pacific,
# tau=2024090418 (issue 18:00 UTC 4 Sep 2024), t1=2024090500 (valid 00:00 UTC).
# An independent future Sentinel-1A SAR swath crosses the eyewall at
# 22:18 UTC 4 Sep 2024 (never assimilated).
#
#   a) six-hour forecast WITHOUT assimilation
#   b) six-hour forecast WITH training-free multi-satellite assimilation
#   c) DA - No-DA correction (dashed: independent future SAR 33 m/s eyewall)
#   d) independent future SAR wind (the arbiter)
#   e) forecast vs observed SAR point cloud, No DA vs DA
#   f) forecast-minus-observed bias histogram, No DA vs DA
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "docs", "NC_manuscript", "figures")
DA_DIR = os.environ.get(
    "CASE_DIR", os.path.join(HERE, "data", "hafs_pred_v6_deploy_da"))
RES = 0.06; SIZE = 336; C = (SIZE - 1) / 2.0
BASIN = os.environ.get("CASE_BASIN", "WP")
SID = os.environ.get("CASE_SID", "2026170N12145")
TAU = os.environ.get("CASE_TAU", "2026062306")
T1 = os.environ.get("CASE_T1", "2026062312")
OUT_TAG = os.environ.get("CASE_TAG", "")
RAD = float(os.environ.get("CASE_RAD", "2.2"))
VMIN, VMAX = 0, 60
CMIN, CMAX = 6.0, 6.0
C_NO = "#f0a35e"; C_DA = "#1f4e79"

TCWIND = LinearSegmentedColormap.from_list(
    "tcwind",
    [(0.00, "#ffffff"), (0.14, "#f2f6d0"), (0.28, "#fde89a"), (0.42, "#f6c04b"),
     (0.56, "#ef8a32"), (0.70, "#d94e2a"), (0.84, "#a8221b"), (1.00, "#6b1414")])

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "legend.frameon": False, "font.family": "DejaVu Sans"})


def load_src(s):
    z = np.load(os.path.join(HERE, f"data/observations/compat/{s}_global_30min.npz"),
                allow_pickle=True)
    out = dict(t=pd.to_datetime(z["time_ns"]), lat=z["lat"].astype(float),
               lon=z["lon"].astype(float), ws=z["ws"].astype(float),
               sat=z["sat"] if "sat" in z.files else None)
    return out


def sensor_name(keys):
    low = " ".join(str(k).lower() for k in keys)
    names = []
    for tag, label in [("s1a", "Sentinel-1A"), ("s1b", "Sentinel-1B"),
                       ("s1c", "Sentinel-1C"), ("s1d", "Sentinel-1D"),
                       ("rs2", "RADARSAT-2"), ("radarsat", "RADARSAT-2"),
                       ("rcm", "RADARSAT Constellation Mission")]:
        if tag in low and label not in names:
            names.append(label)
    if not names:
        return "SAR"
    return " and ".join(names) + " SAR"


def sensor_short(keys):
    low = " ".join(str(k).lower() for k in keys)
    codes = []
    for tag, code in [("s1a", "S1A"), ("s1b", "S1B"), ("s1c", "S1C"),
                      ("s1d", "S1D"), ("rs2", "RS2"), ("radarsat", "RS2"),
                      ("rcm", "RCM")]:
        if tag in low and code not in codes:
            codes.append(code)
    return "/".join(codes) if codes else "SAR"


def bilinear(W, lat0, lon0, ql, qo):
    fi = (ql - lat0) / RES; fj = (qo - lon0) / RES
    i0 = np.floor(fi).astype(int); j0 = np.floor(fj).astype(int)
    ok = (i0 >= 0) & (i0 < SIZE - 1) & (j0 >= 0) & (j0 < SIZE - 1)
    out = np.full(ql.shape, np.nan)
    ai = (fi - i0)[ok]; aj = (fj - j0)[ok]; ii = i0[ok]; jj = j0[ok]
    out[ok] = ((1 - ai) * (1 - aj) * W[ii, jj] + ai * (1 - aj) * W[ii + 1, jj]
               + (1 - ai) * aj * W[ii, jj + 1] + ai * aj * W[ii + 1, jj + 1])
    return out


def main():
    meta = pd.read_csv(os.path.join(DA_DIR, f"meta_{BASIN}_da.csv"))
    if T1:
        k = int(meta.index[(meta.sid == SID) & (meta.t1 == int(T1))][0])
    else:
        k = int(meta.index[(meta.sid == SID) & (meta.tau == int(TAU))][0])
    mt = meta.loc[k]
    clat, clon = float(mt.clat), float(mt.clon)
    t1 = pd.to_datetime(str(int(mt.t1)), format="%Y%m%d%H")
    zda = np.load(os.path.join(DA_DIR, f"pred_{BASIN}_da.npz"))["f"]
    zno = np.load(os.path.join(DA_DIR, f"pred_{BASIN}_noda.npz"))["f"]
    Wd = np.hypot(zda[k, 0], zda[k, 1])
    Wn = np.hypot(zno[k, 0], zno[k, 1])
    lat0 = clat - C * RES; lon0 = clon - C * RES
    extent = [clon - RAD, clon + RAD, clat - RAD, clat + RAD]

    sar = load_src("sar")
    dt = (sar["t"] - t1).total_seconds() / 3600.0
    dlat = np.abs(sar["lat"] - clat); dlon = np.abs((sar["lon"] - clon + 180) % 360 - 180)
    sel = (np.abs(dt) <= 3.0) & (dlat < RAD) & (dlon < RAD)
    slat, slon, sy = sar["lat"][sel], sar["lon"][sel], sar["ws"][sel]
    ssat = sar["sat"][sel] if sar["sat"] is not None else None
    pn = bilinear(Wn, lat0, lon0, slat, slon)
    pd_ = bilinear(Wd, lat0, lon0, slat, slon)
    ok = np.isfinite(pn) & np.isfinite(pd_) & np.isfinite(sy) \
        & np.isfinite(slat) & np.isfinite(slon)
    slat, slon, sy, pn, pd_ = slat[ok], slon[ok], sy[ok], pn[ok], pd_[ok]
    if ssat is not None:
        ssat = ssat[ok]
    rn = np.sqrt(np.mean((pn - sy) ** 2)); rd = np.sqrt(np.mean((pd_ - sy) ** 2))
    bn = np.mean(pn - sy); bd = np.mean(pd_ - sy)
    n = len(sy)
    sname = sensor_name(pd.unique(ssat)) if ssat is not None else "SAR"
    sshort = sensor_short(pd.unique(ssat)) if ssat is not None else "SAR"

    fig = plt.figure(figsize=(14.6, 9.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.035],
                          hspace=0.42, wspace=0.20)
    kk = dict(ha="left", va="top", fontsize=14, fontweight="bold", zorder=10)

    def tag(ax, t):
        ax.text(0.03, 0.97, t, transform=ax.transAxes, **kk,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))

    def map_ax(ax, W, title, ylab=True):
        im = ax.imshow(W, origin="lower", extent=extent, cmap=TCWIND,
                       vmin=VMIN, vmax=VMAX, zorder=0)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.plot(clon, clat, "+", color="k", ms=9, mew=1.8, zorder=5)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xlabel("Longitude ($^\\circ$E)")
        if ylab:
            ax.set_ylabel("Latitude ($^\\circ$N)")
        return im

    axa = fig.add_subplot(gs[0, 0]); im = map_ax(axa, Wn, "a  Forecast without assimilation")
    axb = fig.add_subplot(gs[0, 1])
    im = map_ax(axb, Wd, "b  Forecast with training-free assimilation", ylab=False)

    # c: correction
    axc = fig.add_subplot(gs[0, 2])
    dW = Wd - Wn
    imc = axc.imshow(dW, origin="lower", extent=extent, cmap="RdBu_r",
                     vmin=-CMIN, vmax=CMAX, zorder=0)
    cf = np.isfinite(sy) & np.isfinite(slat) & np.isfinite(slon)
    clatc, slonc, syc = slat[cf], slon[cf], sy[cf]
    tri = mtri.Triangulation(slonc, clatc)
    xe = slonc[tri.triangles]; ye = clatc[tri.triangles]
    edge = np.maximum.reduce([
        np.hypot(xe[:, 0] - xe[:, 1], ye[:, 0] - ye[:, 1]),
        np.hypot(xe[:, 1] - xe[:, 2], ye[:, 1] - ye[:, 2]),
        np.hypot(xe[:, 2] - xe[:, 0], ye[:, 2] - ye[:, 0])])
    tri.set_mask(edge > 0.15)
    axc.tricontour(tri, syc, levels=[33], colors="k", linewidths=1.2,
                   linestyles="--", zorder=4)
    axc.plot(clon, clat, "+", color="k", ms=9, mew=1.8, zorder=5)
    axc.set_xlim(extent[0], extent[1]); axc.set_ylim(extent[2], extent[3])
    axc.set_aspect("equal")
    axc.set_xlabel("Longitude ($^\\circ$E)")
    axc.set_title("c  Correction applied (DA $-$ No DA)", fontsize=11, pad=6)
    tag(axc, "")  # title already carries the letter

    # d: independent future SAR
    axd = fig.add_subplot(gs[1, 0])
    scd = axd.scatter(slon, slat, c=sy, s=11, cmap=TCWIND, vmin=VMIN, vmax=VMAX,
                      edgecolors="0.35", linewidths=0.05, marker="s", zorder=2)
    axd.plot(clon, clat, "+", color="k", ms=9, mew=1.8, zorder=5)
    axd.set_xlim(extent[0], extent[1]); axd.set_ylim(extent[2], extent[3])
    axd.set_aspect("equal")
    axd.set_xlabel("Longitude ($^\\circ$E)"); axd.set_ylabel("Latitude ($^\\circ$N)")
    axd.set_title(f"d  Independent future SAR ({sshort}; never assimilated)", fontsize=11, pad=6)

    # e: scatter
    axe = fig.add_subplot(gs[1, 1])
    axe.scatter(sy, pn, s=7, alpha=0.30, color=C_NO, rasterized=True)
    axe.scatter(sy, pd_, s=7, alpha=0.30, color=C_DA, rasterized=True)
    axe.plot([0, 60], [0, 60], "k--", lw=0.9)
    axe.scatter([], [], color=C_NO, label=f"No DA: RMSE {rn:.2f} m s$^{{-1}}$")
    axe.scatter([], [], color=C_DA, label=f"DA: RMSE {rd:.2f} m s$^{{-1}}$")
    axe.set_xlim(0, 60); axe.set_ylim(0, 60); axe.set_aspect("equal")
    axe.set_xlabel("Observed future SAR wind (m s$^{-1}$)")
    axe.set_ylabel("Forecast wind (m s$^{-1}$)")
    axe.set_title(f"e  Forecast vs independent SAR (n={n:,})", fontsize=11, pad=6)
    axe.legend(loc="upper left", fontsize=9)
    axe.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    # f: bias histogram
    axf = fig.add_subplot(gs[1, 2])
    bins = np.linspace(-25, 25, 61)
    axf.hist(pn - sy, bins=bins, density=True, color=C_NO, alpha=0.55,
             label=f"No DA: bias {bn:+.1f} m s$^{{-1}}$")
    axf.hist(pd_ - sy, bins=bins, density=True, color=C_DA, alpha=0.55,
             label=f"DA: bias {bd:+.1f} m s$^{{-1}}$")
    axf.axvline(0, color="k", lw=0.9)
    axf.set_xlabel("Forecast $-$ observed (m s$^{-1}$)")
    axf.set_ylabel("Probability density")
    axf.set_xlim(-22, 22)
    axf.set_title("f  Bias against independent future SAR", fontsize=11, pad=6)
    axf.legend(loc="upper right", fontsize=9)
    axf.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    # shared horizontal colorbars in the bottom strip
    caxw = fig.add_subplot(gs[2, 0:2])
    cbw = fig.colorbar(im, cax=caxw, orientation="horizontal", extend="both",
                       extendfrac=0.03)
    cbw.set_label("Wind speed (m s$^{-1}$)", fontsize=10)
    cbw.ax.tick_params(labelsize=8)
    caxc = fig.add_subplot(gs[2, 2])
    cbc = fig.colorbar(imc, cax=caxc, orientation="horizontal", extend="both",
                       extendfrac=0.04)
    cbc.set_label("$\\Delta$ wind (m s$^{-1}$)", fontsize=10)
    cbc.ax.tick_params(labelsize=8)

    name = "fig_da_case" + (f"_{OUT_TAG}" if OUT_TAG else "")
    out = os.path.join(FIG, name + ".pdf")
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"),
                                                      dpi=200, bbox_inches="tight")
    print("saved", out, "| sensor:", sname, "| n=", n,
          "RMSE", round(float(rn), 2), "->", round(float(rd), 2),
          "bias", round(float(bn), 2), "->", round(float(bd), 2))


if __name__ == "__main__":
    main()
