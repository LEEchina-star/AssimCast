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

# plot_fig4_global_case.py - global peak-intensity recovery + one category-five case.
#   a) predicted vs best-track Vmax across all basins (one-to-one), coloured by basin
#   b) the strongest held-out WP case, full forecast wind field (inner box)
#   c) azimuthal-max radial profiles: era / persistence / Stage A / full / teacher
#   d) that case's inner-core Vmax and the 6-h displacement
# Reads results/ibtracs_v5.csv and data/hafs_pred_v5/pred_WP_test.npz. NC vector PDF.
import os, glob, sys, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

# TC wind colormap: weak winds near-white (de-emphasised), warm inner core (yellow->red->dark)
TCWIND = LinearSegmentedColormap.from_list(
    "tcwind",
    [(0.00, "#ffffff"), (0.14, "#f2f6d0"), (0.28, "#fde89a"), (0.42, "#f6c04b"),
     (0.56, "#ef8a32"), (0.70, "#d94e2a"), (0.84, "#a8221b"), (1.00, "#6b1414")])

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v7ep12")
ap.add_argument("--pred", default=os.path.join(HERE, "data/hafs_pred_v7_final"))
ap.add_argument("--out", default=os.path.join(HERE, "figures", "fig4_global_case.pdf"))
ap.add_argument("--case_sid", default="23w",
                help="showcase storm id (best-matched category-5 case, e.g. 23w=Trami)")
ARGS = ap.parse_args()
IB = os.path.join(HERE, "results", f"ibtracs_{ARGS.tag}.csv")
PRED = ARGS.pred
OUT = ARGS.out
RES = 0.06; G = 336
WP_NAMES = {"12w": "Yagi", "20w": "Krathon", "23w": "Trami", "24w": "Kong-rey",
            "25w": "Man-yi", "26w": "Toraji", "27w": "Usagi"}
BCOL = {"WP": "#c0392b", "NA": "#2471a3", "EP": "#1e8449", "SI": "#b9770e",
        "SP": "#7d3c98", "NI": "#5d6d7e", "CP": "#117a65"}
BLAB = {"WP": "West Pacific", "NA": "North Atlantic", "EP": "East Pacific",
        "SI": "South Indian", "SP": "South Pacific", "NI": "North Indian"}


def wsp(f, i):
    return np.hypot(f[i], f[i + 1])


def vmax_xy(w, msk, half=None):
    ww = np.where(msk, w, np.nan)
    idx = np.unravel_index(np.nanargmax(ww), ww.shape)
    return float(ww[idx]), idx[0], idx[1]


def main():
    d = pd.read_csv(IB, keep_default_na=False)
    for c in ["ib_wind_mps", "pred_vmax", "tgt_vmax", "adv_vmax", "era_vmax", "pers_vmax"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["ib_wind_mps", "pred_vmax"])

    # ---- pick the showcase storm at its peak (best-matched category-5) ----
    z = np.load(os.path.join(PRED, "pred_WP_test.npz"))["f"]
    m = pd.read_csv(os.path.join(PRED, "meta_WP_test.csv"), keep_default_na=False)
    m["sid"] = m["sid"].astype(str)
    tgt = np.hypot(z[:, 2], z[:, 3]); msk = z[:, 10] > 0.5
    peaks = [vmax_xy(tgt[i], msk[i])[0] for i in range(z.shape[0])]
    cand = [i for i in range(z.shape[0]) if m.iloc[i]["sid"] == ARGS.case_sid]
    k = int(cand[int(np.argmax([peaks[i] for i in cand]))]) if cand else int(np.argmax(peaks))
    f = z[k]
    pred, tgtf, adv, pers, era = wsp(f, 0), wsp(f, 2), wsp(f, 4), wsp(f, 6), wsp(f, 8)
    fm = f[10] > 0.5
    # centres from field peaks (eyewall centroid approximation)
    _, py, px = vmax_xy(pers, fm)
    _, ty, tx = vmax_xy(tgtf, fm)
    case_sid = str(m.iloc[k]["sid"])
    case_name = WP_NAMES.get(case_sid, case_sid)
    case_disp = np.hypot(ty - py, tx - px) * RES * 111.0

    fig = plt.figure(figsize=(11.5, 9.6))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0]); axd = fig.add_subplot(gs[1, 1])

    # ---- panel a: global Vmax scatter ----
    for b in ["WP", "NA", "EP", "SI", "SP", "NI", "CP"]:
        g = d[d["basin"] == b]
        if len(g) == 0:
            continue
        axa.scatter(g["ib_wind_mps"], g["pred_vmax"], s=14, alpha=0.65,
                    color=BCOL.get(b, "#444"), edgecolor="none",
                    label=f"{BLAB.get(b,b)} (n={len(g)})")
    lim = [10, 90]
    axa.plot(lim, lim, "--", color="k", lw=1.0)
    axa.set_xlim(lim); axa.set_ylim(lim); axa.set_aspect("equal")
    axa.set_xlabel("Best-track $V_{\\max}$ (m s$^{-1}$)")
    axa.set_ylabel("Forecast $V_{\\max}$ (m s$^{-1}$)")
    axa.set_title("Global peak-intensity recovery, 2024 held out", fontsize=10)
    axa.tick_params(labelsize=8)
    axa.legend(fontsize=6.5, frameon=False, loc="upper left", markerscale=1.4)

    # ---- panel b: full forecast inner box ----
    half = 5.0 / RES
    midy, midx = 0.5 * (ty + py), 0.5 * (tx + px)
    y0, y1 = int(midy - half), int(midy + half)
    x0, x1 = int(midx - half), int(midx + half)
    y0 = max(0, y0); x0 = max(0, x0); y1 = min(G, y1); x1 = min(G, x1)
    ext = [-(x1 - x0) / 2 * RES, (x1 - x0) / 2 * RES, -(y1 - y0) / 2 * RES, (y1 - y0) / 2 * RES]
    im = axb.imshow(pred[y0:y1, x0:x1], extent=ext, origin="lower", cmap=TCWIND,
                    vmin=0, vmax=70)
    axb.set_aspect("equal")
    axb.set_xlabel("Longitude offset (deg)"); axb.set_ylabel("Latitude offset (deg)")
    axb.set_title(f"Full forecast, Typhoon {case_name} (cat.5, 6 h)", fontsize=10)
    cb = fig.colorbar(im, ax=axb, shrink=0.85, pad=0.02, extend="both")
    cb.set_label("10-m wind (m s$^{-1}$)", fontsize=8); cb.ax.tick_params(labelsize=7)

    # ---- panel c: radial profiles about t1 ----
    # Each field is centred on its own eyewall peak so a track offset does not smear
    # the structural comparison; this isolates the radial shape (motion removed).
    yy, xx = np.mgrid[0:G, 0:G].astype(np.float32)
    rb = np.linspace(0, 3, 31); rc = 0.5 * (rb[:-1] + rb[1:])

    def prof(w, cy, cx):
        rw = np.hypot(yy - cy, xx - cx) * RES
        out = []
        for a0, b0 in zip(rb[:-1], rb[1:]):
            sel = (rw >= a0) & (rw < b0) & fm
            out.append(np.nanmax(np.where(sel, w, np.nan)))
        return np.array(out)

    def ctr(w):
        return vmax_xy(w, fm)[1:]
    series = [("ERA5", era, "#9e9e9e", "--"), ("Persistence", pers, "#f0a35e", "-"),
              ("Stage A", adv, "#7fb3d5", "-"), ("Full forecast", pred, "#1f4e79", "-"),
              ("HAFS reference f006", tgtf, "#111111", "--")]
    for lab, w, col, ls in series:
        wcy, wcx = ctr(w)
        axc.plot(rc, prof(w, wcy, wcx), ls, color=col, lw=1.8, label=lab)
    axc.set_xlim(0, 3); axc.set_ylim(0, 90)
    axc.set_xlabel("Radius from t1 centre (deg)"); axc.set_ylabel("Swath $V_{\\max}$ (m s$^{-1}$)")
    axc.set_title("Radial profile of the restored eyewall", fontsize=10)
    axc.tick_params(labelsize=8); axc.legend(fontsize=7, frameon=False); axc.grid(alpha=0.3)

    # ---- panel d: inner-core Vmax bars (each field centred on its own peak) ----
    names = ["ERA5", "Persistence", "Stage A", "Full\nforecast", "HAFS\nreference"]

    def core_max(w):
        wcy, wcx = ctr(w)
        rw = np.hypot(yy - wcy, xx - wcx) * RES
        return np.nanmax(np.where(rw < 1.5, w, np.nan))

    vals = [core_max(era), core_max(pers), core_max(adv),
            core_max(pred), core_max(tgtf)]
    cols = ["#9e9e9e", "#f0a35e", "#7fb3d5", "#1f4e79", "#111111"]
    bars = axd.bar(np.arange(5), vals, 0.62, color=cols, edgecolor="white")
    for i, v in enumerate(vals):
        axd.text(i, v + 1.2, f"{v:.0f}", ha="center", fontsize=8)
    axd.set_xticks(np.arange(5)); axd.set_xticklabels(names, fontsize=7.5)
    axd.set_ylabel("Inner-core $V_{\\max}$ (m s$^{-1}$)"); axd.set_ylim(0, 90)
    axd.set_title(f"Eyewall peak; observed 6-h move {case_disp:.0f} km", fontsize=10)
    axd.tick_params(labelsize=8)

    for i, axx in enumerate([axa, axb, axc, axd]):
        axx.text(-0.13, 1.04, "abcd"[i], transform=axx.transAxes,
                 fontsize=13, fontweight="bold", va="top")
    fig.suptitle("A recovered category-five core across the 2024 global season", fontsize=11, y=0.985)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print("wrote", OUT, "| case", case_sid, "move", round(case_disp, 1), "km")
    print("Vmax ERA5/pers/adv/pred/tgt =", [round(v, 1) for v in vals])


if __name__ == "__main__":
    main()
