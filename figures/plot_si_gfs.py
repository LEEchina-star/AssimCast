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

"""Supplementary figure: GFS deployment diagnosis and 0.5-deg smoothing fix."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
C_ORANGE = "#E2A36A"; C_NAVY = "#2E4D72"; C_GRAY = "#9AA5B1"; C_TEAL = "#3E7C8C"

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
})

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

# ---------- panel a: input coarse peak, GFS vs ERA5 (lag=0) ----------
cl = pd.read_csv(os.path.join(ROOT, "results/gfs_era5_input_closeness.csv"))
l0 = cl[cl["lag"] == 0].drop_duplicates(subset=["tau"])
ax[0].scatter(l0["era_vmax"], l0["gfs_vmax"], s=28, c=C_TEAL, alpha=0.75,
              edgecolor="white", linewidth=0.4)
lim = [0, 70]
ax[0].plot(lim, lim, color="k", lw=1.0, ls="--", zorder=0)
g = np.poly1d(np.polyfit(l0["era_vmax"], l0["gfs_vmax"], 1))
xx = np.linspace(lim[0], lim[1], 50)
ax[0].plot(xx, g(xx), color=C_ORANGE, lw=1.6)
ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("ERA5 input coarse peak (m s$^{-1}$)")
ax[0].set_ylabel("GFS input coarse peak (m s$^{-1}$)")
ax[0].set_title("The operational input already retains a sharper core", fontsize=11)
ax[0].text(0.05, 0.92, f"mean ERA5 {l0.era_vmax.mean():.1f}\nmean GFS {l0.gfs_vmax.mean():.1f}",
           transform=ax[0].transAxes, va="top", fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6"))

# ---------- panel b: deployment peak bias, all + cat3 ----------
def load(tag):
    return json.load(open(os.path.join(ROOT, f"results/ibtracs_v7gfs{tag}.json")))
raw = load(""); sm = load("sm")
era = json.load(open(os.path.join(ROOT, "results/ibtracs_v7ep12.json")))
groups = ["All cases", "Category 3+"]
def bias(d, key):
    return d["vmax"][f"pred_vmax_{key}"]["bias"]
raw_b = [bias(raw, "all"), bias(raw, "cat3+")]
sm_b = [bias(sm, "all"), bias(sm, "cat3+")]
x = np.arange(2); w = 0.36
b1 = ax[1].bar(x - w/2, raw_b, w, label="GFS raw input", color=C_ORANGE, edgecolor="white")
b2 = ax[1].bar(x + w/2, sm_b, w, label="GFS + 0.5° smoothing", color=C_NAVY, edgecolor="white")
ax[1].axhline(0, color="k", lw=0.9)
ax[1].set_xticks(x); ax[1].set_xticklabels(groups)
ax[1].set_ylabel("Forecast peak-wind bias (m s$^{-1}$)")
ax[1].set_title("Raw input overshoots; smoothing removes it", fontsize=11)
for bars in (b1, b2):
    for r in bars:
        ax[1].text(r.get_x()+r.get_width()/2, r.get_height()+
                   (0.6 if r.get_height() >= 0 else -1.6),
                   f"{r.get_height():+.1f}", ha="center", fontsize=9.5)
ax[1].legend(frameon=False, fontsize=9.5, loc="upper right")

# ---------- panel c: smoothed GFS vs ERA5-driven deployment (cat3) ----------
def vmax_mae(d, key):
    return d["vmax"][f"pred_vmax_{key}"]["mae"]
labels = ["GFS\n+0.5° smooth", "ERA5-driven\n(reanalysis)"]
mae_v = [vmax_mae(sm, "cat3+"), 11.35]
bias_v = [bias(sm, "cat3+"), -9.77]
xx = np.arange(2)
b3 = ax[2].bar(xx-0.2, mae_v, 0.38, label="MAE", color=C_NAVY, edgecolor="white")
b4 = ax[2].bar(xx+0.2, bias_v, 0.38, label="Bias", color=C_GRAY, edgecolor="white")
ax[2].axhline(0, color="k", lw=0.9)
ax[2].set_xticks(xx); ax[2].set_xticklabels(labels)
ax[2].set_ylabel("Category 3+ peak-wind error (m s$^{-1}$)")
ax[2].set_title("Deployment matches the reanalysis-driven model", fontsize=11)
for bars in (b3, b4):
    for r in bars:
        ax[2].text(r.get_x()+r.get_width()/2, r.get_height()-1.6,
                   f"{r.get_height():.1f}", ha="center", va="top", fontsize=9.5, color="white")
ax[2].legend(frameon=False, fontsize=9.5, loc="lower left")

for a, t in zip(ax, "abc"):
    a.text(-0.13, 1.02, t, transform=a.transAxes, fontsize=14, fontweight="bold")

fig.tight_layout()
out = os.path.join(ROOT, "figures/figS1_gfs_deployment.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
print("wrote", out)
