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

# plot_fig5_multiyear.py — multi-year (2024, 2025, 2026) global, leakage-free
# DPS assimilation figure. Concatenates the 2024 score CSV
# (results/da_future_obs_global.csv) with the 2025/2026 deploy CSV
# (results/da_future_obs_deploy.csv) and renders a 2x2 NC-style figure:
#   a) global map of independent future-overpass points, coloured by basin;
#   b) pooled RMSE by year (no DA vs DA), with improved-event fractions;
#   c) RMSE by basin pooled over all years;
#   d) RMSE by verification source, cross-source subset and SAR >=33.
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(FIG, "fig5_da_global.pdf")

d24 = pd.read_csv(os.path.join(HERE, "results", "da_future_obs_global.csv"))
d25 = pd.read_csv(os.path.join(HERE, "results", "da_future_obs_deploy.csv"))
df = pd.concat([d24, d25], ignore_index=True)
fin = np.isfinite(df.y) & np.isfinite(df.da) & np.isfinite(df.noda)
df = df[fin].copy()

C_NO = "#f0a35e"; C_DA = "#1f4e79"
BASIN_ORDER = ["WP", "NA", "SI", "SP", "EP", "NI"]
BASIN_NAME = {"WP": "W. Pacific", "NA": "N. Atlantic", "EP": "E. Pacific",
              "SI": "S. Indian", "SP": "S. Pacific", "NI": "N. Indian"}
SRC_LAB = {"sar": "SAR", "smap": "SMAP", "amsr2": "AMSR2"}
BC = {"WP": "#1f77b4", "NA": "#d62728", "SI": "#2ca02c", "SP": "#9467bd",
      "EP": "#17becf", "NI": "#8c564b"}
YEARS = [2024, 2025, 2026]

def pooled(t):
    return np.sqrt(np.mean((t.noda - t.y) ** 2)), np.sqrt(np.mean((t.da - t.y) ** 2))

def event_stats(t):
    tt = t.copy()
    tt["se_no"] = (tt.noda - tt.y) ** 2; tt["se_da"] = (tt.da - tt.y) ** 2
    ev = tt.groupby(["sid", "t1"]).agg(n=("y", "size"), se_no=("se_no", "sum"),
                                       se_da=("se_da", "sum")).reset_index()
    ev["rn"] = np.sqrt(ev.se_no / ev.n); ev["rd"] = np.sqrt(ev.se_da / ev.n)
    ev["d"] = ev.rn - ev.rd
    nev = len(ev); npos = int((ev.d > 0).sum())
    return nev, npos, binomtest(npos, nev, 0.5).pvalue

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CART = True
except Exception:
    HAS_CART = False

fig = plt.figure(figsize=(13.4, 9.6))
gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.20)

# ---------- panel a: global map ----------
if HAS_CART:
    axm = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=180))
    axm.add_feature(cfeature.LAND, facecolor="#e8e8e8", edgecolor="none", zorder=0)
    axm.add_feature(cfeature.OCEAN, facecolor="#f7f9fc", edgecolor="none", zorder=0)
    axm.coastlines(lw=0.4, color="#999999", zorder=1)
    axm.set_global()
    for b in BASIN_ORDER:
        dd = df[df.basin == b]
        if len(dd) == 0:
            continue
        axm.scatter(dd.lon.values, dd.lat.values, s=3, color=BC[b], alpha=0.4,
                    transform=ccrs.PlateCarree(), zorder=2,
                    label=f"{BASIN_NAME[b]} ({dd.groupby(['sid','t1']).ngroups})")
    axm.set_title("Independent future overpasses, 2024–2026", fontsize=10.5)
    axm.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=4,
               frameon=False, fontsize=8, markerscale=2.5, handletextpad=0.3,
               columnspacing=1.0)
    axm.text(-0.06, 1.06, "a", transform=axm.transAxes, fontsize=14, fontweight="bold")
else:
    axm = fig.add_subplot(gs[0, 0]); axm.text(0.5, 0.5, "cartopy unavailable",
                                              ha="center"); axm.set_axis_off()

# ---------- panel b: pooled RMSE by year ----------
ax = fig.add_subplot(gs[0, 1])
w = 0.38; x = np.arange(len(YEARS))
yno, yda, ylab = [], [], []
for yr in YEARS:
    t = df[df.year == yr]
    rn, rd = pooled(t); yno.append(rn); yda.append(rd)
    nev, npos, p = event_stats(t)
    star = "★" if p < 0.05 else ""
    ylab.append(f"{npos}/{nev}{star}")
hb_no = ax.bar(x - w/2, yno, w, label="No DA", color=C_NO, edgecolor="white")
hb_da = ax.bar(x + w/2, yda, w, label="DA", color=C_DA, edgecolor="white")
for i in range(len(YEARS)):
    ax.text(i, max(yno[i], yda[i]) + 0.18, ylab[i], ha="center", fontsize=8)
    ax.text(i, min(yno[i], yda[i]) - 0.5, f"{yno[i]-yda[i]:.2f}", ha="center",
            fontsize=7.5, color="#333333")
ax.set_xticks(x); ax.set_xticklabels([str(y) for y in YEARS], fontsize=9)
ax.set_xlabel("Verification season")
ax.set_ylabel("RMSE (m s$^{-1}$)")
ax.set_title("Improvement holds across three independent seasons", fontsize=10.5)
ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, max(yno) * 1.18)
ax.text(-0.08, 1.05, "b", transform=ax.transAxes, fontsize=14, fontweight="bold")

# ---------- panel c: RMSE by basin (all years) ----------
ax = fig.add_subplot(gs[1, 0])
bnames = [b for b in BASIN_ORDER if b in set(df.basin)]
x = np.arange(len(bnames))
bno, bda, btext = [], [], []
for b in bnames:
    t = df[df.basin == b]
    rn, rd = pooled(t); bno.append(rn); bda.append(rd)
    nev, npos, p = event_stats(t)
    btext.append(f"{npos}/{nev}{'★' if p < 0.05 else ''}")
ax.bar(x - w/2, bno, w, label="No DA", color=C_NO, edgecolor="white")
ax.bar(x + w/2, bda, w, label="DA", color=C_DA, edgecolor="white")
for i in range(len(bnames)):
    ax.text(i, max(bno[i], bda[i]) + 0.18, btext[i], ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels([BASIN_NAME[b] for b in bnames], fontsize=8.5,
                                    rotation=15)
ax.set_ylabel("RMSE (m s$^{-1}$)")
ax.set_title("Reduction across basins; the East Pacific has three events", fontsize=10.5)
ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, max(bno) * 1.22)
ax.text(-0.08, 1.05, "c", transform=ax.transAxes, fontsize=14, fontweight="bold")

# ---------- panel d: by source / cross-source / SAR high wind ----------
ax = fig.add_subplot(gs[1, 1])
groups = []
for s in ["sar", "smap", "amsr2"]:
    t = df[df.src == s]
    if len(t):
        rn, rd = pooled(t)
        groups.append((SRC_LAB[s], rn, rd, len(t), event_stats(t)[2]))
cs = df[df.cross_source == 1] if "cross_source" in df.columns else df.iloc[0:0]
if len(cs):
    rn, rd = pooled(cs); groups.append(("Cross-source", rn, rd, len(cs), event_stats(cs)[2]))
hi = df[(df.src == "sar") & (df.y >= 33)]
if len(hi):
    rn, rd = pooled(hi); groups.append(("SAR ≥33", rn, rd, len(hi), event_stats(hi)[2]))
x = np.arange(len(groups))
gno = [g[1] for g in groups]; gda = [g[2] for g in groups]
ax.bar(x - w/2, gno, w, label="No DA", color=C_NO, edgecolor="white")
ax.bar(x + w/2, gda, w, label="DA", color=C_DA, edgecolor="white")
for i, (nm, rn, rd, n, p) in enumerate(groups):
    ax.text(i, max(rn, rd) + 0.25, f"n={n:,}{'★' if p < 0.05 else ''}",
            ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
ax.set_ylabel("RMSE (m s$^{-1}$)")
ax.set_title("Independent source and cross-source verification", fontsize=10.5)
ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, max(gno) * 1.22)
ax.text(-0.08, 1.05, "d", transform=ax.transAxes, fontsize=14, fontweight="bold")

fig.legend([hb_no, hb_da], ["No DA", "DA"], loc="lower center", ncol=2,
           frameon=False, fontsize=10, bbox_to_anchor=(0.72, 0.005))

nev_tot, npos_tot, _ = event_stats(df)
fig.suptitle(f"Training-free multi-source assimilation across three seasons "
             f"({len(df):,} independent future points in {nev_tot} cat3+ events; "
             f"{npos_tot}/{nev_tot} events improve)",
             fontsize=13, y=0.975)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
print("wrote", OUT)
# print a compact summary for the text
for yr in YEARS:
    t = df[df.year == yr]; rn, rd = pooled(t); nev, npos, p = event_stats(t)
    print(f"{yr}: n={len(t):,} ev={nev} {rn:.2f}->{rd:.2f} {npos}/{nev} p={p:.2e}")
rn, rd = pooled(df); nev, npos, p = event_stats(df)
print(f"ALL: n={len(df):,} ev={nev} {rn:.2f}->{rd:.2f} {npos}/{nev} p={p:.2e}")
for b in bnames:
    t = df[df.basin == b]; rn, rd = pooled(t); nev, npos, pb = event_stats(t)
    print(f"  basin {b}: n={len(t):,} {rn:.2f}->{rd:.2f} {npos}/{nev} p={pb:.2e}")
for nm, rn, rd, n, p in groups:
    print(f"  grp {nm}: n={n:,} {rn:.2f}->{rd:.2f} p={p:.2e}")
