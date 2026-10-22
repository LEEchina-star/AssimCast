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

# plot_figS6_multiyear.py — supplementary multi-year breakdown of the closed-loop
# assimilation test: by year x basin, by year x source, by observed wind bin,
# and the strict half-hour / cross-source subsets. Concatenates 2024 and
# 2025/2026 score CSVs.
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
d24 = pd.read_csv(os.path.join(HERE, "results", "da_future_obs_global.csv"))
d25 = pd.read_csv(os.path.join(HERE, "results", "da_future_obs_deploy.csv"))
df = pd.concat([d24, d25], ignore_index=True)
df = df[np.isfinite(df.y) & np.isfinite(df.da) & np.isfinite(df.noda)].copy()
OUT = os.path.join(HERE, "figures", "figS6_multiyear_da.pdf")

C_NO = "#f0a35e"; C_DA = "#1f4e79"
YEARS = [2024, 2025, 2026]
YCOL = {2024: "#9ec9e3", 2025: "#4c8bb4", 2026: "#1f4e79"}
BASIN_ORDER = ["WP", "SI", "SP", "EP"]
BASIN_NAME = {"WP": "W. Pacific", "SI": "S. Indian", "SP": "S. Pacific", "EP": "E. Pacific"}
SRC_ORDER = ["sar", "smap", "amsr2"]
SRC_LAB = {"sar": "SAR", "smap": "SMAP", "amsr2": "AMSR2"}
BINS = [(3, 17.5), (17.5, 25.7), (25.7, 33), (33, 50), (50, 90)]
BLAB = ["3-17.5", "17.5-25.7", "25.7-33", "33-50", ">=50"]

def rm(t):
    return np.sqrt(np.mean((t.noda - t.y) ** 2)), np.sqrt(np.mean((t.da - t.y) ** 2))

def nev(t):
    return t.groupby(["sid", "t1"]).ngroups

fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
w = 0.13

# ---- a: by year x basin (noDA/DA pairs) ----
ax = axes[0, 0]
nb = len(BASIN_ORDER)
base = np.arange(nb)
for j, yr in enumerate(YEARS):
    off = (j - 1) * 3 * w
    no, da = [], []
    for b in BASIN_ORDER:
        t = df[(df.year == yr) & (df.basin == b)]
        rn, rd = rm(t) if len(t) else (np.nan, np.nan)
        no.append(rn); da.append(rd)
    ax.bar(base + off - w/2, no, w, color=C_NO, alpha=0.35 + 0.25 * j,
           edgecolor="white", lw=0.4)
    ax.bar(base + off + w/2, da, w, color=YCOL[yr], edgecolor="white", lw=0.4,
           label=str(yr))
ax.set_xticks(base); ax.set_xticklabels([BASIN_NAME[b] for b in BASIN_ORDER], fontsize=9)
ax.set_ylabel("RMSE (m s$^{-1}$)")
ax.set_title("RMSE by basin and year (orange, no DA; blue, DA)", fontsize=10.5)
ax.grid(axis="y", alpha=0.3)
ax.legend(frameon=False, fontsize=8.5, title="season", title_fontsize=8.5)
ax.text(-0.07, 1.05, "a", transform=ax.transAxes, fontsize=14, fontweight="bold")

# ---- b: by year x source ----
ax = axes[0, 1]
ns = len(SRC_ORDER); base = np.arange(ns)
for j, yr in enumerate(YEARS):
    off = (j - 1) * 3 * w
    no, da = [], []
    for s in SRC_ORDER:
        t = df[(df.year == yr) & (df.src == s)]
        rn, rd = rm(t) if len(t) else (np.nan, np.nan)
        no.append(rn); da.append(rd)
    ax.bar(base + off - w/2, no, w, color=C_NO, alpha=0.35 + 0.25 * j,
           edgecolor="white", lw=0.4)
    ax.bar(base + off + w/2, da, w, color=YCOL[yr], edgecolor="white", lw=0.4)
ax.set_xticks(base); ax.set_xticklabels([SRC_LAB[s] for s in SRC_ORDER], fontsize=9)
ax.set_ylabel("RMSE (m s$^{-1}$)")
ax.set_title("RMSE by verification source and year", fontsize=10.5)
ax.grid(axis="y", alpha=0.3)
ax.text(-0.07, 1.05, "b", transform=ax.transAxes, fontsize=14, fontweight="bold")

# ---- c: RMSE improvement by wind bin, per year ----
ax = axes[1, 0]
x = np.arange(len(BINS))
for j, yr in enumerate(YEARS):
    dd = df[df.year == yr]
    gains = []
    for lo, hi in BINS:
        t = dd[(dd.y >= lo) & (dd.y < hi)]
        gains.append((t.noda - t.da).apply(lambda _: 0) if False else
                     (np.sqrt(np.mean((t.noda - t.y) ** 2)) -
                      np.sqrt(np.mean((t.da - t.y) ** 2))) if len(t) else np.nan)
    ax.bar(x + (j - 1) * 0.27, gains, 0.25, color=YCOL[yr], label=str(yr),
           edgecolor="white")
ax.axhline(0, color="#333333", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(BLAB, fontsize=8.5, rotation=12)
ax.set_xlabel("Observed wind speed (m s$^{-1}$)")
ax.set_ylabel("RMSE reduction (no DA $-$ DA, m s$^{-1}$)")
ax.set_title("Pointwise gain by observed wind regime", fontsize=10.5)
ax.grid(axis="y", alpha=0.3)
ax.legend(frameon=False, fontsize=8.5)
ax.text(-0.07, 1.05, "c", transform=ax.transAxes, fontsize=14, fontweight="bold")

# ---- d: event-level improvement distribution per year ----
ax = axes[1, 1]
for yr in YEARS:
    dd = df[df.year == yr].copy()
    dd["se_no"] = (dd.noda - dd.y) ** 2; dd["se_da"] = (dd.da - dd.y) ** 2
    ev = dd.groupby(["sid", "t1"]).agg(n=("y", "size"), se_no=("se_no", "sum"),
                                       se_da=("se_da", "sum")).reset_index()
    delta = np.sqrt(ev.se_no / ev.n) - np.sqrt(ev.se_da / ev.n)
    parts = ax.violinplot([delta], positions=[YEARS.index(yr)], widths=0.7,
                          showmeans=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(YCOL[yr]); pc.set_alpha(0.6)
    parts["cmeans"].set_color("k")
    ax.scatter(np.full(len(delta), YEARS.index(yr)) + np.random.default_rng(0).normal(0, 0.04, len(delta)),
               delta, s=5, color=YCOL[yr], alpha=0.5, zorder=2)
ax.axhline(0, color="#333333", lw=0.8)
ax.set_xticks(range(len(YEARS))); ax.set_xticklabels([str(y) for y in YEARS])
ax.set_xlabel("Verification season")
ax.set_ylabel("Event-level RMSE reduction (m s$^{-1}$)")
ax.set_title("Per-event gain; positive means DA improves", fontsize=10.5)
ax.grid(axis="y", alpha=0.3)
ax.text(-0.07, 1.05, "d", transform=ax.transAxes, fontsize=14, fontweight="bold")

fig.suptitle("Supplementary breakdown of the three-year closed-loop assimilation test",
             fontsize=13, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
print("wrote", OUT)
