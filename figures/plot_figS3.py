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

# plot_figS3.py - SI component-ablation figure.
#  a) IBTrACS cat3+ Vmax bias across ablations
#  b) independent aircraft (SFMR) strong-core >=33 m/s RMSE across ablations
#  c) Stage-A advection ablation (core/outer/all) from warp_ablation.txt
# Missing ablation runs are drawn as gaps; rerun after run_ablations.sh completes.
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ib_cat3_bias(d):
    return None if d is None else d["vmax"]["pred_vmax_cat3+"]["bias"]


_MED_CACHE = {}


def sfmr_strong_median(csv_name, col):
    """Event-level bootstrap median (2000 resamples, seed 0) of the pooled RMSE over
    retrievals >=33 m/s -- the exact paired bootstrap used in Fig.3c and the main text
    (one shared stream, columns drawn in the order pers, adv, pred, tgt), so every
    ablation bar is numerically identical to the Fig.3c bar for the same model."""
    if csv_name not in _MED_CACHE:
        p = os.path.join(R, csv_name)
        if not os.path.exists(p):
            _MED_CACHE[csv_name] = None
        else:
            d = pd.read_csv(p, keep_default_na=False)
            hi = d[d["sfmr"] >= 33].copy().reset_index(drop=True)
            groups = list(hi.groupby("event").groups.values())
            rng = np.random.default_rng(0)
            med = {}
            for c in ("pers", "adv", "pred", "tgt"):
                o = hi[c].values - hi["sfmr"].values
                v = []
                for _ in range(2000):
                    idx = np.concatenate([groups[q] for q in rng.integers(0, len(groups), len(groups))])
                    v.append(np.sqrt(np.mean(o[idx] ** 2)))
                med[c] = float(np.percentile(v, 50))
            _MED_CACHE[csv_name] = med
    return None if _MED_CACHE[csv_name] is None else _MED_CACHE[csv_name][col]


# configs: (label, ibtracs json, sfmr csv, which ibtracs key, which sfmr column)
configs = [
    ("Stage A\nadvection", "ibtracs_v7ep12.json", "sfmr_v7ep12.csv", "adv", "adv"),
    ("ERA5\ntarget", "ibtracs_v6.json", "sfmr_v6.csv", "pred", "pred"),
    ("No struct.\nloss", "ibtracs_v7_noloss.json", "sfmr_v7_noloss.csv", "pred", "pred"),
    ("No env.\nchannel", "ibtracs_v7_nocoarse.json", "sfmr_v7_nocoarse.csv", "pred", "pred"),
    ("Full\nmodel", "ibtracs_v7ep12.json", "sfmr_v7ep12.csv", "pred", "pred"),
    ("HAFS\nreference", "ibtracs_v7ep12.json", "sfmr_v7ep12.csv", "tgt", "tgt"),
]
ib_bias, sf_rmse = [], []
for lbl, fj, sc, ib_which, sf_col in configs:
    fi = load(fj)
    if fi is None:
        ib_bias.append(np.nan)
    else:
        key = {"pred": "pred_vmax_cat3+", "adv": "adv_vmax_cat3+", "tgt": "tgt_vmax_cat3+"}[ib_which]
        ib_bias.append(fi["vmax"][key]["bias"])
    _m = sfmr_strong_median(sc, sf_col)
    sf_rmse.append(np.nan if _m is None else _m)

# Stage-A warp ablation
rows = {}
for line in open(os.path.join(R, "warp_ablation.txt")):
    p = line.split()
    if p and p[0] in ("core", "outer", "all"):
        rows[p[0]] = [float(x) for x in p[1:4]]  # pers, tanh, full

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
colors = ["#999999", "#888888", "#E69F00", "#56B4E9", "#009E73", "#D55E00"]

ax = axes[0]
x = np.arange(len(configs))
bars = ax.bar(x, ib_bias, color=colors, edgecolor="black", lw=0.6, width=0.7)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([c[0] for c in configs], fontsize=7)
ax.set_ylabel("Cat3+ V$_{max}$ bias (m s$^{-1}$)")
ax.set_title("Intensity bias (IBTrACS, n=163)", fontsize=10, fontweight="bold")
for xi, v in zip(x, ib_bias):
    if not np.isnan(v):
        ax.text(xi, v - 1.2, f"{v:.1f}", ha="center", va="top", fontsize=7)
    else:
        ax.bar(xi, -34, width=0.7, color="white", edgecolor="0.5", lw=0.8, hatch="//")
        ax.text(xi, -17, "ablation\nrunning", ha="center", va="center", fontsize=6, color="0.4")
ax.text(0.02, 0.04, "a", transform=ax.transAxes, fontsize=13, fontweight="bold")

ax = axes[1]
bars = ax.bar(x, sf_rmse, color=colors, edgecolor="black", lw=0.6, width=0.7)
ax.set_xticks(x); ax.set_xticklabels([c[0] for c in configs], fontsize=7)
ax.set_ylabel("Strong-core RMSE (m s$^{-1}$, >=33)")
ax.set_title("Independent aircraft SFMR", fontsize=10, fontweight="bold")
for xi, v in zip(x, sf_rmse):
    if not np.isnan(v):
        ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    else:
        ax.bar(xi, 2.0, width=0.7, color="white", edgecolor="0.5", lw=0.8, hatch="//")
        ax.text(xi, 4.5, "ablation\nrunning", ha="center", va="bottom", fontsize=6, color="0.4")
ax.text(0.02, 0.96, "b", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

ax = axes[2]
regs = ["core", "outer", "all"]
w = 0.25
xx = np.arange(len(regs))
wcols = {"persistence": "#888888", "tanh warp": "#56B4E9", "full model": "#009E73"}
for j, (name, ci) in enumerate(zip(["persistence", "tanh warp", "full model"], [0, 1, 2])):
    vals = [rows[r][ci] for r in regs]
    ax.bar(xx + (j - 1) * w, vals, w, label=name, color=list(wcols.values())[j],
           edgecolor="black", lw=0.5)
    for xi, v in zip(xx + (j - 1) * w, vals):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(xx); ax.set_xticklabels(["Inner core", "Outer", "All"])
ax.set_ylabel("RMSE vs HAFS f006 (m s$^{-1}$)")
ax.set_title("Stage-A whole-field advection", fontsize=10, fontweight="bold")
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.text(0.02, 0.96, "c", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

plt.tight_layout()
out = os.path.join(HERE, "docs", "NC_manuscript", "figures", "figS3_component_ablation.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
print("IBTrACS cat3 bias:", [None if np.isnan(v) else round(v, 2) for v in ib_bias])
print("SFMR strong RMSE:", [None if np.isnan(v) else round(v, 2) for v in sf_rmse])
print("saved", out)
