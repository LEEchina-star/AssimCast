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

# plot_fig3_sfmr.py - independent aircraft (SFMR) verification: the "signature" panel.
# Reads results/sfmr_v5.csv (written by score_pred_v5.py). Decomposes 6-h forecast error
# into motion (persistence f000 -> Stage-A advection) and residual structure
# (advection -> diffusion forecast), with the HAFS teacher f006 (tgt) and ERA5 as refs.
# NC style: vector PDF, square panels, English labels, n annotated, CI shown.
import os, sys, glob, json, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v7ep12")
ap.add_argument("--out", default=os.path.join(HERE, "figures", "fig3_sfmr.pdf"))
ARGS = ap.parse_args()
CSV = os.path.join(HERE, "results", f"sfmr_{ARGS.tag}.csv")
JSN = os.path.join(HERE, "results", f"sfmr_{ARGS.tag}.json")
OUT = ARGS.out

FIELDS = ["era", "pers", "adv", "pred", "tgt"]
LABEL = {"era": "ERA5", "pers": "Persistence (f000)", "adv": "Stage A advection",
         "pred": "Full forecast", "tgt": "HAFS reference (f006)"}
COLOR = {"era": "#9e9e9e", "pers": "#f0a35e", "adv": "#7fb3d5",
         "pred": "#1f4e79", "tgt": "#222222"}


def rmse(g, c):
    e = g[c].values - g["sfmr"].values
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    d = pd.read_csv(CSV)
    n = len(d); nev = d["event"].nunique()
    wlab = ["3-17.5", "17.5-25.7", "25.7-33", "33-50", ">=50"]
    rlab = ["r<1 deg", "1-2 deg", "2-3 deg", "3-5 deg", ">=5 deg"]

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))

    # ---- panel a: RMSE by observed wind bin ----
    a = ax[0]
    nb = len(wlab); width = 0.16
    for k, f in enumerate(FIELDS):
        vals = []
        for b in wlab:
            g = d[d["wbin"] == b]
            vals.append(rmse(g, f) if len(g) >= 10 else np.nan)
        a.bar(np.arange(nb) + (k - 2) * width, vals, width,
              color=COLOR[f], label=LABEL[f], edgecolor="white", lw=0.3)
    a.set_xticks(np.arange(nb))
    a.set_xticklabels([b.replace("-", "\n-\n") if "-" in b and not b.startswith(">") else b for b in wlab],
                      fontsize=7)
    a.set_ylabel("RMSE vs SFMR (m s$^{-1}$)")
    a.set_xlabel("Observed wind (m s$^{-1}$)")
    a.set_title("Pointwise error by wind", fontsize=9)
    a.tick_params(labelsize=8)
    a.legend(fontsize=6.5, frameon=False, loc="upper left")

    # ---- panel b: RMSE by radial distance from centre ----
    b = ax[1]
    for k, f in enumerate(FIELDS):
        vals = []
        for rb in rlab:
            g = d[d["rbin"] == rb]
            vals.append(rmse(g, f) if len(g) >= 10 else np.nan)
        b.plot(np.arange(len(rlab)), vals, "o-", color=COLOR[f], ms=4,
               lw=1.4, label=LABEL[f])
    b.set_xticks(np.arange(len(rlab)))
    b.set_xticklabels(["<1", "1-2", "2-3", "3-5", ">=5"], fontsize=8)
    b.set_xlabel("Distance from centre (degree)")
    b.set_ylabel("RMSE vs SFMR (m s$^{-1}$)")
    b.set_title("Error concentrates in the core", fontsize=9)
    b.tick_params(labelsize=8)
    b.legend(fontsize=6.5, frameon=False, loc="upper left")

    # ---- panel c: strong-core gain decomposition (>=33) with event bootstrap CI ----
    c = ax[2]
    hi = d[d["sfmr"] >= 33].copy().reset_index(drop=True)
    groups = list(hi.groupby("event").groups.values())
    rng = np.random.default_rng(0)

    def boot(col):
        o = hi[col].values - hi["sfmr"].values
        v = []
        for _ in range(2000):
            idx = np.concatenate([groups[p] for p in rng.integers(0, len(groups), len(groups))])
            v.append(np.sqrt(np.mean(o[idx] ** 2)))
        return np.percentile(v, 50), np.percentile(v, 2.5), np.percentile(v, 97.5)

    names = ["pers", "adv", "pred", "tgt"]
    m, lo, hi95 = [], [], []
    for col in names:
        mm, l, u = boot(col); m.append(mm); lo.append(l); hi95.append(u)
    m, lo, hi95 = np.array(m), np.array(lo), np.array(hi95)
    cols = [COLOR[n] for n in names]
    c.bar(np.arange(4), m, 0.6, color=cols, edgecolor="white")
    c.errorbar(np.arange(4), m, yerr=[m - lo, hi95 - m], fmt="none", c="k", lw=1, capsize=3)
    for i, (mm, ll, uu) in enumerate(zip(m, lo, hi95)):
        c.text(i, uu + 0.4, f"{mm:.1f}", ha="center", fontsize=8)
    # motion / residual gain brackets
    c.annotate("", xy=(1, m[1] + 1.2), xytext=(0, m[0] + 1.2),
               arrowprops=dict(arrowstyle="->", color="#555", lw=1))
    c.text(0.5, max(m[0], m[1]) + 2.0, "motion", ha="center", fontsize=7, color="#333")
    c.annotate("", xy=(2, m[2] + 1.2), xytext=(1, m[1] + 1.2),
               arrowprops=dict(arrowstyle="->", color="#555", lw=1))
    c.text(1.5, max(m[1], m[2]) + 2.0, "structure", ha="center", fontsize=7, color="#333")
    c.set_xticks(np.arange(4))
    c.set_xticklabels(["Persistence", "Stage A", "Full\nforecast", "HAFS\nreference"], fontsize=7.5)
    c.set_ylabel("RMSE vs SFMR, wind $\\geq$33 (m s$^{-1}$)")
    c.set_title(f"Strong-core gains (n={len(hi)}, {hi['event'].nunique()} events)", fontsize=9)
    c.tick_params(labelsize=8)
    c.set_ylim(0, max(hi95) * 1.25)

    for i, t in enumerate("abc"):
        ax[i].text(-0.12, 1.03, t, transform=ax[i].transAxes,
                   fontsize=12, fontweight="bold", va="top")
    fig.suptitle(f"Independent aircraft verification: {n:,} SFMR fixes, {nev} events (2024)",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print("wrote", OUT)
    for b in wlab:
        g = d[d["wbin"] == b]
        if len(g) >= 10:
            print(f"  {b:9s} n={len(g):6d} " + " ".join(f"{f}={rmse(g,f):5.2f}" for f in FIELDS))


if __name__ == "__main__":
    main()
