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

# plot_external_baselines.py — figure contrasting global AI weather models
# (Pangu-Weather, GraphCast, FourCastNet) with AssimCast on the held-out 2024
# strict 6-h-lead subset.
#
# Panel a: peak-wind bias vs observed intensity (all 651 cases).  Every coarse
#          0.25-deg product (reanalysis and the three global AI models) shares
#          one ceiling that worsens with intensity; AssimCast and the
#          high-resolution reference stay near zero.
# Panel b: cat3+ mean peak in the GFS initial analysis (f000) vs the 6-h AI
#          forecast (f006): global models collapse the strong peak already
#          present in their input, while AssimCast retains it.
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "docs", "NC_manuscript", "figures")

C_ERA = "#d1495b"
C_FULL = "#1f4e79"
C_TEAC = "#2c6fbb"
C_PG = "#555555"
C_GC = "#888888"
C_FN = "#b0b0b0"

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "legend.frameon": False, "font.family": "DejaVu Sans"})

d = pd.read_csv(os.path.join(HERE, "results", "external_baselines_2024.csv"),
                keep_default_na=False, dtype={"basin": str})
y = d.ib_wind_mps.values
bins = [18, 33, 42, 49, 58, 70, 85]
cent = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
series = [("era5_vmax", "ERA5", C_ERA, "-", 1.8),
          ("pangu_vmax", "Pangu-Weather", C_PG, "--", 1.6),
          ("graphcast_vmax", "GraphCast", C_GC, "--", 1.6),
          ("fourcastnet_vmax", "FourCastNet", C_FN, "--", 1.6),
          ("assimcast_vmax", "AssimCast", C_FULL, "-", 2.4),
          ("teacher_vmax", "HAFS reference", C_TEAC, "-", 1.8)]

fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))

for col, lab, c, ls, lw in series:
    mb = []
    for i in range(len(bins)-1):
        m = (y >= bins[i]) & (y < bins[i+1])
        mb.append(np.mean(d[col].values[m] - y[m]) if m.sum() >= 5 else np.nan)
    a.plot(cent, mb, ls, color=c, lw=lw, label=lab, marker="o", ms=3)
a.axhline(0, color="k", lw=0.7)
a.axvspan(49.4, 85, color="0.92", alpha=0.6, zorder=0)
a.text(50, 3, "Cat 3+", fontsize=7, color="0.35")
a.set_xlabel("Observed best-track $V_{max}$ (m s$^{-1}$)")
a.set_ylabel("Peak-wind bias (m s$^{-1}$)")
a.set_xlim(25, 68); a.set_ylim(-55, 8)
a.legend(loc="lower left", fontsize=6.6, ncol=1, handlelength=2.2, labelspacing=0.35)
a.text(0.02, 0.96, "a", transform=a.transAxes, fontsize=11, fontweight="bold")

# panel b: f000 initial -> f006 forecast collapse over cat3+
try:
    f = pd.read_csv(os.path.join(HERE, "results", "external_baselines_f000f006_cat3.csv"),
                    keep_default_na=False, dtype={"basin": str})
    fy = f.ib_wind_mps.values
    models = [("pangu", "Pangu", C_PG), ("graphcast", "GraphCast", C_GC),
              ("fourcastnet", "FourCastNet", C_FN)]
    x = np.arange(len(models)); w = 0.36
    f000 = [np.mean(f[f"{t}_f000"].values) for t, _, _ in models]
    f006 = [np.mean(f[f"{t}_vmax"].values) for t, _, _ in models]
    b.bar(x - w/2, f000, w, color=[c for _, _, c in models], alpha=0.45,
          edgecolor="white", label="GFS initial (f000)")
    b.bar(x + w/2, f006, w, color=[c for _, _, c in models],
          edgecolor="white", label="AI forecast (f006)")
    for xi, v0, v6 in zip(x, f000, f006):
        b.text(xi - w/2, v0 + 1.0, f"{v0:.0f}", ha="center", va="bottom", fontsize=6.4, color="0.35")
        b.text(xi + w/2, v6 + 1.0, f"{v6:.0f}", ha="center", va="bottom", fontsize=6.4, color="k")
    obs = np.mean(fy)
    b.axhline(obs, color=C_ERA, lw=1.6, ls="-", label=f"Observed ({obs:.0f})")
    assim = np.mean(f.assimcast_vmax.values)
    b.axhline(assim, color=C_FULL, lw=1.8, ls="-", label=f"AssimCast f006 ({assim:.0f})")
    b.set_xticks(x); b.set_xticklabels([n for _, n, _ in models])
    b.set_ylabel("Mean cat3+ peak wind (m s$^{-1}$)")
    b.set_ylim(0, 80)
    b.legend(loc="upper right", fontsize=6.4, labelspacing=0.3)
    b.text(0.02, 0.96, "b", transform=b.transAxes, fontsize=11, fontweight="bold")
except Exception as e:
    b.text(0.1, 0.5, "f000 data not yet available", fontsize=8)

fig.tight_layout()
out = os.path.join(FIG, "fig_ext_baselines.pdf")
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=220, bbox_inches="tight")
print("saved", out)
