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

"""Supplementary figure (global 2024): leakage-free assimilation robustness.
(a) strict +/-30 min future window (overall); (b) cross-source verification vs all.
Reads results/da_future_obs_global.json."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
C_ORANGE = "#E2A36A"; C_NAVY = "#2E4D72"

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
})

d = json.load(open(os.path.join(ROOT, "results/da_future_obs_global.json")))
all3 = d["causal_3h"]; strict = d["strict_30min"]; cross = d["cross_source"]
w = 0.38

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3))

# ---- panel a: strict +/-30 min, overall (no per-bin data in global strict subset) ----
seg = strict
noda = seg["no-DA"]["rmse"]; da = seg["DA"]["rmse"]
ci = seg["RMSE_noda_minus_DA_CI"]; el = seg["event_level"]
x = np.array([0])
ax[0].bar(x - w/2, [noda], w, label="No DA", color=C_ORANGE, edgecolor="white")
ax[0].bar(x + w/2, [da], w, label="DA", color=C_NAVY, edgecolor="white")
ax[0].errorbar(x + w/2, [da], yerr=[[0], [0]], fmt="none")  # placeholder
ax[0].text(x[0] - w/2, noda + 0.12, f"{noda:.2f}", ha="center", fontsize=10)
ax[0].text(x[0] + w/2, da + 0.12, f"{da:.2f}", ha="center", fontsize=10)
ax[0].annotate("", xy=(x[0] + w/2, da + 0.35), xytext=(x[0] - w/2, noda + 0.35),
               arrowprops=dict(arrowstyle="->", color="0.25", lw=1.1))
ax[0].set_xticks(x)
ax[0].set_xticklabels(["Future overpass within\n$\\pm$30 min of forecast"])
ax[0].set_ylim(0, max(noda, da) * 1.32)
ax[0].set_ylabel("RMSE vs independent future overpass (m s$^{-1}$)")
ax[0].set_title(f"Strict $\\pm$30 min window: {el['n_events']} events / {seg['n_points']:,} pts",
                fontsize=11)
ax[0].legend(frameon=False, fontsize=9.5, loc="upper right")
ax[0].text(0.03, 0.97,
           f"Paired gain 95% CI {ci[0]:.2f}--{ci[1]:.2f}\n"
           f"{el['events_DA_helps']} of {el['n_events']} events improve\n"
           f"two-sided sign test $p={el['sign_test_p']:.1e}$",
           transform=ax[0].transAxes, ha="left", va="top", fontsize=9.2,
           bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6"))

# ---- panel b: cross-source verification vs all ----
groups = ["All future overpasses\n($\\pm$3 h)", "Verification sensor\n$\\neq$ any assimilated"]
segs = [all3, cross]
noda2 = [s["no-DA"]["rmse"] for s in segs]
da2 = [s["DA"]["rmse"] for s in segs]
ns2 = [s["n_points"] for s in segs]
x2 = np.arange(2)
ax[1].bar(x2 - w/2, noda2, w, label="No DA", color=C_ORANGE, edgecolor="white")
ax[1].bar(x2 + w/2, da2, w, label="DA", color=C_NAVY, edgecolor="white")
for i, s in enumerate(segs):
    e = s["event_level"]
    ax[1].text(x2[i] - w/2, noda2[i] + 0.12, f"{noda2[i]:.2f}", ha="center", fontsize=10)
    ax[1].text(x2[i] + w/2, da2[i] + 0.12, f"{da2[i]:.2f}", ha="center", fontsize=10)
    ax[1].text(x2[i], max(noda2[i], da2[i]) + 0.75,
               f"{s['n_points']:,} pts, {e['events_DA_helps']}/{e['n_events']} events",
               ha="center", fontsize=8.8, color="0.3")
    ax[1].annotate("", xy=(x2[i] + w/2, da2[i] + 0.4), xytext=(x2[i] - w/2, noda2[i] + 0.4),
                   arrowprops=dict(arrowstyle="->", color="0.25", lw=1.1))
ax[1].set_xticks(x2); ax[1].set_xticklabels(groups)
ax[1].set_ylim(0, max(noda2) * 1.28)
ax[1].set_ylabel("RMSE vs future overpass (m s$^{-1}$)")
ax[1].set_title("The gain survives cross-source verification", fontsize=11)
ax[1].legend(frameon=False, fontsize=9.5, loc="upper right")
ax[1].text(0.03, 0.97,
           f"Event-level sign test:\n"
           f"$p={all3['event_level']['sign_test_p']:.1e}$ (all)\n"
           f"$p={cross['event_level']['sign_test_p']:.1e}$ (cross-source)",
           transform=ax[1].transAxes, va="top", ha="left", fontsize=9.2,
           bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6"))

for a, t in zip(ax, "ab"):
    a.text(-0.12, 1.02, t, transform=a.transAxes, fontsize=14, fontweight="bold")

fig.tight_layout()
out = os.path.join(ROOT, "figures/figS2_da_robustness_global.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
print("wrote", out)
