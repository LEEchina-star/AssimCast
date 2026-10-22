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

# plot_figS4.py - SI figure: deterministic DDIM sampler convergence (20/50/100 steps)
# and K=8 stochastic-ensemble spread-skill relation. Reads results/sampling_ensemble_v7.json
# produced by diag_v7_sampling.py on the strongest held-out cat3+ cases.
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
J = json.load(open(os.path.join(HERE, "results", "sampling_ensemble_v7.json")))
steps = J["steps_list"]
det = J["deterministic"]
teacher = J["teacher"]
stageA = J["stageA_advection"]
ens = J["ensemble_mean"]
sp = J["ensemble_spread"]

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
plt.subplots_adjust(wspace=0.42)

# ---- panel a: sampling convergence (Vmax and n50) ----
ax = axes[0]
vmax = [det[str(s)]["vmax_mean"] for s in steps]
n50 = [det[str(s)]["n50_mean"] for s in steps]
ax.plot(steps, vmax, "o-", color="#0072B2", lw=2, ms=6, label="Deterministic forecast")
ax.axhline(teacher["vmax_mean"], color="#D55E00", ls="--", lw=1.8, label="HAFS reference")
ax.axhline(stageA["vmax_mean"], color="#888888", ls=":", lw=1.8, label="Stage A advection")
ax.plot(50, ens["vmax_mean"], "D", color="#009E73", ms=7, label="K=8 ensemble mean")
ax.set_xlabel("DDIM steps")
ax.set_ylabel("Inner-core peak wind speed (m s$^{-1}$)")
ax.set_xticks(steps)
ax.set_ylim(0, max(teacher["vmax_mean"], max(vmax)) * 1.15)
ax2 = ax.twinx()
ax2.plot(steps, n50, "s--", color="#CC79A7", lw=1.5, ms=5, alpha=0.8, label="Pixels >50 m s$^{-1}$")
ax2.set_ylabel("Pixels >50 m s$^{-1}$", color="#CC79A7", fontsize=8.5)
ax2.tick_params(axis="y", colors="#CC79A7", labelsize=8)
ax.set_title("Sampler convergence", fontsize=10, fontweight="bold")
l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, lb1 + lb2, fontsize=6.6, loc="lower right", frameon=False)
ax.text(0.02, 0.98, "a", transform=ax.transAxes, fontsize=13, fontweight="bold",
        va="top", ha="left")

# ---- panel b: spread vs error ----
ax = axes[1]
err = np.array(sp["per_case_error"]); spr = np.array(sp["per_case_spread"])
ax.scatter(err, spr, s=34, color="#0072B2", alpha=0.75, edgecolor="white", lw=0.5, zorder=3)
lim = max(err.max(), spr.max()) * 1.1
ax.plot([0, lim], [0, lim], color="#555555", ls="--", lw=1.2, label="1:1 (reliable)")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Ensemble-mean inner-core error (m s$^{-1}$)")
ax.set_ylabel("Inner-core spread (m s$^{-1}$)")
ax.set_title("K=%d spread-skill (ratio %.2f)" % (J["members"], sp["spread_skill_ratio"]),
             fontsize=10, fontweight="bold")
ax.text(0.04, 0.94, "ratio %.2f (%.0f%% of error)" % (sp["spread_skill_ratio"],
        100 * sp["spread_skill_ratio"]), transform=ax.transAxes, fontsize=8,
        va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#D55E00", alpha=0.9))
ax.legend(fontsize=7, loc="lower right", frameon=False)
ax.set_aspect("equal", adjustable="box")
ax.text(0.02, 0.98, "b", transform=ax.transAxes, fontsize=13, fontweight="bold",
        va="top", ha="left")

plt.tight_layout()
out = os.path.join(HERE, "docs", "NC_manuscript", "figures", "figS4_sampling_ensemble.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
print("saved", out)
