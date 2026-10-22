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

"""plot_fig2_gate_v2.py - extend Fig.2 (the reference gate) with
(c) high-wind wind-speed PDF and (d) azimuthal kinetic-energy spectrum.
a/b reproduce the independent best-track gate; c/d are computed on the
held-out 2024 HAFS-parent fields (target HAFS f006 vs ERA5 reference).
"""
import os, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(HERE, "results", "parent_f006_gate.csv")
OUT = os.path.join(HERE, "figures", "fig2_gate.pdf")
C_ERA = "#d1495b"; C_HAFS = "#2c6fbb"
DX = 0.06 * 111.0          # km per fine cell (~6.66 km)
RMAX = 4.0                 # deg around centre for PDF/spectrum


def stat(x, o):
    g = np.isfinite(x) & np.isfinite(o)
    x, o = x[g], o[g]
    return np.mean(x - o), np.mean(np.abs(x - o)), np.corrcoef(x, o)[0, 1], len(x)


def radial_spectrum(u, v, dx):
    """Azimuthal mean kinetic-energy spectrum of a windowed 2D field."""
    ny, nx = u.shape
    yy, xx = np.mgrid[:ny, :nx].astype(np.float64)
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    r = np.sqrt(((xx - cx) / (cx + 1e-9)) ** 2 + ((yy - cy) / (cy + 1e-9)) ** 2)
    hann = np.clip(1.0 - r, 0, 1) ** 0.5
    win = hann[:, :, None]
    vel = np.stack([u, v], axis=-1)
    vel = (vel - vel.mean(axis=(0, 1))) * win
    fv = np.fft.fftshift(np.fft.fft2(vel, axes=(0, 1)), axes=(0, 1))
    psd = np.abs(fv) ** 2
    psd = psd.mean(axis=-1)
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    nb = 70
    kbins = np.linspace(0, K.max(), nb)
    kc = 0.5 * (kbins[:-1] + kbins[1:])
    idx = np.digitize(K, kbins)
    spec = np.array([psd[idx == i].mean() for i in range(1, nb)])
    wave = 1.0 / (kc + 1e-12)
    return wave, spec


def load_strong_fields(limit=40):
    pdfs = {"hafs": [], "era": []}
    specs = []
    rng = np.random.default_rng(7)
    for f in sorted(glob.glob(os.path.join(HERE, "data", "hafs_parent_v6", "*_2024.npy"))):
        base = os.path.basename(f)
        basin = base.split("_")[3]
        a = np.load(f, mmap_mode="r")
        meta = pd.read_csv(f.replace(".npy", "").replace("hafs_parent_v6_", "meta_") + ".csv",
                           keep_default_na=False)
        vk = pd.to_numeric(meta["vmax_kt"], errors="coerce").values
        strong = np.where(vk >= 96)[0]
        if len(strong) == 0:
            continue
        pick = strong if len(strong) <= limit else rng.choice(strong, limit, replace=False)
        N = a.shape[-1]; C = N // 2; rc = int(round(RMAX / 0.06))
        sl = slice(C - rc, C + rc)
        for i in pick:
            x = np.asarray(a[i], dtype=np.float64)
            hu, hv = x[8], x[9]; eu, ev = x[10], x[11]
            hw = np.sqrt(hu ** 2 + hv ** 2)[sl, sl]
            ew = np.sqrt(eu ** 2 + ev ** 2)[sl, sl]
            pdfs["hafs"].append(hw.ravel()); pdfs["era"].append(ew.ravel())
            wh, sh = radial_spectrum(hu[sl, sl], hv[sl, sl], DX)
            we, se = radial_spectrum(eu[sl, sl], ev[sl, sl], DX)
            specs.append((wh, sh, we, se))
    return pdfs, specs


def main():
    d = pd.read_csv(GATE, keep_default_na=False)
    for c in ["ib_wind_mps", "parent_f006_vmax", "era5_vmax"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    strong = d[d.ib_wind_mps >= 33].dropna(subset=["ib_wind_mps", "parent_f006_vmax", "era5_vmax"])

    pdfs, specs = load_strong_fields()
    print("strong fields for PDF/spectrum:", len(specs))

    fig, ax = plt.subplots(1, 4, figsize=(14.6, 3.5))

    # panel a: Vmax scatter
    a = ax[0]; obs = strong.ib_wind_mps.values
    for col, c, lab in [("parent_f006_vmax", C_HAFS, "HAFS f006 (0.06°)"),
                        ("era5_vmax", C_ERA, "ERA5 (0.25°)")]:
        a.scatter(obs, strong[col].values, s=7, alpha=0.35, c=c,
                  edgecolors="none", label=lab)
    a.plot([15, 95], [15, 95], "k-", lw=1, ls="--")
    b6, m6, r6, n6 = stat(strong.parent_f006_vmax.values, obs)
    be, me, re, ne = stat(strong.era5_vmax.values, obs)
    a.text(0.04, 0.96,
           f"HAFS  bias {b6:+.1f}  MAE {m6:.1f}  r {r6:.2f}\n"
           f"ERA5  bias {be:+.1f}  MAE {me:.1f}  r {re:.2f}\n"
           f"n = {n6} best-track matches",
           transform=a.transAxes, va="top", ha="left", fontsize=7.5,
           family="monospace", bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                          ec="0.5", alpha=0.85))
    a.set_xlim(20, 95); a.set_ylim(0, 95)
    a.set_xlabel("IBTrACS maximum wind (m s$^{-1}$)")
    a.set_ylabel("Field maximum wind (m s$^{-1}$)")
    a.legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.9)
    a.set_title("Core maximum wind", fontsize=9)
    a.set_aspect("equal", adjustable="box"); a.tick_params(labelsize=8)

    # panel b: bias vs intensity
    b = ax[1]; edges = np.array([0, 17.5, 25, 33, 43, 50, 60, 100]); cen = 0.5 * (edges[:-1] + edges[1:])
    for col, c, lab in [("parent_f006_vmax", C_HAFS, "HAFS f006"),
                        ("era5_vmax", C_ERA, "ERA5")]:
        bv = []; xv = []
        for i in range(len(edges) - 1):
            sel = d[(d.ib_wind_mps >= edges[i]) & (d.ib_wind_mps < edges[i + 1])].dropna(subset=[col])
            if len(sel) >= 8:
                bv.append(np.mean(sel[col] - sel.ib_wind_mps)); xv.append(cen[i])
        b.plot(xv, bv, "o-", c=c, ms=4, lw=1.4, label=lab)
    b.axhline(0, color="k", lw=0.8)
    b.fill_betweenx([-60, 5], 33, 100, color="0.9", alpha=0.5, zorder=0)
    b.set_xlabel("Observed maximum wind (m s$^{-1}$)")
    b.set_ylabel("Bias (model $-$ observed, m s$^{-1}$)")
    b.set_xlim(10, 95); b.set_ylim(-55, 5)
    b.legend(loc="lower left", fontsize=7.5, frameon=True)
    b.set_title("Bias grows with intensity", fontsize=9); b.tick_params(labelsize=8)

    # panel c: wind-speed PDF (high-wind tail)
    c = ax[2]
    hw = np.concatenate(pdfs["hafs"]); ew = np.concatenate(pdfs["era"])
    bins = np.linspace(0, 75, 150)
    c.hist(ew, bins=bins, density=True, histtype="step", color=C_ERA, lw=1.8,
           label=f"ERA5 (0.25°)")
    c.hist(hw, bins=bins, density=True, histtype="step", color=C_HAFS, lw=1.8,
           label=f"HAFS f006 (0.06°)")
    c.axvspan(33, 75, color="0.9", alpha=0.5, zorder=0)
    c.set_yscale("log"); c.set_ylim(1e-4, 1e-1)
    c.set_xlabel("Wind speed (m s$^{-1}$)")
    c.set_ylabel("Probability density")
    c.set_title("High-wind tail is truncated", fontsize=9)
    c.legend(fontsize=7.5, frameon=True, loc="upper right"); c.tick_params(labelsize=8)
    frac_h = (hw >= 33).mean() * 100; frac_e = (ew >= 33).mean() * 100
    c.text(0.50, 0.42, f"fraction $\\geq$33 m/s:\nHAFS {frac_h:.2f}%   ERA5 {frac_e:.2f}%",
           transform=c.transAxes, fontsize=7.5, family="monospace",
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85))

    # panel d: azimuthal kinetic-energy spectrum
    e = ax[3]
    wh = specs[0][0]
    sh = np.median(np.stack([s[1] for s in specs]), axis=0)
    se = np.median(np.stack([s[3] for s in specs]), axis=0)
    m = wh < 1000
    e.loglog(wh[m], sh[m], color=C_HAFS, lw=1.8, label="HAFS f006 (0.06°)")
    e.loglog(wh[m], se[m], color=C_ERA, lw=1.8, label="ERA5 (0.25°)")
    for wl, lab in [(100, "~100 km (ERA5 cutoff)"), (25, "~25 km")]:
        e.axvline(wl, color="0.6", ls=":", lw=1)
    e.set_xlim(1000, 10)
    e.set_xlabel("Horizontal wavelength (km)")
    e.set_ylabel("Kinetic-energy spectral power")
    e.set_title("Energy at inner-core scales", fontsize=9)
    e.legend(fontsize=7.5, frameon=True, loc="lower left"); e.tick_params(labelsize=8)

    for i, t in enumerate("abcd"):
        ax[i].text(-0.16, 1.02, t, transform=ax[i].transAxes,
                   fontsize=12, fontweight="bold", va="top")
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}; fraction>=33 HAFS {frac_h:.3f} ERA5 {frac_e:.3f}")


if __name__ == "__main__":
    main()
