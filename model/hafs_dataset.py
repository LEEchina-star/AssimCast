#!/usr/bin/env python
# hafs_dataset.py - torch Dataset for the Z1 HAFS inner-core finetuning arrays.
#
# On-disk per sample (build_hafs_core.py), float16, (13,128,128):
#   [0:8]  fine  = ERA5 0.25 bilinear-upsampled to the tau-anchored 0.05 grid,
#                 frames tau-18,-12,-6,tau, u/v interleaved        (deployment input)
#   [8:10] target= HAFS storm nest conservatively remapped to 0.05, tau+6 (teacher)
#   [10:12]coarse= ERA5 0.25 native 32-deg window at tau           (environment)
#   [12]   mask  = 1 native HAFS target cell, 0 coarse-backfilled
#
# Normalization: inputs (fine+coarse) and the residual target all use the ERA5
# z-score so the diffusion noise scale is continuous with pretraining. The model
# predicts the increment delta = (HAFS(t+6) - ERA5_anchor(t)) / ERA5_std, which is
# ~0 in the outer region (anti-forgetting) and large only in the eyewall.
import os, glob, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

N = 128
NC, NT, NCO, NM = 8, 2, 2, 1


class HAFSCoreDataset(Dataset):
    def __init__(self, data_dir, sets=("train",), augment=False,
                 normalizer_path="data/processed/normalizer.json", basins=None,
                 core_gamma=5.0, core_r0_deg=1.5, wind_gamma=30.0,
                 wind_v0=25.0, wind_vs=15.0, wind_pow=2.0, layout="v2",
                 grid_size=None, ddeg=None):
        self.data_dir = data_dir
        self.augment = augment
        self.layout = layout
        # v2 13ch: [fine8(ERA), target2(HAFS f000), coarse2, mask1]
        # v3 15ch: [fine8(storm f000 0.05), target2(storm f006), era_ref2, coarse2, mask1]
        # v4 15ch: [fine8(parent f000 0.06 20deg), target2(parent f006), era_ref2,
        #          coarse2(ERA5 32deg wide), mask1]
        # v6 15ch (Form-1 operational downscaling): [fine8(ERA5 0.25 UPSAMPLED to 0.06
        #          20deg, the GFS-equivalent coarse input), target2(parent f006 teacher),
        #          era_ref2(==fine tau), coarse2(ERA5 32deg), mask1]
        self.i_fine, self.i_tgt, self.i_coa, self.i_msk = 0, 8, 12, 14
        self.i_era = 10 if layout in ("v3", "v4", "v6") else -1
        if grid_size is None:
            grid_size = 336 if layout in ("v4", "v6") else N
        if ddeg is None:
            ddeg = 0.06 if layout in ("v4", "v6") else 0.05
        self.grid_size, self.ddeg = grid_size, ddeg
        self.core_gamma, self.core_r0_deg = core_gamma, core_r0_deg
        self.wind_gamma, self.wind_v0, self.wind_vs, self.wind_pow = (
            wind_gamma, wind_v0, wind_vs, wind_pow)
        _ax = np.arange(grid_size) - (grid_size - 1) / 2.0
        _gy, _gx = np.meshgrid(_ax, _ax, indexing="ij")
        self._rad_deg = np.hypot(_gy, _gx) * ddeg
        if not os.path.isabs(normalizer_path):
            normalizer_path = os.path.join(
                os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__))),
                normalizer_path)
        with open(normalizer_path) as f:
            nz = json.load(f)
        self.mu = np.array([nz["u10"]["mean"], nz["v10"]["mean"]], np.float32)
        self.sd = np.array([nz["u10"]["std"], nz["v10"]["std"]], np.float32)
        self.files, self.rows, self.metas = [], [], []
        self._mm = {}
        prefix = {"v3": "hafs_core_v3_", "v4": "hafs_parent_v4_",
                  "v6": "hafs_parent_v6_"}.get(self.layout, "hafs_core_")
        for fp in sorted(glob.glob(os.path.join(data_dir, prefix + "*.npy"))):
            tag = os.path.basename(fp)[len(prefix):-4]      # BASIN_YEAR
            mp = os.path.join(data_dir, f"meta_{tag}.csv")
            if not os.path.exists(mp):
                continue
            basin = tag.rsplit("_", 1)[0]
            if basins and basin not in basins:
                continue
            df = pd.read_csv(mp, keep_default_na=False)
            if "set" not in df:
                continue
            fi = len(self.files)
            self.files.append(fp)
            for r, row in df.iterrows():
                if row["set"] in sets:
                    self.rows.append((fi, int(r)))
                    self.metas.append(dict(basin=basin, sid=row.get("sid", ""),
                                           tau=str(row.get("tau", "")), t1=str(row.get("t1", "")),
                                           clat=float(row.get("clat")), clon=float(row.get("clon")),
                                           center_src=row.get("center_src", ""),
                                           vmax_kt=float(row.get("vmax_kt", 0) or 0)))

    def __len__(self):
        return len(self.rows)

    def _arr(self, fi, r):
        if fi not in self._mm:
            self._mm[fi] = np.load(self.files[fi], mmap_mode="r")
        return np.asarray(self._mm[fi][r], np.float32)

    def _z(self, x2):  # (2,H,W) u/v physical -> z-score
        return (x2 - self.mu[:, None, None]) / self.sd[:, None, None]

    def __getitem__(self, i):
        fi, r = self.rows[i]
        a = np.nan_to_num(self._arr(fi, r), nan=0.0, posinf=0.0, neginf=0.0)
        fine = a[self.i_fine:self.i_fine + NC]        # (8,H,W) history, physical
        tgt = a[self.i_tgt:self.i_tgt + NT]          # (2,H,W) target, physical
        coarse = a[self.i_coa:self.i_coa + NCO]      # (2,H,W) ERA5 0.25 environment
        mask = a[self.i_msk:self.i_msk + NM]         # (1,H,W)
        era_ref = (a[self.i_era:self.i_era + NT] if self.i_era >= 0
                   else fine[NC - 2:NC])             # ERA5 0.05 ref (v3); v2 = fine tau
        # z-score inputs (u,v pairs share stats; fine is u,v,u,v,...)
        fine_z = np.empty_like(fine)
        for k in range(0, NC, 2):
            fine_z[k:k + 2] = self._z(fine[k:k + 2])
        coarse_z = self._z(coarse)
        anchor_z = fine_z[NC - 2:NC]                   # tau frame on the fine grid
        tgt_z = self._z(tgt)
        era_ref_z = self._z(era_ref)                   # independent ERA5 0.05 reference
        delta_z = tgt_z - anchor_z
        # per-pixel training weight: base 1 everywhere (keeps the outer field learned,
        # preventing catastrophic forgetting), plus a radial inner-core term and a
        # high-wind term; both bonuses apply only on native HAFS nest cells.
        ws_t = np.hypot(tgt[0], tgt[1]).astype(np.float32)
        w_rad = self.core_gamma * np.exp(-(self._rad_deg / self.core_r0_deg) ** 2)
        w_wind = self.wind_gamma * (np.maximum(ws_t - self.wind_v0, 0.0)
                                    / self.wind_vs) ** self.wind_pow
        wmap = (1.0 + mask[0] * (w_rad + w_wind)).astype(np.float32)          # (H,W)
        out = {
            "history": torch.from_numpy(np.concatenate([fine_z, coarse_z], 0)).float(),  # 10
            "target_delta": torch.from_numpy(delta_z).float(),
            "target": torch.from_numpy(tgt_z).float(),
            "anchor": torch.from_numpy(anchor_z).float(),
            "era_ref": torch.from_numpy(era_ref_z).float(),
            "mask": torch.from_numpy(mask).float(),
            "w": torch.from_numpy(wmap)[None].float(),                        # (1,H,W)
            "hist_centers": torch.zeros(4, 2),
            "target_center": torch.zeros(2),
            "motion": torch.zeros(2),
            "meta": self.metas[i],
        }
        if self.augment:
            out = self._augment(out)
        return out

    def _augment(self, s):
        k = int(np.random.randint(0, 4)); flip = bool(np.random.randint(0, 2))
        if k == 0 and not flip:
            return s

        def rot(t):
            r = torch.rot90(t, k, dims=(-2, -1))
            return torch.flip(r, dims=(-1,)) if flip else r

        for key in ("history", "target", "target_delta", "anchor", "mask", "w"):
            s[key] = rot(s[key])
        return s
