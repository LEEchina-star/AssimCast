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

# predict_v6.py - Form-1 operational downscaling over held-out samples (2024).
# Inputs are COARSE ERA5 (GFS-equivalent at deployment): Stage A predicts the TC
# displacement (CenterHead) and backward-warps the coarse ERA5 tau field -> era_adv;
# Stage B DDIM-diffuses the coarse->fine super-resolution increment, added back to
# era_adv inside the inner-core focus weight (outer region stays on the coarse analysis).
# Output per basin: data/hafs_pred_v6/pred_<B>_test.npz field 'f' (M,11,336,336):
#   [0:2]  pred   = era_adv + foc*DDIM increment (full 0.06 6-h forecast)
#   [2:4]  tgt    = HAFS parent f006 (high-res teacher)
#   [4:6]  adv    = coarse ERA5 advected by the predicted displacement only (Stage A)
#   [6:8]  pers   = coarse ERA5 tau, no advection (persistence baseline)
#   [8:10] era    = coarse ERA5 tau upsampled to 0.06 (== persistence)
#   [10]   mask
import os, sys, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
# model modules are resolved from package "model/" via bootstrap
from train import HAFSV5Dataset, collate
from tc_model import create_tc_model
from diffusion_process import GaussianDiffusion
from v5_warp import warp_field, vortex_weight
ROOT = HERE
KT2MS = 0.5144444


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps", torch.bfloat16
    return torch.device("cpu"), "cpu", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data/hafs_parent_v6"))
    ap.add_argument("--centers", default="",
                    help="override Stage A centres csv (defaults to v6 trak centres)")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs/hafs_v6/checkpoints/best.pt"))
    ap.add_argument("--norm", default=os.path.join(ROOT, "data/processed/normalizer.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/hafs_pred_v6"))
    ap.add_argument("--sets", default="test"); ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=50); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clip_val", type=float, default=8.0,
                    help="DDIM x0 clamp in sigma (v7 trains/ samples at 8; v6 used 3)")
    ap.add_argument("--grid_size", type=int, default=336); ap.add_argument("--ddeg", type=float, default=0.06)
    ap.add_argument("--R_deg", type=float, default=3.0); ap.add_argument("--s_deg", type=float, default=1.5)
    ap.add_argument("--true_center", action="store_true", help="use true trak displacement (oracle)")
    ap.add_argument("--gate_vmax", type=float, default=25.0,
                    help="only super-resolve when the operational f000 trak Vmax (m/s) "
                         "exceeds this; the coarse advected core is smoothed so it cannot "
                         "gate on its own eyewall (unlike v5). Weak systems keep pred=adv.")
    ap.add_argument("--warp_mode", default="tanh", choices=["tanh", "full"],
                    help="Stage-A advection of the coarse field: tanh=vortex moves, outer "
                         "blends to near-stationary (default); full=rigid whole-field shift")
    ap.add_argument("--ablate_coarse", action="store_true",
                    help="ablation: zero the 32-deg native coarse environment channels (8:10)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev, dt, bf = pick_device()
    ds_kw = dict(sets=tuple(args.sets.split(",")), shift_deg=0.0,
                 normalizer_path=args.norm, layout="v6",
                 grid_size=args.grid_size, ddeg=args.ddeg)
    if args.centers:
        ds_kw["center_csv"] = args.centers
    ds = HAFSV5Dataset(args.data, **ds_kw)
    if args.limit:
        ds.rows = ds.rows[:args.limit]; ds.metas = ds.metas[:args.limit]; ds._tag = ds._tag[:args.limit]
    G = ds.grid_size
    print(f"[DATA] {len(ds)} samples ({args.sets}) grid={G} on {dev}", flush=True)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = create_tc_model(ck["config"]).to(dev)
    model.load_state_dict(ck["model_state_dict"])  # full model incl. trained CenterHead
    esd = ck.get("ema_state_dict")
    if esd is not None:
        model.backbone.load_state_dict(esd, strict=False)
        print("[CKPT] full model + EMA backbone")
    else:
        print("[CKPT] model_state_dict")
    model.eval()
    diff = GaussianDiffusion(model.backbone, **ck["config"]["diffusion"]).to(dev)
    mu = torch.tensor(ds.mu, device=dev).view(1, 2, 1, 1)
    sd = torch.tensor(ds.sd, device=dev).view(1, 2, 1, 1)

    buckets = {}; gi = 0

    torch.manual_seed(0)
    with torch.no_grad():
        for bi, b in enumerate(loader):
            hist = b["history"].to(dev); B = hist.shape[0]
            hc = b["hist_centers"].to(dev); mot = b["motion"].to(dev)
            cy = b["cy"].to(dev); cx = b["cx"].to(dev); dvt = b["target_center"].to(dev)
            dvp = model.predict_center(hc, mot)
            dv = dvt if args.true_center else dvp
            dvy = dv[:, 0] * G; dvx = dv[:, 1] * G
            # 6-h track displacement in km (predicted vs true centre displacement)
            km_per_pix = args.ddeg * 111.32
            trk_err_km = (torch.hypot((dvp[:, 0] - dvt[:, 0]),
                                      (dvp[:, 1] - dvt[:, 1])) * G * km_per_pix).detach().cpu().numpy()
            pred_disp_km = (torch.hypot(dvp[:, 0], dvp[:, 1]) * G * km_per_pix).detach().cpu().numpy()
            true_disp_km = (torch.hypot(dvt[:, 0], dvt[:, 1]) * G * km_per_pix).detach().cpu().numpy()
            anchor = hist[:, 6:8]
            if args.warp_mode == "full":
                adv = warp_field(anchor, dvy, dvx, cy, cx, ddeg=args.ddeg,
                                 R_deg=1e3, s_deg=args.s_deg)
            else:  # tanh: whole-state advection; vortex moves, outer blends to near-stationary
                adv = warp_field(anchor, dvy, dvx, cy, cx, ddeg=args.ddeg,
                                 R_deg=args.R_deg, s_deg=args.s_deg)
            coarse_in = hist[:, 8:10]
            if args.ablate_coarse:
                coarse_in = torch.zeros_like(coarse_in)
            cond = torch.cat([hist[:, 0:6], adv, coarse_in], 1)
            # Form-1 strength gate uses the operational f000 trak Vmax: the coarse advected
            # core is smoothed and cannot gate on its own eyewall (the v5 gate). Only strong
            # systems run DDIM super-resolution; weak systems keep the coarse advected field.
            dz = torch.zeros(B, 2, G, G, device=dev)
            ctx = torch.autocast(device_type=dt, dtype=bf) if bf else torch.enable_grad()
            for k in range(B):
                vmax_ms = float(ds.metas[gi + k].get("vmax_kt", 0.0)) * KT2MS
                if vmax_ms >= args.gate_vmax:
                    with ctx:
                        dz[k:k+1] = diff.ddim_sample(
                            (1, 2, G, G), cond=cond[k:k+1],
                            num_steps=args.steps, eta=0.0,
                            clip_val=args.clip_val).float()
            tgt_z = b["target"].to(dev); era_z = b["era_ref"].to(dev)
            foc = vortex_weight(cy, cx, G, args.ddeg, R_deg=3.5, s_deg=1.0, device=dev)[:, None]
            pred = ((adv + foc * dz) * sd + mu).cpu().numpy().astype(np.float16)
            tgt = (tgt_z * sd + mu).cpu().numpy().astype(np.float16)
            advp = (adv * sd + mu).cpu().numpy().astype(np.float16)
            pers = (anchor * sd + mu).cpu().numpy().astype(np.float16)
            era = (era_z * sd + mu).cpu().numpy().astype(np.float16)
            msk = b["mask"].numpy().astype(np.float16)
            f = np.concatenate([pred, tgt, advp, pers, era, msk], axis=1)
            for k in range(B):
                mt = ds.metas[gi]
                bk = buckets.setdefault(mt["basin"], {"f": [], "m": []})
                rec = dict(basin=mt["basin"], sid=mt["sid"], tau=mt["tau"],
                           clat=mt["clat"], clon=mt["clon"],
                           vmax_kt=float(mt.get("vmax_kt", 0.0)),
                           track_err_km=float(trk_err_km[k]),
                           pred_disp_km=float(pred_disp_km[k]),
                           true_disp_km=float(true_disp_km[k]))
                bk["f"].append(f[k]); bk["m"].append(rec); gi += 1
            if (bi + 1) % 20 == 0:
                print(f"  {gi}/{len(ds)}", flush=True)

    tag = args.sets.replace(",", "-") + ("_truec" if args.true_center else "") + \
        (f"_{args.warp_mode}" if args.warp_mode != "tanh" else "")
    tot = 0
    for basin, bk in sorted(buckets.items()):
        F = np.stack(bk["f"], 0)
        np.savez(os.path.join(args.out, f"pred_{basin}_{tag}.npz"), f=F)
        pd.DataFrame(bk["m"]).to_csv(os.path.join(args.out, f"meta_{basin}_{tag}.csv"), index=False)
        print(f"{basin}: {F.shape[0]} -> {F.nbytes/1e6:.0f} MB"); tot += F.shape[0]
    print(f"PREDICT V6 (FORM-1 DOWNSCALING) DONE total={tot}")


if __name__ == "__main__":
    main()
