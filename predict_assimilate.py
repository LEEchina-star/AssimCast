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

# predict_v6_da.py - Stage C training-free, footprint-aware assimilation for v6.
#
# Closed-loop, causal (operational) setting:
#   * inputs are coarse ERA5 (the deployment surrogate for GFS); Stage A advects them;
#   * only whole overpasses whose median time lies in [tau-3h, tau] are used (data
#     already downlinked before the forecast starts);
#   * reverse-DDIM is guided at each step by the observation likelihood on the
#     reconstructed ABSOLUTE 0.06 deg field (adv + dz), using instrument-matched
#     Gaussian footprints (SAR ~1 km, SMAP ~36 km, AMSR2 ~25 km) and speed-only H;
#   * no-DA and DA runs share the SAME initial noise x_T (paired), so the gain is
#     attributable solely to the observations.
# Independent verification (never assimilated): IBTrACS Vmax/RMW/R34, coastal buoys,
# and cross-source overpasses. L1 hold-out-overpass / L2 cross-source are selected by
# which sources are passed in --assim.
import os, sys, pickle, argparse, json
import numpy as np
import pandas as pd
import torch

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
# model modules are resolved from package "model/" via bootstrap
from train import HAFSV5Dataset, collate
from tc_model import create_tc_model
from diffusion_process import GaussianDiffusion, _space_timesteps
from v5_warp import warp_field, vortex_weight
from da_v6_footprint import gaussian_footprint_wind
from torch.utils.data import DataLoader

KT2MS = 0.5144444
SIGMA_Y = {"sar": 2.0, "smap": 2.0, "amsr2": 2.5}      # obs-error std (m/s)
FOOT_SIG = {"sar": 0.03, "smap": 0.14, "amsr2": 0.10}  # deg
SRC_ID = {"sar": 0, "smap": 1, "amsr2": 2}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps", torch.bfloat16
    return torch.device("cpu"), "cpu", None


def guided_ddim(diff, shape, cond, anchor, obs, mu, sd, ddeg, steps=50,
               sigma_y=2.0, guidance_below=300, guide=True, x_T=None, half=7,
               clip_val=8.0):
    """Paired DDIM: with or without footprint likelihood guidance. obs dict holds
    iy,ix (N,), ws (N,), sig_grid (N,). Returns dz (B,2,H,W)."""
    dev = next(diff.model.parameters()).device
    B = shape[0]
    x = torch.randn(shape, device=dev) if x_T is None else x_T.clone()
    timesteps = list(reversed(_space_timesteps(diff.timesteps, steps)))
    have_obs = guide and obs is not None and obs["ws"].numel() > 0
    if have_obs:
        iy, ix = obs["iy"], obs["ix"]
        sigg = obs["sig_grid"]; y = obs["ws"]
        sig_y = obs["sigma_y"]
    for i, t in enumerate(timesteps):
        tb = torch.full((B,), t, device=dev, dtype=torch.long)
        if have_obs and t <= guidance_below:
            with torch.enable_grad():
                xg = x.detach().requires_grad_(True)
                eps = diff.model(xg, tb, cond)
                acp = diff.alphas_cumprod[t].view(-1, 1, 1, 1)
                x0 = (xg - torch.sqrt(1 - acp) * eps) / torch.sqrt(acp)
                x0 = x0.clamp(-clip_val, clip_val)
                abs_f = anchor + x0
                Hx = gaussian_footprint_wind(abs_f, iy, ix, sigg, mu.view(2), sd.view(2), half=half)
                r = Hx - y[None, :]
                # MMPS precision weight 1/(sigma_y^2 + posterior variance proxy c_t)
                c_t = float(((1 - acp) / (acp + 1e-8)).reshape(-1)[0])
                w = 1.0 / (sig_y[None, :] ** 2 + c_t)
                loss = 0.5 * (w * r ** 2).sum()
                grad = torch.autograd.grad(loss, xg)[0]
                gn = grad.norm()
                if gn > grad.new_tensor(50.0):
                    grad = grad * (50.0 / gn)
                x = (x - c_t * grad).detach()
        with torch.no_grad():
            eps = diff.model(x, tb, cond)
            acp = diff.alphas_cumprod[t]
            x0 = ((x - torch.sqrt(1 - acp) * eps) / torch.sqrt(acp)).clamp(-clip_val, clip_val)
            if i < len(timesteps) - 1:
                tn = timesteps[i + 1]; acn = diff.alphas_cumprod[tn]
            else:
                acn = torch.tensor(1.0, device=dev)
            dirx = torch.sqrt(1 - acn) * eps
            x = torch.sqrt(acn) * x0 + dirx
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data/hafs_parent_v6"))
    ap.add_argument("--centers", default="",
                    help="override Stage A centres csv (deploy: data/hafs_parent_v6_deploy/centers_deploy.csv)")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs/hafs_v6/checkpoints/best.pt"))
    ap.add_argument("--norm", default=os.path.join(HERE, "data/processed/normalizer.json"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "data/da_v6/causal_manifest.pkl"))
    ap.add_argument("--out", default=os.path.join(HERE, "data/hafs_pred_v6_da"))
    ap.add_argument("--assim", default="sar,smap,amsr2", help="sources to assimilate")
    ap.add_argument("--steps", type=int, default=50); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clip_val", type=float, default=8.0,
                    help="DDIM x0 clamp in sigma; SAME for no-DA and DA so the paired "
                         "gain is not confounded by different clamps")
    ap.add_argument("--r_assim", type=float, default=4.0)
    ap.add_argument("--min_vmax", type=float, default=0.0, help="keep samples with trak Vmax>=kt")
    ap.add_argument("--max_obs", type=int, default=600, help="subsample inner-core points/sample")
    ap.add_argument("--gate_vmax", type=float, default=25.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    assim = set(args.assim.split(",")) if args.assim else set()
    dev, dt, bf = pick_device()

    entries = pickle.load(open(args.manifest, "rb"))
    # group observations by (basin, sid, tau)
    groups = {}
    for e in entries:
        if e["src"] not in assim:
            continue
        key = (e["basin"], e["sid"], str(e["tau"]))
        groups.setdefault(key, []).append(e)
    print(f"[DA] {len(groups)} samples with causal obs from {sorted(assim)}")

    ds_kw = dict(sets=("test",), shift_deg=0.0, normalizer_path=args.norm,
                 layout="v6", grid_size=336, ddeg=0.06)
    if args.centers:
        ds_kw["center_csv"] = args.centers
    ds = HAFSV5Dataset(args.data, **ds_kw)
    # index samples by (basin,sid,tau)
    idx = {}
    for i, mt in enumerate(ds.metas):
        idx[(mt["basin"], mt["sid"], mt["tau"])] = i
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    G = ds.grid_size; ddeg = 0.06; C = (G - 1) / 2.0
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = create_tc_model(ck["config"]).to(dev); model.load_state_dict(ck["model_state_dict"])
    if ck.get("ema_state_dict") is not None:
        model.backbone.load_state_dict(ck["ema_state_dict"], strict=False)
    model.eval()
    diff = GaussianDiffusion(model.backbone, **ck["config"]["diffusion"]).to(dev)
    mu = torch.tensor(ds.mu, device=dev).view(2); sd = torch.tensor(ds.sd, device=dev).view(2)
    mu4 = mu.view(1, 2, 1, 1); sd4 = sd.view(1, 2, 1, 1)

    keys = list(groups.keys())
    if args.min_vmax:
        keys = [k for k in keys if k in idx and float(ds.metas[idx[k]].get("vmax_kt", 0)) >= args.min_vmax]
    if args.limit:
        keys = keys[:args.limit]
    keyset = set(keys)
    print(f"[DA] selected {len(keys)} samples (min_vmax={args.min_vmax} kt)")
    recs = []
    buckets_da = {}; buckets_noda = {}
    gi = 0
    torch.manual_seed(0)
    for bi, b in enumerate(loader):
        mt = ds.metas[gi]
        key = (mt["basin"], mt["sid"], str(mt["tau"]))
        gi += 1
        if key not in keyset:
            continue
        hist = b["history"].to(dev)
        hc = b["hist_centers"].to(dev); mot = b["motion"].to(dev)
        cy = b["cy"].to(dev); cx = b["cx"].to(dev); dvt = b["target_center"].to(dev)
        dvp = model.predict_center(hc, mot)
        with torch.no_grad():
            dvy_p = float(dvp[0, 0].detach()); dvx_p = float(dvp[0, 1].detach())
        dvy = dvp[:, 0] * G; dvx = dvp[:, 1] * G
        anchor = hist[:, 6:8]
        adv = warp_field(anchor, dvy, dvx, cy, cx, ddeg=ddeg, R_deg=3.0, s_deg=1.5)
        cond = torch.cat([hist[:, 0:6], adv, hist[:, 8:10]], 1)

        # collect observations, advect inner-core points with the predicted displacement
        # deploy meta stores clon in 0..360 while manifest obs lon are -180..180;
        # normalize the centre so inner-core distances and grid indices are consistent.
        clon_c = (float(mt["clon"]) + 180.0) % 360.0 - 180.0
        lat_l, lon_l, ws_l, sig_l, sy_l = [], [], [], [], []
        for e in groups[key]:
            qlat = e["lat"].astype(np.float64); qlon = e["lon"].astype(np.float64)
            # local radius to f000 centre (only inner-core points are assimilated)
            ddlon = ((qlon - clon_c + 180.0) % 360.0 - 180.0)
            dd = np.hypot(qlat - mt["clat"], ddlon * np.cos(np.radians(mt["clat"])))
            keep = dd <= args.r_assim
            qlat = qlat[keep]; qlon = qlon[keep]; qws = e["ws"][keep]
            # advance by predicted 6h displacement (vortex moves)
            qlat = qlat + dvy_p * ddeg
            qlon = qlon + dvx_p * ddeg
            iy = (qlat - mt["clat"]) / ddeg + C
            ix = (((qlon - clon_c + 180.0) % 360.0 - 180.0)) / ddeg + C
            ok = (iy >= 7) & (iy < G - 7) & (ix >= 7) & (ix < G - 7)
            iy, ix, qws = iy[ok], ix[ok], qws[ok]
            sig_grid = np.full_like(iy, FOOT_SIG[e["src"]] / ddeg)
            sy = np.full_like(iy, SIGMA_Y[e["src"]])
            lat_l.append(iy.astype(np.float32)); lon_l.append(ix.astype(np.float32))
            ws_l.append(qws.astype(np.float32))
            sig_l.append(sig_grid.astype(np.float32)); sy_l.append(sy.astype(np.float32))
        if not lat_l:
            continue
        iy = torch.tensor(np.concatenate(lat_l), device=dev, dtype=torch.float32)
        ix = torch.tensor(np.concatenate(lon_l), device=dev, dtype=torch.float32)
        ws = torch.tensor(np.concatenate(ws_l), device=dev, dtype=torch.float32)
        sigg = torch.tensor(np.concatenate(sig_l), device=dev, dtype=torch.float32)
        sy = torch.tensor(np.concatenate(sy_l), device=dev, dtype=torch.float32)
        if ws.numel() > args.max_obs:
            sel = torch.randperm(ws.numel(), device=dev)[:args.max_obs]
            iy, ix, ws, sigg, sy = iy[sel], ix[sel], ws[sel], sigg[sel], sy[sel]
        obs = dict(iy=iy, ix=ix, ws=ws, sig_grid=sigg, sigma_y=sy)

        # paired no-DA / DA from the same x_T
        x_T = torch.randn((1, 2, G, G), device=dev)
        with torch.no_grad():
            dz0 = guided_ddim(diff, (1, 2, G, G), cond, adv, None, mu, sd, ddeg,
                              steps=args.steps, guide=False, x_T=x_T, clip_val=args.clip_val)
        dz1 = guided_ddim(diff, (1, 2, G, G), cond, adv, obs, mu, sd, ddeg,
                          steps=args.steps, guide=True, x_T=x_T, clip_val=args.clip_val)
        foc = vortex_weight(cy, cx, G, ddeg, R_deg=3.5, s_deg=1.0, device=dev)[:, None]
        tgt_z = b["target"].to(dev)

        def core_vmax(field_z):
            w = torch.hypot(field_z[0, 0] * sd[0] + mu[0],
                           field_z[0, 1] * sd[1] + mu[1]).detach()
            # cy/cx are offsets from the grid centre, so the centre pixel is C+cy.
            gy = torch.arange(G, device=dev)[:, None] - (C + float(cy[0]))
            gx = torch.arange(G, device=dev)[None, :] - (C + float(cx[0]))
            rc = torch.hypot(gy, gx) * ddeg
            m = b["mask"][0, 0].to(dev) > 0.5
            return float(w.where((rc < 1.5) & m, torch.full_like(w, -1e9)).max())

        v_adv = core_vmax(adv); v_no = core_vmax(adv + foc * dz0)
        v_da = core_vmax(adv + foc * dz1); v_tgt = core_vmax(tgt_z)
        # pack 11-channel fields (same layout as predict_v6) so score_ibtracs can be reused
        def pack(dz):
            pred = ((adv + foc * dz) * sd4 + mu4).detach().cpu().numpy().astype(np.float16)
            return pred
        tgt = (tgt_z * sd4 + mu4).detach().cpu().numpy().astype(np.float16)
        advp = (adv * sd4 + mu4).detach().cpu().numpy().astype(np.float16)
        pers = (anchor * sd4 + mu4).detach().cpu().numpy().astype(np.float16)
        msk = b["mask"].numpy().astype(np.float16)
        fda = np.concatenate([pack(dz1), tgt, advp, pers, pers, msk], axis=1)
        fno = np.concatenate([pack(dz0), tgt, advp, pers, pers, msk], axis=1)
        rec = dict(basin=mt["basin"], sid=mt["sid"], tau=mt["tau"], t1=mt["t1"],
                   clat=mt["clat"], clon=mt["clon"], vmax_kt=float(mt.get("vmax_kt", 0)),
                   n_obs=int(ws.numel()),
                   vmax_adv=v_adv, vmax_noda=v_no, vmax_da=v_da, vmax_tgt=v_tgt)
        recs.append(rec)
        buckets_da.setdefault(mt["basin"], {"f": [], "m": []})
        buckets_noda.setdefault(mt["basin"], {"f": [], "m": []})
        buckets_da[mt["basin"]]["f"].append(fda[0]); buckets_da[mt["basin"]]["m"].append(rec)
        buckets_noda[mt["basin"]]["f"].append(fno[0]); buckets_noda[mt["basin"]]["m"].append(rec)
        if len(recs) % 5 == 0:
            print(f"  {len(recs)}/{len(keys)}  latest adv {v_adv:.1f} noDA {v_no:.1f} "
                  f"DA {v_da:.1f} tgt {v_tgt:.1f}", flush=True)
        if args.limit and len(recs) >= args.limit:
            break

    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(args.out, "da_vmax_summary.csv"), index=False)
    # write DA / noDA field bundles (score_ibtracs_v5.py --sets da / noda)
    for tag, bk in (("da", buckets_da), ("noda", buckets_noda)):
        for basin, b in bk.items():
            np.savez(os.path.join(args.out, f"pred_{basin}_{tag}.npz"), f=np.stack(b["f"], 0))
            pd.DataFrame(b["m"]).to_csv(os.path.join(args.out, f"meta_{basin}_{tag}.csv"),
                                        index=False)
    if len(df):
        print("\n== inner-core Vmax (m/s), mean ==")
        for c in ("vmax_adv", "vmax_noda", "vmax_da", "vmax_tgt"):
            print(f"  {c:10s} {df[c].mean():6.2f}")
        print(f"  DA-noDA delta {(df.vmax_da-df.vmax_noda).mean():+.2f}")
    print("DA predict done:", len(df), "samples ->", args.out)


if __name__ == "__main__":
    main()
