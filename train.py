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

# finetune_v6.py - Form-1 operational downscaling: "ERA5/GFS coarse input learns motion,
# HAFS high-resolution teacher restores structure" (true coarse->fine super-resolution).
#
#   Inputs (v6 data): ERA5 0.25 deg analyses tau-18..tau bilinear-upsampled to the 0.06
#   deg storm window + a 32-deg native coarse environment -- exactly the fields an
#   operational GFS 0.25 analysis provides at inference. HAFS is NEVER an input.
#
# Stage A (motion): CenterHead predicts the f000->f006 vortex displacement from the
#   history-centre sequence; a continuous tanh whole-state displacement (vortex follows
#   the TC, outer environment blends to the environmental flow) backward-warps the
#   COARSE ERA5 tau field with grid_sample -> era_adv. Teacher forcing uses the true
#   displacement during training.
#
# Stage B (super-resolution): the diffusion learns the motion-compensated coarse->fine
#   increment x0 = HAFS_f006 - era_adv: the eyewall, RMW, high-wind tail, asymmetry and
#   0.06 sharpness that ERA5 smooths away. This is genuine super-resolution (v5 learned a
#   fine->fine residual that reduced to sub-pixel misalignment and was a negative term).
#   Because the coarse anchor has no eyewall, there is no shifted-eyewall dipole.
#
# Channels stay cond 10: fine(tau-18,-12,-6)=6 (ERA5), advected tau=2 (ERA5), coarse=2.
# Warm-starts from v5d (trained CenterHead + 336-geometry backbone). MPS/CUDA.
import os, sys, time, argparse, glob, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
# model modules are resolved from package "model/" via bootstrap
from hafs_dataset import HAFSCoreDataset
from tc_model import create_tc_model
from diffusion_process import GaussianDiffusion, EMA
from v5_warp import warp_field

ROOT = HERE
DEF_CFG = os.path.join(ROOT, "configs/config_assimcast.yaml")
DEF_V5D = os.path.join(ROOT, "runs/hafs_v5d/checkpoints/best.pt")
DEF_NORM = os.path.join(ROOT, "data/processed/normalizer.json")
# v6 rows align 1:1 with v4 (identical sample selection), so the trak centre table is reused
CENTERS = os.path.join(ROOT, "data/hafs_parent_v4/centers_trak_v5.csv")


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps", torch.bfloat16
    return torch.device("cpu"), "cpu", None


class HAFSV5Dataset(HAFSCoreDataset):
    def __init__(self, *a, center_csv=CENTERS, shift_deg=0.0, rot_aug=False, **kw):
        kw["augment"] = False
        super().__init__(*a, **kw)
        self.G = kw.get("grid_size", 336); self.C = (self.G - 1) / 2.0
        self.shift_px = int(shift_deg / kw.get("ddeg", 0.06))
        self.rot_aug = rot_aug
        cen = pd.read_csv(center_csv, keep_default_na=False)
        self._cen = {}
        for _, r in cen.iterrows():
            if int(r.get("avail", 1)) != 1:
                continue
            self._cen[(r["basin"], int(r["year"]), int(r["row"]))] = r
        self._tag = []
        _pfx = {"v3": "hafs_core_v3_", "v4": "hafs_parent_v4_",
                "v6": "hafs_parent_v6_"}.get(getattr(self, "layout", "v4"),
                                             "hafs_parent_v4_")
        for fi, _ in self.rows:
            tag = os.path.basename(self.files[fi])[len(_pfx):-4]
            self._tag.append(tag)

    def __getitem__(self, i):
        fi, r = self.rows[i]
        tag = self._tag[i]; b, yr = tag.rsplit("_", 1)
        s = super().__getitem__(i)
        c = self._cen.get((b, int(yr), int(r)))
        if c is None:
            return None
        G = self.G
        hy = np.array([float(c[f"hy{k}"]) for k in range(4)], np.float32)
        hx = np.array([float(c[f"hx{k}"]) for k in range(4)], np.float32)
        ty, tx = float(c["ty"]), float(c["tx"])
        hc = np.stack([(hy - self.C) / G, (hx - self.C) / G], 1)          # (4,2)
        dv = np.array([(ty - hy[3]) / G, (tx - hx[3]) / G], np.float32)  # target disp
        motion = np.array([(hy[3] - hy[2]) / G, (hx[3] - hx[2]) / G], np.float32)
        s["hist_centers"] = torch.from_numpy(hc).float()
        s["target_center"] = torch.from_numpy(dv).float()
        s["motion"] = torch.from_numpy(motion).float()
        s["cy"] = torch.tensor(float(hy[3] - self.C))
        s["cx"] = torch.tensor(float(hx[3] - self.C))
        # random rigid translation (storm not locked to centre): shift fields + centres
        if self.shift_px > 0:
            sy = int(np.random.randint(-self.shift_px, self.shift_px + 1))
            sx = int(np.random.randint(-self.shift_px, self.shift_px + 1))
            for k in ("history", "target", "mask", "w"):
                s[k] = torch.roll(s[k], (sy, sx), (-2, -1))
            s["cy"] = s["cy"] + sy; s["cx"] = s["cx"] + sx
        return s


def collate(bs):
    bs = [b for b in bs if b is not None]
    keys = bs[0].keys()
    out = {}
    for k in keys:
        if k == "meta":
            out[k] = [b[k] for b in bs]
        else:
            out[k] = torch.stack([b[k] for b in bs], 0)
    return out


def sobel(x):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device)
    ky = kx.T
    kx = kx[None, None]; ky = ky[None, None]
    C = x.shape[1]
    gx = torch.nn.functional.conv2d(x.reshape(-1, 1, x.shape[-2], x.shape[-1]),
                                    kx, padding=1).reshape(x.shape[0], C, x.shape[-2], -1)
    gy = torch.nn.functional.conv2d(x.reshape(-1, 1, x.shape[-2], x.shape[-1]),
                                    ky, padding=1).reshape(x.shape[0], C, x.shape[-2], -1)
    return torch.hypot(gx, gy)


def radial_psd(x, m, nbin=32):
    # azimuthally averaged power spectrum per sample (vectorized scatter_add)
    B, C, H, W = x.shape
    X = x * m
    F = torch.fft.fftshift(torch.fft.fft2(X, norm='ortho'), dim=(-2, -1))
    P = (F.abs() ** 2).mean(1)
    cy, cx = H // 2, W // 2
    ry = (torch.arange(H, device=x.device) - cy).float()
    rx = (torch.arange(W, device=x.device) - cx).float()
    r = torch.hypot(ry[:, None], rx[None, :])
    rmax = min(cy, cx)
    rb = (r / rmax * nbin).long().clamp_(0, nbin - 1).reshape(-1)
    mm = (m[:, 0] > 0.5).float()
    prof = x.new_zeros(B, nbin)
    for b in range(B):
        pf = P[b].reshape(-1); mf = mm[b].reshape(-1)
        num = torch.zeros(nbin, device=x.device).scatter_add(0, rb, pf * mf)
        den = torch.zeros(nbin, device=x.device).scatter_add(0, rb, mf)
        prof[b] = num / den.clamp_min(1e-8)
    return prof

def init_model(model, ckpt_path, log):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    miss, unexp = model.load_state_dict(sd, strict=False)
    log(f"[INIT] v5d warm-start {os.path.basename(ckpt_path)} missing={len(miss)} "
        f"unexpected={len(unexp)}")
    for m in miss:
        if "center_head" not in m:
            log("   non-center missing: " + m)
    return ck


def run_epoch(model, diff, loader, dev, dt, amp, opt=None, ddeg=0.06, R=3.0, s=1.5,
              w_track=1.0, w_grad=0.0, w_spec=0.0, mu=None, sd=None, t_frac=0.7,
              ema=None, sched=None, pred_center=False, clamp=0.5,
              w_hi=0.0, w_peak=0.0, focus_R=3.5, focus_s=1.0, w_out=0.1,
              core_r=1.5, z_hi=8.0, peak_tau=1.5, resid_clip=3.5, under_pen=2.0,
              ablate_coarse=False):
    train = opt is not None
    model.train(train)
    agg = dict(loss=0, track=0, diff=0, grad=0, spec=0, core=0, peak=0, hi=0, n=0)
    grid = None
    for b in loader:
        hist = b["history"].to(dev); tgt = b["target"].to(dev); wmap = b["w"].to(dev)
        mask = b["mask"].to(dev)
        hc = b["hist_centers"].to(dev); dvt = b["target_center"].to(dev)
        mot = b["motion"].to(dev); cy = b["cy"].to(dev); cx = b["cx"].to(dev)
        B, _, H, _ = tgt.shape
        if grid is None:
            ax = torch.arange(H, device=dev) - (H - 1) / 2.0
            gy0, gx0 = torch.meshgrid(ax, ax, indexing="ij")
            grid = (gy0[None], gx0[None])
        gy, gx = grid
        cyp = cy[:, None, None]   # tau centre offset from grid centre (pixels)
        cxp = cx[:, None, None]
        rc = torch.hypot(gy - cyp, gx - cxp) * ddeg                       # radius from tau centre (deg)
        # inner-core focus: 1 inside focus_R, smooth decay to w_out outside
        focus = w_out + (1.0 - w_out) * torch.exp(-torch.relu((rc - focus_R) / focus_s) ** 2)
        focus = focus[:, None]
        anchor = hist[:, 6:8]
        # Stage A: predicted vs true displacement (teacher forcing warps with truth)
        dvp = model.predict_center(hc, mot)
        track_l = nn.functional.smooth_l1_loss(dvp, dvt)
        dv_use = dvp if pred_center else dvt
        dvy = dv_use[:, 0] * H; dvx = dv_use[:, 1] * H
        anc_adv = warp_field(anchor, dvy, dvx, cy, cx, ddeg=ddeg, R_deg=R, s_deg=s)
        # Stage B: residual diffusion
        coarse_in = hist[:, 8:10]
        if ablate_coarse:
            coarse_in = torch.zeros_like(coarse_in)  # ablation: drop 32-deg env channel
        cond = torch.cat([hist[:, 0:6], anc_adv, coarse_in], 1)
        x0 = tgt - anc_adv
        if train:
            t = torch.randint(0, max(1, int(diff.timesteps * t_frac)), (B,), device=dev)
        else:
            t = torch.full((B,), int(diff.timesteps * 0.3), device=dev)
        noise = torch.randn_like(x0)
        if train:
            opt.zero_grad()
        ctx = torch.autocast(device_type=dt, dtype=amp) if amp else torch.enable_grad()
        with ctx:
            xn = diff.q_sample(x0, t, noise)
            eps = model(xn, t, cond)
            res = (eps.float() - noise.float()) ** 2
            ww = (wmap * focus).expand(-1, res.shape[1], -1, -1)
            diff_l = (ww * res).sum() / ww.sum()
            loss = w_track * track_l + diff_l
        # float32 reconstruction + inner-core physical (anti-peak-shaving) losses
        t32 = t.float()
        ra = diff.sqrt_recip_alphas_cumprod[t32.long()][:, None, None, None].float()
        rm1 = diff.sqrt_recipm1_alphas_cumprod[t32.long()][:, None, None, None].float()
        x0p = ra * xn.float() - rm1 * eps.float()
        # clamp the reconstructed residual before any physical / anti-shaving loss: extreme
        # x0p values (large t, strong eyewall) otherwise make the huber/peak losses explode
        # and poison the weights (NaN). The diffusion (eps) loss itself stays unclamped.
        x0p_c = x0p.clamp(-resid_clip, resid_clip)
        x0f = x0.float()
        pred_z = anc_adv.float() + x0p_c
        tgt_z = tgt.float()
        wsp_p = torch.hypot(pred_z[:, 0], pred_z[:, 1])
        wsp_t = torch.hypot(tgt_z[:, 0], tgt_z[:, 1])
        land = mask[:, 0] > 0.5
        mc = (rc < core_r) & land
        # per-sample strength gate: weak storms (low core Vmax) do NOT get structure
        # sharpening (avoids injecting eyewall noise into weak cores); only strong
        # systems (cat1+ in the target) activate the anti-shaving physical losses.
        zlo, zhi = 4.5, 6.0   # ~19 and ~25 m/s core Vmax in normalised units
        core_strength = wsp_t.where(mc, torch.full_like(wsp_t, -1e9)).flatten(1).max(1).values
        gate = ((core_strength - zlo) / (zhi - zlo)).clamp(0.0, 1.0)
        hi = (rc < focus_R) & land & (wsp_t > z_hi)
        # robust Huber errors (not L1) so a single extreme eyewall sample cannot dominate.
        # Asymmetric weighting: the model's failure mode is systematic UNDER-prediction
        # (peak shaving), so under-estimates of high-wind pixels cost (1+under_pen)x more.
        hub = nn.functional.smooth_l1_loss(wsp_p, wsp_t, reduction="none", beta=1.0)
        aw = 1.0 + under_pen * (wsp_p < wsp_t).float()
        hub = hub * aw
        hi_den = hi.flatten(1).sum(1).clamp_min(1.0)
        hi_l = (gate * (hub * hi).flatten(1).sum(1) / hi_den).mean()
        # differentiable inner-core Vmax (softmax-pooled on the target wind)
        wt = torch.where(mc, wsp_t / peak_tau, torch.full_like(wsp_t, -1e9)).flatten(1)
        wpf = wsp_p.flatten(1); wtf = wsp_t.flatten(1)
        a = torch.softmax(wt, 1)
        # peak term also asymmetric: under-predicting the pooled Vmax is penalised harder
        pv = (a * wpf).sum(1); tv = (a * wtf).sum(1)
        peak_h = nn.functional.smooth_l1_loss(pv, tv, reduction="none", beta=1.0)
        peak_h = peak_h * (1.0 + under_pen * (pv < tv).float())
        peak_l = (gate * peak_h).mean()
        if train:
            loss = loss + w_hi * hi_l + w_peak * peak_l
        gl = torch.tensor(0.0, device=dev); spl = torch.tensor(0.0, device=dev)
        if train and (w_grad > 0 or w_spec > 0):
            f2 = focus.expand(-1, 2, -1, -1)
            gw = gate[:, None, None, None]
            if w_grad > 0:
                gl = (gw * f2 * (sobel(x0p_c) - sobel(x0f)).abs()).sum() / \
                     (gw * f2).sum().clamp_min(1.0)
                loss = loss + w_grad * gl
            if w_spec > 0:
                # keep only strong-sample rows for the spectrum penalty
                keep = gate > 0.5
                if keep.any():
                    spl = (radial_psd(x0p_c[keep], mc[:, None][keep])
                           - radial_psd(x0f[keep], mc[:, None][keep])).abs().mean()
                    loss = loss + w_spec * spl
        if train:
            if not torch.isfinite(loss):
                opt.zero_grad()  # skip this non-finite batch without touching the weights
                continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), clamp)
            if torch.isfinite(gn):
                opt.step()
            else:
                opt.zero_grad()
            if sched is not None:
                sched.step()
            if ema is not None:
                ema.update()
        with torch.no_grad():
            # correct selection metric: inner-core reconstructed wind MAE (z units)
            mcf = mc[:, None]
            core_mae = (mcf * (wsp_p - wsp_t).abs()[:, None]).sum() / mcf.sum()
        agg["loss"] += float(loss); agg["track"] += float(track_l)
        agg["diff"] += float(diff_l); agg["grad"] += float(gl)
        agg["spec"] += float(spl); agg["core"] += float(core_mae)
        agg["peak"] += float(peak_l); agg["hi"] += float(hi_l); agg["n"] += 1
    return {k: v / max(agg["n"], 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data/hafs_parent_v6"))
    ap.add_argument("--cfg", default=DEF_CFG); ap.add_argument("--ckpt", default=DEF_V5D)
    ap.add_argument("--norm", default=DEF_NORM); ap.add_argument("--out", default=os.path.join(ROOT, "runs/hafs_v6"))
    ap.add_argument("--batch", type=int, default=1); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--lr_center", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0); ap.add_argument("--amp", default="bf16")
    ap.add_argument("--w_track", type=float, default=20.0); ap.add_argument("--w_grad", type=float, default=0.10)
    ap.add_argument("--w_spec", type=float, default=0.05); ap.add_argument("--R_deg", type=float, default=3.0)
    # Form-1 super-resolution: the coarse->fine increment is the whole signal, so the
    # anti-shaving high-wind / peak / sharpness weights are raised above v5 (final values
    # are set from the residual-scale diagnostic; resid_clip widened for the larger SR gap)
    ap.add_argument("--w_hi", type=float, default=0.80); ap.add_argument("--w_peak", type=float, default=1.00)
    ap.add_argument("--focus_R", type=float, default=3.5); ap.add_argument("--focus_s", type=float, default=1.0)
    ap.add_argument("--w_out", type=float, default=0.1); ap.add_argument("--core_r", type=float, default=1.5)
    ap.add_argument("--z_hi", type=float, default=8.0); ap.add_argument("--peak_tau", type=float, default=1.5)
    ap.add_argument("--under_pen", type=float, default=2.0,
                    help="asymmetric penalty multiplier for under-predicted high winds / Vmax")
    ap.add_argument("--strong_oversample", type=float, default=2.0,
                    help="extra sampling weight for cat3+ (>=96 kt) target samples; 0 disables")
    ap.add_argument("--clip_val", type=float, default=8.0,
                    help="DDIM x0 clamp (sigma) used in validation sampling; matches training scale")
    ap.add_argument("--s_deg", type=float, default=1.5); ap.add_argument("--shift_deg", type=float, default=2.0)
    ap.add_argument("--grid_size", type=int, default=336); ap.add_argument("--ddeg", type=float, default=0.06)
    ap.add_argument("--max_samples", type=int, default=0); ap.add_argument("--t_frac", type=float, default=0.7)
    ap.add_argument("--ema_decay", type=float, default=0.998); ap.add_argument("--pred_center", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--resid_clip", type=float, default=8.0,
                    help="clamp the reconstructed residual used by physical losses (z units); "
                         "cat3+ SR increment z-p99~5.4, p99.9~8-10, so 8 keeps the eyewall")
    ap.add_argument("--ablate_coarse", action="store_true",
                    help="ablation: zero the 32-deg native coarse environment channels (8:10)")
    ap.add_argument("--seed", type=int, default=0,
                    help="base random seed for weight init, shuffling and sampling (all ablations share 0)")
    args = ap.parse_args()

    run = Path(args.out); (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)
    logf = open(run / "logs" / "train_v6.log", "a")
    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    with open(args.cfg) as f:
        config = yaml.safe_load(f)
    config["model"]["cond_channels"] = 10
    dev, dt, bf = pick_device()
    amp = bf if (args.amp == "bf16" and dt != "cpu") else None
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if dt == "mps":
        torch.mps.manual_seed(args.seed)
    elif dt == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    log(f"[DEVICE] {dev} amp={'bf16' if amp else 'no'} G={args.grid_size} seed={args.seed}")

    dkw = dict(normalizer_path=args.norm, layout="v6", grid_size=args.grid_size, ddeg=args.ddeg)
    tr = HAFSV5Dataset(args.data, sets=("train",), shift_deg=args.shift_deg, **dkw)
    va = HAFSV5Dataset(args.data, sets=("val",), shift_deg=0.0, **dkw)
    if args.max_samples:
        tr.rows = tr.rows[:args.max_samples]; tr.metas = tr.metas[:args.max_samples]
        tr._tag = tr._tag[:args.max_samples]; va.rows = va.rows[:max(16, args.max_samples // 8)]
        va.metas = va.metas[:max(16, args.max_samples // 8)]; va._tag = va._tag[:max(16, args.max_samples // 8)]
    log(f"[DATA] train={len(tr)} val={len(va)}")
    n_strong = sum(1 for m in tr.metas if m["vmax_kt"] >= 96.0)
    if args.strong_oversample > 0 and n_strong > 0:
        from torch.utils.data import WeightedRandomSampler
        sw = [args.strong_oversample if m["vmax_kt"] >= 96.0 else 1.0 for m in tr.metas]
        sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
        log(f"[DATA] WeightedRandomSampler: cat3+ n={n_strong}/{len(tr)} weight x{args.strong_oversample}")
        tl = DataLoader(tr, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                        collate_fn=collate, drop_last=True)
    else:
        tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        collate_fn=collate, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate)

    model = create_tc_model(config).to(dev)
    if args.ckpt and os.path.exists(args.ckpt):
        init_model(model, args.ckpt, log)
    else:
        log(f"[INIT] no warm-start checkpoint at {args.ckpt}; training Stage B from Stage A weights")
    diff = GaussianDiffusion(model.backbone, **config["diffusion"]).to(dev)
    ema = EMA(model.backbone, decay=args.ema_decay)
    for p in model.backbone.parameters():
        p.requires_grad = True
    opt = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr},
        {"params": model.center_head.parameters(), "lr": args.lr_center},
    ], weight_decay=0.0)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(tl) * args.epochs))

    best = float("inf")
    for ep in range(args.epochs):
        t0 = time.time()
        kw = dict(ddeg=args.ddeg, R=args.R_deg, s=args.s_deg, w_track=args.w_track,
                  w_grad=args.w_grad, w_spec=args.w_spec, t_frac=args.t_frac,
                  w_hi=args.w_hi, w_peak=args.w_peak, focus_R=args.focus_R,
                  focus_s=args.focus_s, w_out=args.w_out, core_r=args.core_r,
                  z_hi=args.z_hi, peak_tau=args.peak_tau,
                  clamp=args.grad_clip, resid_clip=args.resid_clip,
                  under_pen=args.under_pen, ablate_coarse=args.ablate_coarse)
        a = run_epoch(model, diff, tl, dev, dt, amp, opt, ema=ema, sched=sched,
                      pred_center=False, **kw)
        v = run_epoch(model, diff, vl, dev, dt, amp, None, pred_center=True, **kw)
        # selection: inner-core reconstructed wind MAE (physical, z units); track already converged
        score = v["core"]
        log(f"[EP {ep+1}/{args.epochs}] tr loss {a['loss']:.4f} track {a['track']:.4f} "
            f"diff {a['diff']:.4f} hi {a['hi']:.3f} peak {a['peak']:.3f} grad {a['grad']:.3f} | "
            f"val track {v['track']:.4f} core {v['core']:.3f} hi {v['hi']:.3f} peak {v['peak']:.3f} "
            f"score {score:.3f} ({time.time()-t0:.0f}s)")
        ck = {"epoch": ep, "model_state_dict": model.state_dict(),
              "ema_state_dict": ema.shadow, "val_score": score, "args": vars(args), "config": config}
        torch.save(ck, run / "checkpoints" / "latest.pt")
        torch.save(ck, run / "checkpoints" / f"ep{ep+1}.pt")  # keep every epoch for strong-case selection
        if score < best:
            best = score
            torch.save(ck, run / "checkpoints" / "best.pt")
            log(f"     * best score {score:.4f}")
    log("V6 FORMM-1 DOWNSCALING FINETUNE DONE")


if __name__ == "__main__":
    main()
