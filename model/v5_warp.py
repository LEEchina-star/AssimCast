#!/usr/bin/env python
# v5_warp.py - Stage A whole-state continuous advection (differentiable).
# A continuous displacement field moves the whole state: the vortex gets the predicted
# TC displacement, the field blends smoothly (tanh) to the environmental flow near the
# outer region (which is ~0 over 6 h). No hard mask. Implemented as a backward warp with
# grid_sample so the CenterHead can be trained end-to-end and the warped state feeds the
# residual diffusion.
import torch
import torch.nn.functional as F


def vortex_weight(cy, cx, G, ddeg, R_deg=3.0, s_deg=1.5, device="cpu"):
    """Tanh weight W(r): 1 at the f000 vortex centre -> 0 in the outer environment.
    cy,cx: (B,) f000 centre pixel offset from the grid centre. Returns (B,G,G)."""
    ax = torch.arange(G, device=device, dtype=torch.float32) - (G - 1) / 2.0
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    ry = gy[None] - cy[:, None, None]
    rx = gx[None] - cx[:, None, None]
    rdeg = torch.hypot(ry, rx) * ddeg
    return 0.5 * (1.0 - torch.tanh((rdeg - R_deg) / s_deg))


def warp_field(field, dvy, dvx, cy, cx, ddeg=0.06, R_deg=3.0, s_deg=1.5,
               env_y=None, env_x=None):
    """Backward-warp `field` (B,C,G,G) by a continuous whole-state displacement.

    dvy,dvx: (B,) vortex displacement in pixels (lat, lon), f000 centre -> t1 centre.
    cy,cx:  (B,) f000 centre pixel offset from the grid centre.
    env_y,env_x: optional (B,G,G) environmental displacement fields (pixels); defaults 0.
    Returns warped field (B,C,G,G). Forward displacement moves content to the new centre,
    so the backward sampling grid is coordinate - displacement.
    """
    B, C, G, _ = field.shape
    dev = field.device
    W = vortex_weight(cy, cx, G, ddeg, R_deg, s_deg, device=dev)
    fdy = dvy[:, None, None] * W
    fdx = dvx[:, None, None] * W
    if env_y is not None:
        fdy = fdy + env_y * (1.0 - W)
        fdx = fdx + env_x * (1.0 - W)
    ax = (torch.arange(G, device=dev, dtype=torch.float32) - (G - 1) / 2.0) / ((G - 1) / 2.0)
    ny, nx = torch.meshgrid(ax, ax, indexing="ij")
    ny = ny[None].expand(B, G, G); nx = nx[None].expand(B, G, G)
    fny = 2.0 * fdy / (G - 1); fnx = 2.0 * fdx / (G - 1)
    grid = torch.stack([nx - fnx, ny - fny], dim=-1)
    return F.grid_sample(field, grid, mode="bilinear", padding_mode="reflection",
                         align_corners=True)


def random_shift(batch, keys2d, max_deg=2.0, ddeg=0.06):
    """Training augmentation: translate every 2D field and the recorded centre by the
    same random integer offset so the storm is not locked to the grid centre."""
    b = batch
    G = b["anchor"].shape[-1]
    max_px = int(max_deg / ddeg)
    sy = int(torch.randint(-max_px, max_px + 1, (1,)).item())
    sx = int(torch.randint(-max_px, max_px + 1, (1,)).item())
    for k in keys2d:
        if k in b and torch.is_tensor(b[k]):
            b[k] = torch.roll(b[k], shifts=(sy, sx), dims=(-2, -1))
    for k in ("cy", "cx"):
        if k in b:
            b[k] = b[k] + float(sy if k == "cy" else sx)
    return b
