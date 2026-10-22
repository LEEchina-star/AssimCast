"""
tc_model.py — TC-WindDiff dual-head model (Scheme B).

Structure:
  - self.backbone: the existing SimpleUNet (structural diffusion head). UNCHANGED, so
    it can load the existing v2 weights. forward(x,t,cond) -> epsilon_theta.
  - self.center_head: a small MLP that predicts the future typhoon center (grid iy,ix)
    from the historical center sequence + motion vector. Independent of the U-Net.

The two heads share only sample-level context; they are trained jointly but do not
share parameters. This keeps the diffusion backbone intact (reuses v2 weights) while
adding the missing "motion-aware position" signal that fix the smoothed mean-field /
underestimated-eyewall failure.

Usage:
  model = TCWindDiffModel(config)
  eps = model(x_t, t, cond)                    # structural, for DDPM
  center = model.predict_center(hist_centers)  # position, for center head + remap
"""
import torch
import torch.nn as nn
from diffusion_model import SimpleUNet, timestep_embedding


class CenterHead(nn.Module):
    """Predict future center displacement (grid coords) from a history center sequence."""
    def __init__(self, seq_len=4, hidden=64, out=2, use_motion=True):
        super().__init__()
        self.seq_len = seq_len
        self.use_motion = use_motion
        # encode the center sequence (seq_len x 2)
        in_dim = seq_len * 2 + (2 if use_motion else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out),
        )

    def forward(self, hist_centers, motion=None):
        # hist_centers: (B, seq_len, 2); motion: (B,2) optional
        B = hist_centers.shape[0]
        x = hist_centers.reshape(B, -1)
        if self.use_motion and motion is not None:
            x = torch.cat([x, motion], dim=-1)
        return self.net(x)  # (B,2) predicted center displacement (dx,dy)


class TCWindDiffModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        mc = config['model']
        self.backbone = SimpleUNet(
            in_channels=mc['in_channels'], model_channels=mc['model_channels'],
            out_channels=mc['out_channels'], num_res_blocks=mc['num_res_blocks'],
            attention_resolutions=tuple(mc['attention_resolutions']), dropout=mc['dropout'],
            channel_mult=tuple(mc['channel_mult']), cond_channels=mc['cond_channels'],
            time_embed_dim=mc.get('timestep_embed_dim', 256))
        # center head config (defaults if not present)
        ch_cfg = config.get('center_head', {})
        self.seq_len = ch_cfg.get('seq_len', 4)
        self.center_head = CenterHead(seq_len=self.seq_len, hidden=ch_cfg.get('hidden', 64),
                                      out=2, use_motion=ch_cfg.get('use_motion', True))
        self.use_center = ch_cfg.get('enabled', True)

    # ---- structural (DDPM) interface: returns epsilon ----
    def forward(self, x, t, cond):
        return self.backbone(x, t, cond)

    # ---- position interface ----
    def predict_center(self, hist_centers, motion=None):
        return self.center_head(hist_centers, motion)

    def num_structural_params(self):
        return sum(p.numel() for p in self.backbone.parameters())

    def num_center_params(self):
        return sum(p.numel() for p in self.center_head.parameters())


def create_tc_model(config):
    model = TCWindDiffModel(config)
    n = sum(p.numel() for p in model.parameters())
    print(f"[TC-MODEL] backbone={model.num_structural_params()/1e6:.2f}M  "
          f"center_head={model.num_center_params()/1e3:.1f}K  total={n/1e6:.2f}M")
    return model
