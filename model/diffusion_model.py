"""
TC-WindDiff: Conditional UNet Diffusion Model for 2D Wind Field Prediction

Architecture (v2 - Spatial Concatenation Conditioning):
  - Encoder: 4 downsampling stages with residual blocks + attention
  - Bottleneck: residual blocks with self-attention
  - Decoder: 4 upsampling stages with skip connections
  - Conditioning: history wind fields concatenated with noisy input (spatial conditioning)

Key change from v1:
  - v1 used global average pooling + FiLM (lost all spatial info, conditioning was ineffective)
  - v2 concatenates history wind fields directly with noisy input at channel dimension
    (preserves full spatial structure, model can "see" where the typhoon is)

Inputs:
  - x_t: (B, 2, H, W) — noisy wind field at timestep t
  - t: (B,) — diffusion timestep
  - cond: (B, 8, H, W) — history wind fields (4 steps × 2 channels)

Outputs:
  - epsilon: (B, 2, H, W) — predicted noise
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================
# Building Blocks
# ============================================================

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    return GroupNorm32(32, channels)


class AttentionBlock(nn.Module):
    """Self-attention block for 2D feature maps."""
    
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = normalization(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Attention
        scale = 1.0 / math.sqrt(C // self.num_heads)
        attn = torch.einsum("bnhd,bmhd->bnhm", q, k) * scale
        attn = F.softmax(attn, dim=-1)
        h = torch.einsum("bnhm,bmhd->bnhd", attn, v)
        
        h = h.reshape(B, C, H, W)
        h = self.proj(h)
        return x + h


class ResBlock(nn.Module):
    """Residual block with timestep embedding (no FiLM conditioning in v2)."""
    
    def __init__(self, in_channels, out_channels, time_embed_dim, 
                 cond_channels=None, dropout=0.0, use_attention=False, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_embed = nn.Sequential(
            SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
        
        # Second conv
        self.norm2 = normalization(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        
        # Attention
        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionBlock(out_channels, num_heads)
    
    def forward(self, x, time_embed, cond=None):
        # First conv
        h = self.norm1(x)
        h = SiLU()(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_embed(time_embed)[:, :, None, None]
        h = h + time_emb
        
        # Second conv
        h = self.norm2(h)
        h = SiLU()(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Skip connection
        h = h + self.skip(x)
        
        # Attention
        if self.use_attention:
            h = self.attn(h)
        
        return h


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ============================================================
# SimpleUNet v2: Spatial Concatenation Conditioning
# ============================================================

class SimpleUNet(nn.Module):
    """
    Standard conditional UNet for 2D wind field diffusion (v2).
    
    Key feature: History wind fields are concatenated with noisy input
    at the channel dimension, preserving full spatial structure.
    
    Architecture:
      - Input: concat(x_t (2ch), cond (8ch)) = 10 channels
      - Encoder: each level has num_res_blocks, save last feature as skip, then downsample
      - Bottleneck: 2 residual blocks with attention
      - Decoder: each level upsample first, concat skip, then num_res_blocks
      - Output: 2 channels (predicted noise)
    """
    
    def __init__(self, in_channels=2, out_channels=2, model_channels=64,
                 channel_mult=(1, 2, 4, 8), num_res_blocks=2,
                 attention_resolutions=(16, 8), dropout=0.1,
                 cond_channels=8, time_embed_dim=256):
        super().__init__()
        
        self.in_channels = in_channels  # noise channels (2)
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.time_embed_dim = time_embed_dim
        self.num_res_blocks = num_res_blocks
        self.cond_channels = cond_channels
        
        # Total input channels = noise + condition
        total_in_channels = in_channels + cond_channels
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Input convolution (takes concatenated noise + condition)
        self.input_conv = nn.Conv2d(total_in_channels, model_channels, 3, padding=1)
        
        # Build encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            use_attn = (256 // (2 ** level)) in attention_resolutions
            
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                # cond_channels=None because conditioning is via input concat, not FiLM
                blocks.append(ResBlock(ch, out_ch, time_embed_dim, None, dropout, use_attn))
                ch = out_ch
            self.encoder_blocks.append(blocks)
            
            if level < len(channel_mult) - 1:
                self.encoder_downsamples.append(Downsample(ch))
        
        # Bottleneck
        self.mid1 = ResBlock(ch, ch, time_embed_dim, None, dropout, True)
        self.mid2 = ResBlock(ch, ch, time_embed_dim, None, dropout, False)
        
        # Build decoder (reversed order)
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            use_attn = (256 // (2 ** level)) in attention_resolutions
            
            # First block takes concat(skip, h), rest take just h
            blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                in_ch = ch + out_ch if i == 0 else ch
                blocks.append(ResBlock(in_ch, out_ch, time_embed_dim, None, dropout, use_attn))
                ch = out_ch
            self.decoder_blocks.append(blocks)
            
            if level > 0:
                self.decoder_upsamples.append(Upsample(ch))
        
        # Output
        self.out_norm = normalization(ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, timesteps, cond=None):
        """
        Args:
            x: (B, 2, H, W) — noisy input
            timesteps: (B,) — diffusion timesteps
            cond: (B, 8, H, W) — conditioning (history wind fields)
        
        Returns:
            (B, 2, H, W) — predicted noise
        """
        # Time embedding
        t_emb = timestep_embedding(timesteps, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # Spatial conditioning: concatenate condition with input
        if cond is not None:
            h = torch.cat([x, cond], dim=1)  # (B, 10, H, W)
        else:
            # If no condition, pad with zeros (for unconditional generation)
            zero_cond = torch.zeros(x.shape[0], self.cond_channels, x.shape[2], x.shape[3], 
                                     device=x.device, dtype=x.dtype)
            h = torch.cat([x, zero_cond], dim=1)
        
        # Input conv
        h = self.input_conv(h)
        
        # Encoder: save last feature of each level as skip
        skips = []
        for level, blocks in enumerate(self.encoder_blocks):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)  # Save last feature of this level
            if level < len(self.encoder_downsamples):
                h = self.encoder_downsamples[level](h)
        
        # Bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)
        
        # Decoder: upsample first, then concat skip, then res blocks
        for level, blocks in enumerate(self.decoder_blocks):
            if level > 0:  # First decoder level (bottleneck) doesn't need upsample
                h = self.decoder_upsamples[level - 1](h)
            
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            
            for i, block in enumerate(blocks):
                h = block(h, t_emb)
        
        # Output
        h = self.out_norm(h)
        h = SiLU()(h)
        h = self.out_conv(h)
        
        return h


# ============================================================
# Factory function
# ============================================================

def create_model(config):
    """Create diffusion model from config."""
    model_cfg = config['model']
    
    model = SimpleUNet(
        in_channels=model_cfg['in_channels'],
        out_channels=model_cfg['out_channels'],
        model_channels=model_cfg['model_channels'],
        channel_mult=tuple(model_cfg['channel_mult']),
        num_res_blocks=model_cfg['num_res_blocks'],
        attention_resolutions=tuple(model_cfg['attention_resolutions']),
        dropout=model_cfg['dropout'],
        cond_channels=model_cfg['cond_channels'],
        time_embed_dim=model_cfg['timestep_embed_dim']
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] SimpleUNet v2 (spatial concat) created: {n_params/1e6:.2f}M parameters")
    
    return model


if __name__ == '__main__':
    # Test model
    import yaml
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = create_model(config)
    
    # Test forward pass
    B = 2
    x = torch.randn(B, 2, 256, 256)
    t = torch.randint(0, 1000, (B,))
    cond = torch.randn(B, 8, 256, 256)
    
    with torch.no_grad():
        out = model(x, t, cond)
    
    print(f"\nInput noise: {x.shape}")
    print(f"Condition: {cond.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Test conditioning effectiveness
    with torch.no_grad():
        out_cond = model(x, t, cond)
        out_nocond = model(x, t, None)
    diff = (out_cond - out_nocond).abs().mean().item()
    print(f"\nConditioning effectiveness (有条件vs无条件输出差异): {diff:.6f}")
    print(f"  (v1 was ~0.033, v2 should be much larger)")
