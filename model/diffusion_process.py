"""
TC-WindDiff: Diffusion Process (DDPM) and SDA Assimilation

Implements:
  1. Gaussian Diffusion (forward/backward process)
  2. Training loss computation
  3. Sampling (DDPM/DDIM)
  4. SDA (Score-based Data Assimilation) with multi-source observations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple
import math


# ============================================================
# Gaussian Diffusion
# ============================================================

def _space_timesteps(num_timesteps, num_steps):
    """openai/guided-diffusion space_timesteps(..., 'ddimN') -- see respace.py."""
    for stride in range(1, num_timesteps):
        if len(range(0, num_timesteps, stride)) == num_steps:
            return list(range(0, num_timesteps, stride))
    idx = np.round(np.linspace(0, num_timesteps - 1, num_steps)).astype(int)
    return sorted(set(int(v) for v in idx))


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion process for 2D wind fields.
    
    Args:
        model: UNet model
        timesteps: Number of diffusion steps
        beta_schedule: 'linear' or 'cosine'
        beta_start: Starting beta value
        beta_end: Ending beta value
        loss_type: 'mse' or 'l1'
        model_mean_type: 'epsilon' or 'sample'
    """
    
    def __init__(self, model, timesteps=1000, beta_schedule='cosine',
                 beta_start=0.0001, beta_end=0.02, loss_type='mse',
                 model_mean_type='epsilon'):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        
        # Register betas
        betas = self._get_beta_schedule(beta_schedule, timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        
        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        self.register_buffer('alphas', torch.tensor(alphas, dtype=torch.float32))
        self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer('alphas_cumprod_prev', torch.tensor(alphas_cumprod_prev, dtype=torch.float32))
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.tensor(np.sqrt(alphas_cumprod), dtype=torch.float32))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.tensor(np.sqrt(1.0 - alphas_cumprod), dtype=torch.float32))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.tensor(np.log(1.0 - alphas_cumprod), dtype=torch.float32))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.tensor(np.sqrt(1.0 / alphas_cumprod), dtype=torch.float32))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.tensor(np.sqrt(1.0 / alphas_cumprod - 1), dtype=torch.float32))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', torch.tensor(posterior_variance, dtype=torch.float32))
        self.register_buffer('posterior_log_variance_clipped', torch.tensor(np.log(np.maximum(posterior_variance, 1e-20)), dtype=torch.float32))
        self.register_buffer('posterior_mean_coef1', torch.tensor(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod), dtype=torch.float32))
        self.register_buffer('posterior_mean_coef2', torch.tensor((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod), dtype=torch.float32))
    
    def _get_beta_schedule(self, schedule, timesteps, beta_start, beta_end):
        if schedule == 'linear':
            return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
        elif schedule == 'cosine':
            s = 0.008
            steps = timesteps + 1
            x = np.linspace(0, timesteps, steps, dtype=np.float64)
            alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return np.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")
    
    def q_sample(self, x_start, t, noise=None):
        """Diffuse the data: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def training_loss(self, x_start, cond=None):
        """
        Compute training loss.
        
        Args:
            x_start: (B, C, H, W) — clean target wind field
            cond: (B, cond_C, H, W) — conditioning (history)
        
        Returns:
            loss: scalar
        """
        B = x_start.shape[0]
        device = x_start.device
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (B,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Get noisy sample
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        noise_pred = self.model(x_noisy, t, cond)
        
        # Compute loss
        if self.loss_type == 'mse':
            loss = F.mse_loss(noise_pred, noise)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(noise_pred, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, x, t, cond=None, clip_denoised=True):
        """Single reverse diffusion step."""
        B = x.shape[0]
        device = x.device
        
        # Predict noise
        noise_pred = self.model(x, t, cond)
        
        # Predict x_0
        x0_pred = self.sqrt_recip_alphas_cumprod[t][:, None, None, None] * x - \
                  self.sqrt_recipm1_alphas_cumprod[t][:, None, None, None] * noise_pred
        
        if clip_denoised:
            x0_pred = x0_pred.clamp(-3, 3)  # Normalized range
        
        # Compute posterior mean
        model_mean = self.posterior_mean_coef1[t][:, None, None, None] * x0_pred + \
                     self.posterior_mean_coef2[t][:, None, None, None] * x
        
        # Compute posterior variance
        model_log_variance = self.posterior_log_variance_clipped[t][:, None, None, None]
        
        # Add noise (except for t=0)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float()[:, None, None, None]
        
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    @torch.no_grad()
    def sample(self, shape, cond=None, clip_denoised=True, progress=False):
        """
        Generate samples from the model.
        
        Args:
            shape: (B, C, H, W)
            cond: conditioning tensor
            clip_denoised: whether to clip denoised samples
            progress: whether to show progress
        
        Returns:
            x_0: generated samples
        """
        device = next(self.model.parameters()).device
        B = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion
        for i in reversed(range(self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, cond, clip_denoised)
            
            if progress and i % 100 == 0:
                print(f"  Sampling step {i}/{self.timesteps}")
        
        return x
    
    @torch.no_grad()
    def _t_from_abar(self, abar):
        """Timestep index whose cumulative alpha is closest to abar."""
        ac = self.alphas_cumprod.cpu().numpy()
        return int(np.clip(np.searchsorted(-ac, -float(abar), side='left'),
                           0, self.timesteps - 1))

    def heun_sample(self, shape, cond=None, num_steps=20, sigma_min=0.02,
                    sigma_max=600.0, rho=7.0, clip_denoised=True, x_init=None):
        """Second-order Heun sampler on the VP probability-flow ODE.

        The sampling timesteps follow a Karras rho=7 sigma grid (Karras et al. 2022), but the
        update itself stays in the variance-preserving parameterisation the model was trained
        in:  abar = 1/(1+sigma^2) maps every sigma to a timestep, and Heun averages the two
        x0/eps estimates.

        NOTE on units: an earlier version fed sigma directly to the network as if the model
        were an EDM denoiser in data units.  Our fields are z-scored, so the VP forward
        process has noise std sqrt(1-abar) <= 1, while the VP->VE variance ratio reaches
        sigma = 641 at t = 999; using sigma_max = 80 (a GenCast image setting) rescaled the
        initial noise by ~80x and produced garbage.  Everything below is in VP units.
        """
        device = next(self.model.parameters()).device
        B = shape[0]
        n = int(num_steps)

        # Timesteps: the SAME canonical spacing DDIM uses, so that the only difference
        # between the two curves is the order of the update (1st vs 2nd order).  A Karras
        # sigma grid and a uniform-lambda grid were both tried and collapse on a cosine
        # schedule (several targets map to the same integer t near t = 0 and t = 999).
        ts = list(reversed(_space_timesteps(self.timesteps, n)))
        if len(ts) < 3:
            return self.ddim_sample(shape, cond=cond, num_steps=n, clip_denoised=clip_denoised,
                                    x_init=x_init)

        ac = self.alphas_cumprod
        x = torch.randn(shape, device=device) if x_init is None else x_init.clone()

        def eps_of(xx, t):
            tb = torch.full((B,), int(t), device=device, dtype=torch.long)
            return self.model(xx, tb, cond)

        def x0_of(xx, t, e):
            a = ac[int(t)]
            v = (xx - torch.sqrt(1 - a) * e) / torch.sqrt(a)
            return v.clamp(-3, 3) if clip_denoised else v

        for i in range(len(ts) - 1):
            t, tn = ts[i], ts[i + 1]
            at, atn = ac[t], ac[tn]
            e_t = eps_of(x, t)
            x0_t = x0_of(x, t, e_t)
            if i == len(ts) - 2:
                x = torch.sqrt(atn) * x0_t + torch.sqrt(1 - atn) * e_t
                break
            x_e = torch.sqrt(atn) * x0_t + torch.sqrt(1 - atn) * e_t
            e_n = eps_of(x_e, tn)
            x0_n = x0_of(x_e, tn, e_n)
            x = (torch.sqrt(atn) * 0.5 * (x0_t + x0_n)
                 + torch.sqrt(1 - atn) * 0.5 * (e_t + e_n))
        return x

    def ddim_sample(self, shape, cond=None, num_steps=50, eta=0.0, clip_denoised=True,
                    x_init=None, clip_val=3.0):
        """
        DDIM sampling (faster than DDPM).
        
        Args:
            shape: (B, C, H, W)
            cond: conditioning
            num_steps: number of DDIM steps
            eta: stochasticity (0=deterministic, 1=DDPM)
            clip_denoised: clip denoised samples
        
        Returns:
            x_0: generated samples
        """
        device = next(self.model.parameters()).device
        B = shape[0]
        
        # DDIM timesteps -- openai/guided-diffusion space_timesteps('ddimN'):
        # the retained set is range(0, T, stride), so it ALWAYS contains t=0 and the
        # reverse loop runs [T-stride, ..., stride, 0].  The previous version used
        # arange(0, T, stride) + 1, which dropped t=0, shifted every step by one and
        # made the starting noise level depend on num_steps.
        timesteps = list(reversed(_space_timesteps(self.timesteps, num_steps)))
        
        # Start from noise (or a supplied x_T, needed for paired evaluation)
        x = torch.randn(shape, device=device) if x_init is None else x_init.clone()

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.model(x, t_batch, cond)
            
            # Predict x_0
            alpha_t = self.alphas_cumprod[t]
            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            if clip_denoised:
                x0_pred = x0_pred.clamp(-clip_val, clip_val)
            
            # Get next timestep
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_next = self.alphas_cumprod[t_next]
            else:
                alpha_next = torch.tensor(1.0, device=device)
            
            # DDIM update
            sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
            dir_xt = torch.sqrt(1 - alpha_next - sigma**2) * noise_pred
            noise = torch.randn_like(x) if eta > 0 else 0
            
            x = torch.sqrt(alpha_next) * x0_pred + dir_xt + sigma * noise
        
        return x


# ============================================================
# SDA (Score-based Data Assimilation)
# ============================================================

class SDAAssimilation:
    """
    Score-based Data Assimilation for multi-source GNSS-R observations.
    
    Observations:
      - Tianmu-1 GNSS-R (along-track wind speed)
      - FY-3 E/F/G GNSS-R (along-track wind speed)
      - CYGNSS GNSS-R (along-track wind speed)
      - China coastal buoys (in-situ wind speed/direction)
    
    SDA guides the reverse diffusion process towards observations
    by computing gradient of observation likelihood w.r.t. x_t.
    """
    
    def __init__(self, config, diffusion_model, normalizer):
        self.config = config
        self.diffusion = diffusion_model
        self.normalizer = normalizer
        self.sda_cfg = config['sda']
        
        self.lambda_init = self.sda_cfg['lambda']
        self.grad_clip = self.sda_cfg['grad_clip']
        self.observation_noise = self.sda_cfg['observation_noise']
        
        # Observation sources
        self.obs_sources = self.sda_cfg['observations']
        
        # Grid coordinates (will be set)
        self.lat_grid = None
        self.lon_grid = None
    
    def set_grid(self, lat, lon):
        """Set grid coordinates for observation interpolation."""
        self.lat_grid = lat  # (H,)
        self.lon_grid = lon  # (W,)
    
    def observation_operator(self, x_wind, obs_lats, obs_lons):
        """
        Bilinear interpolation from 2D wind field to observation points.

        IMPORTANT: the sampling grid may have DESCENDING latitude (ERA5 convention:
        row 0 = northernmost). We therefore map lat/lon -> fractional row/col index
        using the ACTUAL grid arrays (whatever their order) and then convert to the
        [-1,1] convention required by grid_sample (y=-1 at row 0, y=+1 at last row).

        Args:
            x_wind: (B, 2, H, W) — wind field (u, v) in normalized units
            obs_lats: (N,) — observation latitudes
            obs_lons: (N,) — observation longitudes

        Returns:
            obs_wind: (B, N, 2) — interpolated wind at observation points
        """
        B, C, H, W = x_wind.shape
        device = x_wind.device

        # to numpy for the index lookup (N is small)
        la = obs_lats.detach().cpu().numpy().astype(np.float64)
        lo = obs_lons.detach().cpu().numpy().astype(np.float64)
        lat_g = np.asarray(self.lat_grid, dtype=np.float64)
        lon_g = np.asarray(self.lon_grid, dtype=np.float64)

        # fractional row index (handle descending latitude)
        if lat_g[0] > lat_g[-1]:
            iy = np.interp(la, lat_g[::-1], np.arange(H, dtype=np.float64)[::-1])
        else:
            iy = np.interp(la, lat_g, np.arange(H, dtype=np.float64))
        # fractional col index (handle descending longitude, rare)
        if lon_g[0] > lon_g[-1]:
            ix = np.interp(lo, lon_g[::-1], np.arange(W, dtype=np.float64)[::-1])
        else:
            ix = np.interp(lo, lon_g, np.arange(W, dtype=np.float64))

        if device.type == 'mps':
            # grid_sample (and its backward) are unsupported on MPS; use a
            # differentiable gather-based bilinear. iy already accounts for any
            # descending latitude, so fractional row/col indices are used directly.
            fi = torch.tensor(iy, dtype=x_wind.dtype, device=device)
            fj = torch.tensor(ix, dtype=x_wind.dtype, device=device)
            y0 = torch.floor(fi).clamp(0, H - 2).long()
            x0 = torch.floor(fj).clamp(0, W - 2).long()
            wy = (fi - y0.to(fi.dtype)).view(1, 1, -1)
            wx = (fj - x0.to(fj.dtype)).view(1, 1, -1)

            def _g(yy, xx):
                return x_wind[:, :, yy, xx]  # (B, 2, N)

            obs_wind = (_g(y0, x0) * (1 - wy) * (1 - wx)
                        + _g(y0 + 1, x0) * wy * (1 - wx)
                        + _g(y0, x0 + 1) * (1 - wy) * wx
                        + _g(y0 + 1, x0 + 1) * wy * wx)
            return obs_wind.permute(0, 2, 1)  # (B, N, 2)

        # normalize to [-1,1] (grid_sample convention)
        y = 2.0 * iy / (H - 1) - 1.0
        x = 2.0 * ix / (W - 1) - 1.0
        y = torch.tensor(y, dtype=x_wind.dtype, device=device)
        x = torch.tensor(x, dtype=x_wind.dtype, device=device)

        # Create grid for grid_sample: (B, N, 1, 2)
        grid = torch.stack([x, y], dim=-1)[None, :, None, :].expand(B, -1, -1, -1).to(device)

        # Bilinear interpolation
        obs_wind = F.grid_sample(x_wind, grid, mode='bilinear',
                                  padding_mode='border', align_corners=True)
        # obs_wind: (B, 2, N, 1) -> (B, N, 2)
        obs_wind = obs_wind.squeeze(-1).permute(0, 2, 1)

        return obs_wind
    
    def compute_observation_loss(self, x_wind, observations):
        """
        Compute observation mismatch loss.
        
        Args:
            x_wind: (B, 2, H, W) — wind field (normalized)
            observations: dict with keys:
                'lats': (N,) — observation latitudes
                'lons': (N,) — observation longitudes
                'wind_speed': (N,) — observed wind speed (m/s, physical units)
                'wind_dir': (N,) optional — observed wind direction (degrees)
                'weights': (N,) optional — observation weights
                'mask': (N,) — valid observation mask
        
        Returns:
            loss: scalar — observation mismatch
        """
        obs_lats = observations['lats']
        obs_lons = observations['lons']
        obs_speed = observations['wind_speed']
        obs_mask = observations.get('mask', torch.ones_like(obs_speed, dtype=torch.bool))
        obs_weights = observations.get('weights', torch.ones_like(obs_speed))
        
        if obs_mask.sum() == 0:
            return torch.tensor(0.0, device=x_wind.device)
        
        # Interpolate model wind to observation points
        model_wind = self.observation_operator(x_wind, obs_lats, obs_lons)  # (B, N, 2)
        
        # Convert normalized wind to physical units and compute speed
        mean = torch.tensor(self.normalizer['mean'], device=x_wind.device)
        std = torch.tensor(self.normalizer['std'], device=x_wind.device)
        
        model_u = model_wind[:, :, 0] * std[0] + mean[0]  # (B, N)
        model_v = model_wind[:, :, 1] * std[1] + mean[1]
        model_speed = torch.sqrt(model_u**2 + model_v**2 + 1e-8)  # (B, N)
        
        # Compute mismatch (broadcast obs to batch)
        obs_speed_batch = obs_speed[None, :].expand_as(model_speed)  # (B, N)
        obs_weights_batch = obs_weights[None, :].expand_as(model_speed)
        obs_mask_batch = obs_mask[None, :].expand_as(model_speed)
        
        # Weighted MSE
        diff = (model_speed - obs_speed_batch) ** 2
        diff = diff * obs_weights_batch * obs_mask_batch.float()
        
        loss = diff.sum() / (obs_mask_batch.float().sum() + 1e-8)
        
        return loss
    
    def sda_guidance(self, x_t, t, cond, observations, lambda_scale=None):
        """
        Apply SDA guidance to reverse diffusion step.
        
        Args:
            x_t: (B, C, H, W) — current noisy sample
            t: (B,) — diffusion timestep
            cond: conditioning tensor
            observations: dict of observations
            lambda_scale: SDA strength (overrides config)
        
        Returns:
            x_t_guided: (B, C, H, W) — guided sample
        """
        if observations is None or len(observations) == 0:
            return x_t
        
        lambda_val = lambda_scale if lambda_scale is not None else self.lambda_init
        
        # Enable gradient computation
        x_t_grad = x_t.detach().requires_grad_(True)
        
        # Compute observation loss
        obs_loss = self.compute_observation_loss(x_t_grad, observations)
        
        # Compute gradient w.r.t. x_t
        grad = torch.autograd.grad(obs_loss, x_t_grad, create_graph=False)[0]
        
        # Gradient clipping
        if self.grad_clip > 0:
            grad_norm = grad.norm()
            if grad_norm > self.grad_clip:
                grad = grad * (self.grad_clip / grad_norm)
        
        # Apply guidance: x_t = x_t - lambda * grad * (1 - alpha_t) / sqrt(alpha_t)
        # This is the Tweedie's formula-based guidance
        alpha_t = self.diffusion.alphas_cumprod[t][:, None, None, None]
        guidance_scale = lambda_val * (1 - alpha_t) / (torch.sqrt(alpha_t) + 1e-8)
        
        x_t_guided = x_t - guidance_scale * grad
        
        return x_t_guided.detach()

    def sda_guidance_residual(self, x_t, t, cond, anchor, observations,
                              normalizer=None, lambda_scale=None):
        """SDA guidance for a RESIDUAL (delta) diffusion model.

        The model generates delta = target - anchor. The observation operator must act
        on the reconstructed ABSOLUTE field (anchor + x0_hat), not on the noisy delta.
        Here x0_hat is the predicted clean residual from the current noisy x_t, so the
        observation gradient propagates back into x_t and drives the reverse diffusion.

        Args:
            x_t: (B,2,H,W) noisy residual state (normalized).
            t: (B,) timesteps.
            cond: (B,8,H,W) conditioning.
            anchor: (B,2,H,W) latest observed wind field (normalized).
            observations: dict (lats/lons/wind_speed/mask/weights).
            normalizer: {'mean':[mu_u,mu_v],'std':[sd_u,sd_v]} for physical conversion.
            lambda_scale: override lambda.
        Returns:
            (B,2,H,W) guided noisy state.
        """
        if observations is None or len(observations) == 0:
            return x_t
        lam = lambda_scale if lambda_scale is not None else self.lambda_init
        x = x_t.detach().requires_grad_(True)

        # predicted clean residual from noisy x_t (x0 estimate, DDPM)
        eps = self.diffusion.model(x, t, cond)
        alpha_cp = self.diffusion.alphas_cumprod[t].view(-1, 1, 1, 1)
        # NOTE: forward uses q_sample(x0,t,eps) with noise param; x here is x_noisy
        x0_hat = (x - torch.sqrt(1 - alpha_cp) * eps) / (torch.sqrt(alpha_cp) + 1e-8)
        abs_norm = anchor + x0_hat                      # normalized absolute field

        # observation loss on physical absolute field (reuse existing compute with normalizer)
        if normalizer is not None:
            self._nrm = normalizer
        obs_loss = self.compute_observation_loss(abs_norm, observations)
        grad = torch.autograd.grad(obs_loss, x, create_graph=False)[0]
        if self.grad_clip > 0:
            gn = grad.norm()
            if gn > self.grad_clip:
                grad = grad * (self.grad_clip / gn)
        alpha_t = self.diffusion.alphas_cumprod[t].view(-1, 1, 1, 1)
        guidance_scale = lam * (1 - alpha_t) / (torch.sqrt(alpha_t) + 1e-8)
        return (x_t - guidance_scale * grad).detach()
    
    # ------------------------------------------------------------------
    #  MMPS: moment-matching posterior sampling (Rozet et al.)
    #  posterior score = prior score + likelihood score, with
    #     p(y | x_t) ~ N( y | H(x0_hat),  Sigma_y + J Var[x0|x_t] J^T )
    #  Under the diagonal (per-observation) moment-matching approximation this
    #  is exactly a precision-weighted gradient step with NO hand-tuned lambda.
    # ------------------------------------------------------------------
    def _obs_weight_sq(self, observations, H, W):
        """s_i = sum_j J_ij^2 for the bilinear observation operator.

        For bilinear sampling the 4 weights are (1-wx)(1-wy), wx(1-wy),
        (1-wx)wy, wx*wy, so s_i = [(1-wx)^2+wx^2]*[(1-wy)^2+wy^2] in [0.25, 1].
        """
        la = observations['lats'].detach().cpu().numpy().astype(np.float64)
        lo = observations['lons'].detach().cpu().numpy().astype(np.float64)
        lat_g = np.asarray(self.lat_grid, dtype=np.float64)
        lon_g = np.asarray(self.lon_grid, dtype=np.float64)
        if lat_g[0] > lat_g[-1]:
            iy = np.interp(la, lat_g[::-1], np.arange(H, dtype=np.float64)[::-1])
        else:
            iy = np.interp(la, lat_g, np.arange(H, dtype=np.float64))
        if lon_g[0] > lon_g[-1]:
            ix = np.interp(lo, lon_g[::-1], np.arange(W, dtype=np.float64)[::-1])
        else:
            ix = np.interp(lo, lon_g, np.arange(W, dtype=np.float64))
        ix = np.clip(ix, 0, W - 1); iy = np.clip(iy, 0, H - 1)
        wx = ix - np.floor(ix); wy = iy - np.floor(iy)
        return (((1 - wx) ** 2 + wx ** 2) * ((1 - wy) ** 2 + wy ** 2)).astype(np.float32)

    def _mmps_loss(self, x_wind, observations, w):
        """0.5 * sum_i w_i (H(x)_i - y_i)^2  -- the UNNORMALISED MMPS objective.

        Its gradient is  -J^T (w * r), i.e. exactly the MMPS guidance direction.
        """
        mw = self.observation_operator(x_wind, observations['lats'], observations['lons'])
        mean = torch.tensor(self.normalizer['mean'], device=x_wind.device)
        std = torch.tensor(self.normalizer['std'], device=x_wind.device)
        mu = mw[:, :, 0] * std[0] + mean[0]
        mv = mw[:, :, 1] * std[1] + mean[1]
        spd = torch.sqrt(mu ** 2 + mv ** 2 + 1e-8)
        r = spd - observations['wind_speed'][None, :]
        msk = observations.get('mask', torch.ones_like(observations['wind_speed'], dtype=torch.bool))
        return 0.5 * (w * r ** 2 * msk.float()[None, :]).sum()

    def sda_guidance_mmps(self, x_t, t, cond, anchor, observations,
                          normalizer=None, sigma_y=None):
        """MMPS posterior guidance for the residual diffusion model.

        Args:
            sigma_y: observation-error std (m/s). Replaces the hand-tuned lambda.
        Returns:
            (B,2,H,W) guided noisy state.
        """
        if observations is None or observations['wind_speed'].numel() == 0:
            return x_t
        if normalizer is not None:
            self.normalizer = normalizer                      # (fixes a dead assignment)
        sig = float(sigma_y if sigma_y is not None else
                    self.sda_cfg.get('sigma_y', 2.0))

        x = x_t.detach().requires_grad_(True)
        eps = self.diffusion.model(x, t, cond)
        acp = self.diffusion.alphas_cumprod[t].view(-1, 1, 1, 1)
        x0_hat = (x - torch.sqrt(1 - acp) * eps) / (torch.sqrt(acp) + 1e-8)
        abs_norm = anchor + x0_hat

        B, _, H, W = abs_norm.shape
        s = torch.as_tensor(self._obs_weight_sq(observations, H, W),
                            dtype=abs_norm.dtype, device=abs_norm.device)      # (N,)
        c_t = ((1 - acp) / (acp + 1e-8)).reshape(-1)                            # (B,)
        # observation-error std: constant, or the empirical wind-speed-dependent curve
        # per-source observation error (sources differ by 2-4x in accuracy)
        src_map = self.sda_cfg.get('sigma_y_by_source')
        src_col = observations.get('source')
        if src_map and src_col is not None:
            base = torch.tensor([float(src_map.get(str(x), sig)) for x in src_col],
                                dtype=abs_norm.dtype, device=abs_norm.device)
            sig = base.mean().item()
        use_ws = self.sda_cfg.get('sigma_y_windspeed', False)
        curve = self.sda_cfg.get('sigma_y_curve')
        if use_ws and curve is not None:
            yv = observations['wind_speed'].detach().cpu().numpy().astype(np.float64)
            cv_ = np.asarray(curve['v'], dtype=np.float64)
            cs_ = np.asarray(curve['s'], dtype=np.float64)
            sg = np.interp(yv, cv_, cs_)
            # linear continuation beyond the last tabulated wind speed
            hi = yv > cv_[-1]
            if hi.any():
                slope = (cs_[-1] - cs_[-2]) / (cv_[-1] - cv_[-2])
                sg[hi] = cs_[-1] + slope * (yv[hi] - cv_[-1])
            sig_t = torch.as_tensor(sg, dtype=abs_norm.dtype, device=abs_norm.device)[None, :]
        elif src_map and src_col is not None:
            sig_t = base[None, :].to(abs_norm.dtype)
        else:
            sig_t = torch.full((1, s.numel()), sig, dtype=abs_norm.dtype, device=abs_norm.device)
        # per-observation MMPS precision (detached, as in moment matching)
        w = 1.0 / (sig_t ** 2 + c_t[:, None] * s[None, :])                      # (B,N)

        loss = self._mmps_loss(abs_norm, observations, w)
        grad = torch.autograd.grad(loss, x, create_graph=False)[0]
        # step size is Var[x0|x_t] = c_t, with NO hand-tuned lambda and no global clip
        return (x_t - c_t.view(-1, 1, 1, 1) * grad).detach()

    @torch.no_grad()
    def sample_with_mmps_residual(self, shape, cond, anchor, observations=None,
                                  use_ddim=True, ddim_steps=50, normalizer=None,
                                  guidance_below=300, sigma_y=None, x_init=None):
        """DDIM sampling with MMPS posterior guidance (see sda_guidance_mmps)."""
        device = next(self.diffusion.model.parameters()).device
        B = shape[0]
        x = torch.randn(shape, device=device) if x_init is None else x_init.clone()
        step_ratio = self.diffusion.timesteps // ddim_steps
        timesteps = list(reversed(list(np.arange(0, self.diffusion.timesteps, step_ratio) + 1)))
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            if observations is not None and t <= guidance_below:
                with torch.enable_grad():
                    x = self.sda_guidance_mmps(x, t_batch, cond, anchor, observations,
                                               normalizer=normalizer, sigma_y=sigma_y)
            with torch.no_grad():
                noise_pred = self.diffusion.model(x, t_batch, cond)
                alpha_t = self.diffusion.alphas_cumprod[t]
                x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                x0_pred = x0_pred.clamp(-3, 3)
                if i < len(timesteps) - 1:
                    t_next = timesteps[i + 1]
                    alpha_next = self.diffusion.alphas_cumprod[t_next]
                else:
                    alpha_next = torch.tensor(1.0, device=device)
                x = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * noise_pred
        return x

    @torch.no_grad()
    def sample_with_sda(self, shape, cond, observations=None, 
                        guidance_steps=None, lambda_schedule=None,
                        use_ddim=False, ddim_steps=50):
        """See above (v2 interface). Use sda_guidance_residual for Scheme-B residual models."""
        device = next(self.diffusion.model.parameters()).device
        B = shape[0]
        
        if guidance_steps is None:
            guidance_steps = self.sda_cfg['guidance_steps']
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        if use_ddim:
            # DDIM with SDA
            step_ratio = self.diffusion.timesteps // ddim_steps
            timesteps = np.arange(0, self.diffusion.timesteps, step_ratio) + 1
            timesteps = list(reversed(timesteps))
            
            for i, t in enumerate(timesteps):
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)
                
                # Apply SDA guidance at specified steps
                if t in guidance_steps or -1 in guidance_steps:
                    lambda_val = lambda_schedule if lambda_schedule is not None else self.lambda_init
                    x = self.sda_guidance(x, t_batch, cond, observations, lambda_val)
                
                # DDIM step
                with torch.no_grad():
                    noise_pred = self.diffusion.model(x, t_batch, cond)
                    alpha_t = self.diffusion.alphas_cumprod[t]
                    x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                    x0_pred = x0_pred.clamp(-3, 3)
                    
                    if i < len(timesteps) - 1:
                        t_next = timesteps[i + 1]
                        alpha_next = self.diffusion.alphas_cumprod[t_next]
                    else:
                        alpha_next = torch.tensor(1.0, device=device)
                    
                    dir_xt = torch.sqrt(1 - alpha_next) * noise_pred
                    x = torch.sqrt(alpha_next) * x0_pred + dir_xt
        else:
            # DDPM with SDA
            for i in reversed(range(self.diffusion.timesteps)):
                t_batch = torch.full((B,), i, device=device, dtype=torch.long)
                
                # Apply SDA guidance at specified steps
                if i in guidance_steps or -1 in guidance_steps:
                    lambda_val = lambda_schedule if lambda_schedule is not None else self.lambda_init
                    x = self.sda_guidance(x, t_batch, cond, observations, lambda_val)
                
                # Reverse diffusion step
                x = self.diffusion.p_sample(x, t_batch, cond)
        
        return x

    @torch.no_grad()
    def sample_with_sda_residual(self, shape, cond, anchor, observations=None,
                                 guidance_steps=None, lambda_schedule=None,
                                 use_ddim=False, ddim_steps=50, normalizer=None,
                                 guidance_below=None, x_init=None):
        """Sample with SDA for a RESIDUAL (anchor + delta) model.

        Args:
            guidance_below: if set, SDA guidance is applied at ALL steps with t <=
                guidance_below (a timestep threshold). This is the robust way to guide
                over a *range* of late diffusion steps; the older `guidance_steps`
                exact-membership test rarely matched DDIM's sparse timesteps.
            x_init: optional initial noise; pass the SAME tensor used by the baseline
                to make "with vs without assimilation" a fair paired comparison.
        """
        device = next(self.diffusion.model.parameters()).device
        B = shape[0]
        if guidance_steps is None:
            guidance_steps = self.sda_cfg['guidance_steps']
        x = torch.randn(shape, device=device) if x_init is None else x_init.clone()

        def _guide(t_scalar):
            """whether to guide at timestep t_scalar"""
            if guidance_below is not None:
                return t_scalar <= guidance_below
            return (t_scalar in guidance_steps) or (-1 in guidance_steps)

        if use_ddim:
            step_ratio = self.diffusion.timesteps // ddim_steps
            timesteps = list(reversed(list(np.arange(0, self.diffusion.timesteps, step_ratio) + 1)))
            for i, t in enumerate(timesteps):
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)
                if _guide(t):
                    lam = lambda_schedule if lambda_schedule is not None else self.lambda_init
                    with torch.enable_grad():
                        x = self.sda_guidance_residual(x, t_batch, cond, anchor, observations,
                                                       normalizer=normalizer, lambda_scale=lam)
                with torch.no_grad():
                    noise_pred = self.diffusion.model(x, t_batch, cond)
                    alpha_t = self.diffusion.alphas_cumprod[t]
                    x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                    x0_pred = x0_pred.clamp(-3, 3)
                    if i < len(timesteps) - 1:
                        t_next = timesteps[i + 1]; alpha_next = self.diffusion.alphas_cumprod[t_next]
                    else:
                        alpha_next = torch.tensor(1.0, device=device)
                    dir_xt = torch.sqrt(1 - alpha_next) * noise_pred
                    x = torch.sqrt(alpha_next) * x0_pred + dir_xt
        else:
            for i in reversed(range(self.diffusion.timesteps)):
                t_batch = torch.full((B,), i, device=device, dtype=torch.long)
                if _guide(i):
                    lam = lambda_schedule if lambda_schedule is not None else self.lambda_init
                    with torch.enable_grad():
                        x = self.sda_guidance_residual(x, t_batch, cond, anchor, observations,
                                                       normalizer=normalizer, lambda_scale=lam)
                x = self.diffusion.p_sample(x, t_batch, cond)
        return x


# ============================================================
# EMA (Exponential Moving Average)
# ============================================================

class EMA:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


if __name__ == '__main__':
    # Test diffusion
    import yaml
    from diffusion_model import create_model
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = create_model(config)
    diffusion = GaussianDiffusion(model, **config['diffusion'])
    
    # Test training loss
    B = 2
    x = torch.randn(B, 2, 256, 256)
    cond = torch.randn(B, 8, 256, 256)
    
    loss = diffusion.training_loss(x, cond)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test sampling (small number of steps for speed)
    diffusion.timesteps = 10  # Override for testing
    with torch.no_grad():
        sample = diffusion.sample((1, 2, 256, 256), cond[:1])
    print(f"Sample shape: {sample.shape}")
