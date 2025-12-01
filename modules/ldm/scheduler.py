import torch
import torch.nn.functional as F
import torch.nn as nn
from diffusers import DDPMScheduler, DDIMScheduler

import numpy as np
import math

def get_beta_schedule(
        num_train_timesteps: int,
        start: float = 1e-4,
        end:   float = 2e-2,
        schedule_type: str = 'linear',
        *,
        s: float = 0.008,          # small offset used in the paper
        max_beta: float = 0.999    # clip for numerical stability
) -> torch.Tensor:
    """
    Return a tensor of length `num_train_timesteps` containing the β_t schedule.

    Parameters
    ----------
    num_train_timesteps : int
        Number of diffusion steps T.
    start, end : float
        Only used when schedule_type == 'linear'.
    schedule_type : {'linear', 'cosine'}
        Type of schedule.
    s : float
        Small offset for the cosine schedule (Nichol & Dhariwal 2021).
    max_beta : float
        Upper bound to keep β_t < 1 for stability.

    Returns
    -------
    torch.Tensor
        1‑D tensor of shape (T,) with dtype=torch.float32.
    """
    if schedule_type == 'linear':
        return torch.linspace(start, end, num_train_timesteps, dtype=torch.float32)

    elif schedule_type == 'cosine':
        # t runs from 0 … T
        steps = torch.arange(num_train_timesteps + 1, dtype=torch.float64)

        # \bar{α}_t = cos²(((t/T) + s)/(1 + s) · π/2)
        t_bar = (steps / num_train_timesteps + s) / (1 + s)
        alphas_bar = torch.cos(0.5 * math.pi * t_bar).pow(2)

        # Convert cumulative ᾱ to incremental β_t
        alphas_bar_prev = alphas_bar[:-1]      # ᾱ_{t-1}
        alphas_bar_next = alphas_bar[1:]       # ᾱ_{t}
        betas = 1.0 - (alphas_bar_next / alphas_bar_prev)

        # Numerical clipping
        betas = betas.clamp(max=max_beta).float()
        return betas

    else:
        raise NotImplementedError(f"Unknown schedule_type: {schedule_type!r}")


class DDPM_Scheduler:
    """
    Basic DDPM Scheduler that precomputes:
      - betas
      - alphas = 1 - betas
      - alphas_cumprod
      - sqrt_alphas_cumprod
      - sqrt_one_minus_alphas_cumprod
      - sqrt_recip_alphas
      - posterior_variance
    These are used in the forward and reverse diffusion processes.
    """
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        schedule_type='linear'
    ):
        self.num_train_timesteps = num_train_timesteps

        # 1) Beta schedule
        self.betas = get_beta_schedule(
            num_train_timesteps, start=beta_start, end=beta_end, schedule_type=schedule_type
        )  # shape: [num_train_timesteps]

        # 2) Alphas and their cumulative products
        self.alphas = 1.0 - self.betas                        # shape: [num_train_timesteps]
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)   # shape: [num_train_timesteps]
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )  # same shape, but shifted by 1 with a pad of 1.0 at t=0

        # 3) Commonly used quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # The term for p(x_{t-1} | x_t)
        # posterior_variance = beta_t * (1 - alpha_{t-1}^cumprod) / (1 - alpha_t^cumprod)
        # from the DDPM paper's eq. (4)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device):
        """
        Move all tensors to a specified device (e.g. 'cuda' or 'cpu').
        """
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self



def clip_noise_schedule(alphas2: np.ndarray, clip_value: float = 0.001):
    """Clip ᾱ_t to avoid exact 0 or 1 (prevents Inf/NaN)."""
    return np.clip(alphas2, clip_value, 1.0 - clip_value)


def polynomial_alpha_bar(timesteps: int, s: float = 1e-4, power: int = 3):
    """
    GeoLDM-style polynomial noise schedule.
        ᾱ_t = (1 - (t/T)^power)^2   (plus a tiny bias s)
    """
    steps = timesteps + 1                        # include t = 0
    x = np.linspace(0, steps, steps, dtype=np.float64)
    alphas2 = (1.0 - (x / steps) ** power) ** 2
    alphas2 = clip_noise_schedule(alphas2, 0.001)
    alphas2 = (1.0 - 2.0 * s) * alphas2 + s      # precision shift
    return alphas2                                # shape (T+1,)


class PolynomialDDPMScheduler(DDPMScheduler):
    """
    Drop-in replacement for DDPMScheduler using GeoLDM's polynomial ᾱ schedule.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        power: int = 3,
        s: float = 1e-4,
        **kwargs,          # pass clip_sample, prediction_type, etc.
    ):
        T = num_train_timesteps                                # training steps
        alpha_bar = torch.linspace(1.0, 0.0, T)     # linear decay
        alphas    = torch.where(torch.arange(T)==0,
                                alpha_bar,          # α₀
                                alpha_bar[1:] / alpha_bar[:-1])  # αₜ = ᾱₜ / ᾱₜ₋₁
        betas     = 1 - alphas

        # 3. call parent ctor with our betas
        super().__init__(trained_betas=betas, **kwargs)

        # (optional) store meta
        self.power = power
        self.s     = s



class ScoreScheduler:
    """
    Log-linear (ρ) VE scheduler like EDM / KarrasVe.
    """

    def __init__(self,
                 num_train_timesteps: int = 1000,
                 sigma_min : float = 0.002,
                 sigma_max : float = 80.0,
                 rho       : float = 7.0,
                 sigma_data: float = 0.5,
                 prediction_type: str = "score",
                 device    = "cpu"):
        self.prediction_type = prediction_type
        self.num_train_timesteps = num_train_timesteps
        self.sigma_data = sigma_data
        self.device = torch.device(device)

        # pre-compute continuous σ curve   σ(t) = (σ_max^(1-t)  σ_min^t)^(1/ρ)
        t = torch.linspace(0, 1, num_train_timesteps, device=self.device)
        sigmas = (sigma_max ** (1 - t) * sigma_min ** t) ** (1 / rho)

        with torch.no_grad():
            self.sigmas = sigmas.float()          # (T,)


    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        t : integer indices (0 … T-1)  or float indices
        returns σ_t  (same shape as t, broadcastable).
        """
        return torch.index_select(self.sigmas, 0, t.long())

    def sample_training_sigma(self, batch_size: int) -> torch.Tensor:
        """
        Draw sigmas from log-uniform distribution over [σ_min, σ_max].
        """
        u = torch.rand(batch_size, device=self.device)
        log_sigma = u * (self.sigmas[-1].log() - self.sigmas[0].log()) + self.sigmas[0].log()
        return log_sigma.exp()


    # ----- inference timetable (like diffusers .set_timesteps) ----------

    def get_inference_sigmas(self, num_inference_steps: int):
        """
        Returns monotonically decreasing σ list of length num_inference_steps+1
        using Karras et al. (2022) formula.
        """
        ramp = torch.linspace(0, 1, num_inference_steps+1, device=self.device)
        sigmas = (self.sigmas[-1] ** (1 - ramp) * self.sigmas[0] ** ramp) ** (1 / 1.0)
        return sigmas

    # simple single-step ODE (Euler) for demo
    def step(self, x, score, sigma, sigma_next):
        """
        x_{σ_next} = x_σ + (sigma_next - sigma) * score(x_σ, σ)
        Euler integration of VE ODE.
        """
        d = sigma_next - sigma
        return x + d * score




