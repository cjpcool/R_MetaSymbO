import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from modules.ldm.scheduler import DDPM_Scheduler
from torch_scatter import scatter_mean, scatter_add

from utils.ldm_utils import get_node_mask, unpadding_max_num_node2graph, convert_edge_list_to_edge_index
from modules.geom_autoencoder import GeomVAE
from modules.transformer_backbone import Encoder as TransformerEncoder
from modules.gps_backbone import GPSModel
from modules.submodules import DisentangledDenoise
import math
import time
import os
import wandb


class LatentDiffusionDDIM_GeomVAE:
    def __init__(
        self,
        geomvae: GeomVAE,
        diffusion_model: DisentangledDenoise,
        scheduler,
        lr_vae=1e-4,
        lr_diffusion=1e-4,
        device='cuda',
        is_condition=False,
        condition_dim=12,
        is_diff_on_coords = False
    ):
        self.is_diff_on_coords = is_diff_on_coords
        self.geomvae = geomvae.to(device)
        self.diffusion_model = diffusion_model.to(device)
        self.scheduler = scheduler
        self.condition_dim = condition_dim
        self.is_condition = is_condition

        self.device = device

        self.optim_vae = torch.optim.Adam(self.geomvae.parameters(), lr=lr_vae)
        self.scheduler_vae = torch.optim.lr_scheduler.StepLR(self.optim_vae, step_size=100, gamma=0.9)
        self.optim_diffusion = torch.optim.Adam(self.diffusion_model.parameters(), lr=lr_diffusion)
        self.scheduler_diffusion = torch.optim.lr_scheduler.StepLR(self.optim_diffusion, step_size=200, gamma=1.)

    def _forward_diffusion(self, z0, t):
        noise = torch.randn_like(z0)
        z_t = self.scheduler.add_noise(z0, noise, t)
        # sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t].unsqueeze(-1) #[nodes,1]
        # sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        # z_t = sqrt_alpha * z0 + sqrt_one_minus_alpha * noise
        if self.scheduler.prediction_type == 'v_prediction':
            target = self.scheduler.get_velocity(z0, noise, t)
        elif self.scheduler.prediction_type == 'epsilon':
            target = noise

        return z_t, target


    def disentangle_latent(self, z0, num_nodes, condition=None, batch=None):
        """
        Disentangle the latent space into edge, position, and semantic components.
        """
        repeated_latent = torch.repeat_interleave(condition, repeats=num_nodes, dim=0)
        z0_cat = torch.cat([z0, repeated_latent], dim=-1)
        z0_e = self.geomvae.decoder.proj_in_edge(z0_cat)
        z0_p = self.geomvae.decoder.proj_in_pos(z0_cat)
        z0_s = self.geomvae.decoder.proj_in_semantic(z0_cat)
        if self.is_diff_on_coords:
            z0_p = self.geomvae.decoder.decode_coords(z0_p, None, batch)
        return z0_e, z0_p, z0_s, 
    