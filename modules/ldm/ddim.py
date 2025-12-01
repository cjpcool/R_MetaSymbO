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
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import wandb


@torch.no_grad()
def ddim_denoising_loop(z, z_latent, batch, num_nodes, diffusion_model, scheduler, ddim_steps, eta, total_steps):
    num_samples  = len(num_nodes)
    step_interval = max(total_steps // ddim_steps, 1)
    time_sequence = list(range(0, total_steps, step_interval))
    if time_sequence[-1] != (total_steps - 1):
        time_sequence.append(total_steps - 1)
    time_sequence = time_sequence[::-1]
    

    for i in range(len(time_sequence)):
        t = time_sequence[i]
        t_tensor = z.new_full((num_samples,), t, dtype=torch.long)

        noise_pred = diffusion_model(z, None, batch, num_nodes, t_tensor, latent_condition=z_latent)

        alpha_cumprod_t = scheduler.alphas_cumprod[t]
        sqrt_alpha_cumprod_t = alpha_cumprod_t.sqrt()
        sqrt_one_minus_alpha_cumprod_t = (1. - alpha_cumprod_t).sqrt()

        z0 = (z - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t

        if i == len(time_sequence) - 1:
            z_prev = z0
        else:
            t_next = time_sequence[i + 1]
            alpha_cumprod_t_next = scheduler.alphas_cumprod[t_next]

            sigma_t = eta * torch.sqrt((1 - alpha_cumprod_t_next) / (1 - alpha_cumprod_t))
            x_coef = alpha_cumprod_t_next.sqrt()
            e_coef = torch.sqrt(1.0 - alpha_cumprod_t_next - sigma_t**2)

            z_prev = x_coef * z0 + e_coef * noise_pred
            if eta > 0:
                z_prev += sigma_t * torch.randn_like(z)
        z = z_prev
    return z



class DenoiseTransformer(nn.Module):
    def __init__(self, latent_dim=128, n_heads=8, n_layers=3, time_emb_dim=128, is_condition=False, condition_dim=12):
        super().__init__()
        d_model= latent_dim * 2 + condition_dim if is_condition else latent_dim * 2
        self.d_model = d_model
        self.is_condition = is_condition
        self.condition_dim= condition_dim

        # A simple MLP for time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, latent_dim),
        )

        self.in_proj = nn.Linear(d_model, latent_dim)  # 
        self.transformer = TransformerEncoder(
            d_model=latent_dim, 
            n_head=n_heads, 
            ffn_hidden=latent_dim,
            n_layers=n_layers,
            drop_prob=0.1
        )
        # self.transformer = nn.Sequential(
        #     nn.Linear(latent_dim, latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(latent_dim, latent_dim),
        # )
        self.out_proj = nn.Linear(latent_dim, latent_dim)  # 3 -> [Δx, Δy, Δz]

    def sinusoidal_time_embedding(self, t, dim=128):
        """
        t: (batch_size,) integer timesteps
        Returns: (batch_size, dim)
        """
        device = t.device
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim
        )
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding

    def forward(self, x_t, t, node_mask=None, latent_condition=None):
        # x_t: shape [B, N, d_model], t: shape [B]
        B, N, _ = x_t.shape

        # 1) Time embedding
        t_emb = self.sinusoidal_time_embedding(t)  # [B, latent_dim]
        t_emb = self.time_mlp(t_emb)              # [B, latent_dim]
        # Expand for each node
        
        t_emb_expanded = t_emb.unsqueeze(1).expand(B, N, x_t.shape[-1])  # [B, N, d_model]

        # 2) Combine x_t with time embedding
        x_in = x_t + t_emb_expanded

        
        latent_condition = latent_condition.unsqueeze(1).expand(B, N, -1)
        print(latent_condition.shape)
        print(latent_condition)
        x_in = torch.cat([x_in, latent_condition], dim=-1)
        x_in = x_in * node_mask if node_mask is not None else x_in
        # 3) Transformer encoding
        #    If node_mask is (B, N), we typically convert it to (B, 1, N) or (B, N) 
        #    depending on your TransformerEncoder design
        x_in = self.in_proj(x_in)  # [B, N, latent]
        x_in = x_in * node_mask if node_mask is not None else x_in
        encoded = self.transformer(x_in)  # [B, N, d_model]
        encoded = encoded * node_mask if node_mask is not None else encoded
        # 4) Predict noise (Δx, Δy, Δz) for each node
        noise_pred = self.out_proj(encoded)  # [B, N, latent]

        # 5) Optionally mask out padded nodes
        if node_mask is not None:
            noise_pred = noise_pred * node_mask

        return noise_pred



class DenoiseGPS(nn.Module):
    def __init__(self, latent_dim=128, time_emb_dim=128, is_condition=False, condition_dim=12, conv_type=None):
        super().__init__()
        d_model= latent_dim * 2 + condition_dim if is_condition else latent_dim * 2
        self.d_model = d_model
        self.is_condition = is_condition
        self.condition_dim= condition_dim

        self.in_proj = nn.Linear(d_model, latent_dim)  # 

        self.GPSModel = GPSModel(channels=latent_dim, num_layers=4, attn_type='multihead', conv_type=None)
        # A simple MLP for time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, latent_dim),
        )

        self.in_proj = nn.Linear(d_model, latent_dim)  # 
        
        self.out_proj = nn.Linear(latent_dim, latent_dim)  # 3 -> [Δx, Δy, Δz]

    def sinusoidal_time_embedding(self, t, dim=128):
        """
        t: (batch_size,) integer timesteps
        Returns: (batch_size, dim)
        """
        device = t.device
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim
        )
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding

    def forward(self, x_t,edge_index, batch, num_nodes, t, latent_condition=None):
        # x_t: shape [B, N, d_model], t: shape [B]
        N = x_t.shape
        B = batch.max() + 1

        # 1) Time embedding
        t_emb = self.sinusoidal_time_embedding(t) 
        t_emb = self.time_mlp(t_emb)
        
        t_emb_expanded = torch.repeat_interleave(t_emb, repeats=num_nodes, dim=0)
        # t_emb_expanded = t_emb_expanded.expand(x_t.shape[0], x_t.shape[-1])

        # 2) Combine x_t with time embedding
        x_in = x_t + t_emb_expanded

        
        latent_condition = torch.repeat_interleave(latent_condition, repeats=num_nodes, dim=0)
        x_in = torch.cat([x_in, latent_condition], dim=-1)

        x_in = self.in_proj(x_in)  # [B, N, latent]
        encoded = self.GPSModel(x_in, edge_index, batch)

        noise_pred = self.out_proj(encoded) 

        return noise_pred


class HeadSemantic(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.proj_in = nn.Linear(latent_dim, latent_dim)

        self.proj_out = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, latent_dim)
        )
    def forward(self, x, batch):
        x = self.proj_in(x)
        x = scatter_add(x, batch, dim=0, dim_size=batch.max() + 1)
        x = self.proj_out(x)
        return x



class LatentDiffusionDDIM_GeomVAE:
    def __init__(
        self,
        geomvae: GeomVAE,
        diffusion_model: nn.Module,
        scheduler: DDPM_Scheduler,
        lr_vae=1e-4,
        lr_diffusion=1e-4,
        device='cuda',
        is_condition=False,
        condition_dim=12
    ):
        self.geomvae = geomvae.to(device)
        self.diffusion_model = diffusion_model.to(device)
        self.scheduler = scheduler.to(device)
        self.condition_dim = condition_dim
        self.is_condition = is_condition

        self.device = device

        self.optim_vae = torch.optim.Adam(self.geomvae.parameters(), lr=lr_vae)
        self.scheduler_vae = torch.optim.lr_scheduler.StepLR(self.optim_vae, step_size=50, gamma=0.9)
        self.optim_diffusion = torch.optim.Adam(self.diffusion_model.parameters(), lr=lr_diffusion)
        self.scheduler_diffusion = torch.optim.lr_scheduler.StepLR(self.optim_diffusion, step_size=100, gamma=0.9)

    def _forward_diffusion(self, z0, t):
        noise = torch.randn_like(z0)
        sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t].unsqueeze(-1) #[nodes,1]
        sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        z_t = sqrt_alpha * z0 + sqrt_one_minus_alpha * noise
        return z_t, noise

    def _ddpm_loss(self, z0, batch, num_nodes, t, condition=None):
        repeated_t = torch.repeat_interleave(t, repeats=num_nodes, dim=0)
        
        # diffusion in z_g
        zt, real_noise= self._forward_diffusion(z0, repeated_t)
        noise_pred = self.diffusion_model(zt, None, batch, num_nodes, t, latent_condition=condition)
        loss =  torch.sqrt(scatter_mean((noise_pred - real_noise)**2, index=batch, dim=0, dim_size=batch.max() + 1)).mean()

        # TODO: Disentangle the z_g, and diffusion in each sub-space
        # z0_cat = torch.cat([z0, repeated_latent], dim=-1)
        # z0_e = self.geomvae.decoder.proj_in_edge(z0_cat)
        # z0_p = self.geomvae.decoder.proj_in_pos(z0_cat)
        # z0_s = self.geomvae.decoder.proj_in_semantic(z0_cat)

        # zt_e, real_noise_e = self._forward_diffusion(z0_e, t)
        # zt_p, real_noise_p = self._forward_diffusion(z0_p, t)
        # zt_s, real_noise_s = self._forward_diffusion(z0_s, t)

        # noise_pred_e = self.diffusion_model(zt_e, None, batch, num_nodes, t, latent_condition=condition)
        # noise_pred_p = self.diffusion_model(zt_p, None, batch, num_nodes, t, latent_condition=condition)
        # noise_pred_s = self.diffusion_model(zt_s, None, batch, num_nodes, t, latent_condition=condition)
        # loss = torch.sqrt(scatter_mean((noise_pred_e - real_noise_e)**2, index=batch, dim=0, dim_size=batch.max() + 1)).mean()
        # loss += torch.sqrt(scatter_mean((noise_pred_p - real_noise_p)**2, index=batch, dim=0, dim_size=batch.max() + 1)).mean()
        # loss += torch.sqrt(scatter_mean((noise_pred_s - real_noise_s)**2, index=batch, dim=0, dim_size=batch.max() + 1)).mean()
        return loss

    def train(self, dataloader, checkpoint_dir='./checkpoints/ldm', num_epochs=10, beta_kl=1.0, train_vae=True, use_wandb=True):
        os.makedirs(checkpoint_dir, exist_ok=True)

        best_ae_loss = float('inf')
        best_denoise_loss = float('inf')
        for epoch in range(num_epochs):
            self.geomvae.train()
            self.diffusion_model.train()
            diff_loss_total = 0.0

            if train_vae:
                total_loss = total_kld = total_recon = 0.0
                total_lattice_len = total_lattice_ang = total_coords = 0.0
                total_edge = total_node_num = total_y = 0.0

            epoch_start_time = time.time()
            num_batches = 0

            for batch_idx, batch_data in enumerate(dataloader):
                batch_data = batch_data.to(self.device)
                if self.is_condition:
                    condition = batch_data.y
                else:
                    condition=None

                # Train VAE
                if train_vae:
                    self.optim_vae.zero_grad()

                    loss_dict = self.geomvae.compute_loss(batch_data, device=self.device, beta=beta_kl)
                    loss = loss_dict['total_loss']
                    loss.backward()
                    self.optim_vae.step()

                    total_loss += loss.item()
                    total_recon += loss_dict['recon_loss'].item()
                    total_kld += loss_dict['kl_divergence'].item()
                    total_lattice_len += loss_dict['loss_lattice_len'].item()
                    total_lattice_ang += loss_dict['loss_lattice_ang'].item()
                    total_coords += loss_dict['loss_coords'].item()
                    total_edge += loss_dict['loss_edge'].item()
                    total_node_num += loss_dict['loss_node_num'].item()
                    total_y += loss_dict['loss_y'].item()

                else:
                    loss = torch.tensor(0.0, device=self.device)

                # Train diffusion model
                self.optim_diffusion.zero_grad()

                with torch.no_grad():
                    lengths_normed, angles_normd = self.geomvae.normalizer(batch_data.lengths, batch_data.angles)
                    latent_lattice, zg = self.geomvae.encoder(
                        batch_data.node_type, batch_data.cart_coords, batch_data.edge_index,
                        batch_data.batch, lengths_normed, angles_normd, batch_data.num_atoms, condition=condition
                    )
                    latent_lattice = self.geomvae.reparameterize(latent_lattice)[0]

                batch_size = latent_lattice.size(0)
                t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device).long()

                if self.is_condition:
                    latent_condition=torch.cat([latent_lattice, condition], dim=-1) if self.is_condition else latent_lattice
                else:
                    latent_condition=latent_lattice
                # Note: num_atoms != num_nodes
                diff_loss = self._ddpm_loss(zg, batch_data.batch, batch_data.num_atoms, t, latent_condition)
                diff_loss.backward()
                self.optim_diffusion.step()
                
                diff_loss_total += diff_loss.item()
                num_batches += 1

            epoch_end_time = time.time()

            if self.scheduler_vae.get_last_lr()[0] > 1e-5:
                        self.scheduler_vae.step()
            if self.scheduler_diffusion.get_last_lr()[0] > 1e-5:
                    self.scheduler_diffusion.step()
    

            avg_diff_loss = diff_loss_total / num_batches

            if avg_diff_loss < best_denoise_loss:
                best_denoise_loss = avg_diff_loss
                print("Saving best diffusion model...", os.path.join(checkpoint_dir, 'best_diff_model.pt'))
                torch.save(self.diffusion_model.state_dict(), os.path.join(checkpoint_dir, 'best_diff_model.pt'))
                torch.save(self.geomvae.state_dict(), os.path.join(checkpoint_dir, 'best_ae_model.pt'))


            log_dict = {'Diffusion Loss': avg_diff_loss}
            print_str = f"Epoch[{epoch+1}/{num_epochs}] | Diffusion: {avg_diff_loss:.4f}"

            if train_vae:
                avg_loss = total_loss / num_batches
                avg_recon = total_recon / num_batches
                avg_kld = total_kld / num_batches
                avg_len = total_lattice_len / num_batches
                avg_ang = total_lattice_ang / num_batches
                avg_coord = total_coords / num_batches
                avg_edge = total_edge / num_batches
                avg_node_num = total_node_num / num_batches
                avg_y = total_y / num_batches
                
                log_dict.update({
                    'Total Loss': avg_loss,
                    'Recon Loss': avg_recon,
                    'KL Divergence': avg_kld,
                    'Lattice Length Loss': avg_len,
                    'Lattice Angle Loss': avg_ang,
                    'Coords Loss': avg_coord,
                    'Edge Loss': avg_edge,
                    'Node Num Loss': avg_node_num
                })
                if best_ae_loss > avg_loss:
                        best_ae_loss = avg_loss
                        torch.save(self.geomvae.state_dict(), os.path.join(checkpoint_dir, 'best_ae_model.pt'))

                print_str += (
                    f" || Loss={avg_loss:.4f}"
                    f" | Recon={avg_recon:.4f}"
                    f" | KL_lat={avg_kld:.4f}"
                    f" | L_len={avg_len:.4f}"
                    f"  L_ang={avg_ang:.4f}"
                    f"  L_coords={avg_coord:.4f}"
                    f"  L_edge={avg_edge:.4f}"
                    f"  L_nnum={avg_node_num:.4f}"
                    f"  L_y={avg_y:.4f}"
                )
            if use_wandb:
                wandb.log(log_dict)
            print(print_str + f"  || Time={epoch_end_time - epoch_start_time:.2f}s")
        if use_wandb:
            wandb.finish()

    @torch.no_grad()
    def sample_ddim(self, num_samples=8, ddim_steps=50, eta=0.0, z_lattice=None, z_g=None, is_recon=False, condition=None):
        self.geomvae.eval()
        self.diffusion_model.eval()

        assert (z_lattice == None and z_g == None) or (z_lattice != None and z_g != None), "Either z_lattice or z_g should be provided, not both."


        _, node_num_pred, lengths_pred, angles_pred, z_lattice = self.geomvae.sample_lattice(num_samples, z_lattice, device=self.device, random_scale_latent=1.)

        print('node_num_pred' ,node_num_pred)
        num_nodes = node_num_pred
        if is_recon:
            # forward diffusion
            t = torch.LongTensor([self.scheduler.num_timesteps]).to(self.device)-1
            repeated_t = torch.repeat_interleave(t, repeats=z_g.shape[0], dim=0)
            zt_g = self._forward_diffusion(z_g, repeated_t)[0]
            zt_g_all_samples = []
            for i, nnum in enumerate(num_nodes):
                if nnum <= zt_g.shape[0]:
                    zt_g_all_samples.append(zt_g[:nnum])
                else:
                    zt_g_all_samples.append(torch.cat([zt_g, torch.randn((nnum - zt_g.shape[0], zt_g.shape[1]), device=self.device)], dim=0))
            zt_g = torch.cat(zt_g_all_samples, dim=0)
            t = torch.repeat_interleave(t, repeats=num_samples, dim=0)
        else:
            zt_g =  torch.randn(
                    (num_nodes.sum(), self.geomvae.latent_dim),
                    device=self.device
                )
    
        batch = torch.arange(num_samples, device=self.device).repeat_interleave(num_nodes)


        # if z_g is not None:
        #     # Disentangle the z_g
        #     repeated_latent = torch.repeat_interleave(condition, repeats=node_num_pred, dim=0)
        #     z0_cat = torch.cat([z_g, repeated_latent], dim=-1)
        #     z0_e = self.geomvae.decoder.proj_in_edge(z0_cat)
        #     z0_p = self.geomvae.decoder.proj_in_pos(z0_cat)
        #     z0_s = self.geomvae.decoder.proj_in_semantic(z0_cat)
        # else:
        #     zT_e = torch.randn(
        #         (node_num_pred.sum(), self.geomvae.latent_dim),
        #         device=self.device
        #     )
        #     zT_p = torch.randn(
        #         (node_num_pred.sum(), self.geomvae.latent_dim),
        #         device=self.device
        #     )
        #     zT_s = torch.randn(
        #         (node_num_pred.sum(), self.geomvae.latent_dim),
        #         device=self.device
        #     )

        # decode global lattice information via vae
        # TODO: add condition
        if self.is_condition:
            if condition is None:
                condition = 0.1 * torch.randn((num_samples, self.condition_dim), device=self.device)
            else:
                condition = condition.expand(num_samples, -1).to(self.device)
            z_lattice = torch.cat([z_lattice, condition], dim=-1)
        
        # node_mask = get_node_mask(node_num_pred, max_num_nodes)

        # denoising
        z0_g = ddim_denoising_loop(
            z=zt_g,
            z_latent=z_lattice,
            batch=batch,
            num_nodes=num_nodes,
            total_steps = self.scheduler.num_timesteps,
            diffusion_model=self.diffusion_model,
            scheduler=self.scheduler,
            ddim_steps=ddim_steps,
            eta=eta
            )

        # Decode the latent coordinates to get the reconstructed graph
        repeated_latent = torch.repeat_interleave(z_lattice, repeats=num_nodes, dim=0)
        encoded_coords = torch.cat([z0_g, repeated_latent], dim=-1)
        _, edge_index_list, _ = self.geomvae.decoder.decode_edges(encoded_coords, num_nodes, batch)
        edge_index = convert_edge_list_to_edge_index(edge_index_list, num_nodes)
        coords_pred = self.geomvae.decoder.decode_coords(encoded_coords, edge_index, batch)
        y_pred =  self.geomvae.decoder.decode_semantic(encoded_coords, batch)


        # obtain batch coords with node mask
        coords_pred_list = []
        nnum_offset = 0
        for i, nnum in enumerate(num_nodes):
            coords_pred_list.append(coords_pred[nnum_offset:nnum_offset + nnum])
            nnum_offset += nnum

        return lengths_pred, angles_pred, num_nodes, coords_pred_list, edge_index_list, y_pred
    


