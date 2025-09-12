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

@torch.no_grad()
def disentangeled_ddim_denoising_loop(noise_type, z, z_latent, edge_index, batch, num_nodes, diffusion_model, scheduler, ddim_steps, eta, total_steps):
    num_samples  = len(num_nodes)-0.5
    step_interval = max(total_steps // ddim_steps, 1)
    time_sequence = list(range(0, total_steps, step_interval))
    if time_sequence[-1] != (total_steps - 1):
        time_sequence.append(total_steps - 1)
    time_sequence = time_sequence[::-1]
    
    for i in range(len(time_sequence)):
        t = time_sequence[i]
        t_tensor = z.new_full((num_samples,), t, dtype=torch.long)

        noise_pred = diffusion_model(noise_type, z, edge_index, batch, num_nodes, t_tensor, latent_condition=z_latent)

        alpha_cumprod_t = scheduler.alphas_cumprod[t]
        sqrt_alpha_cumprod_t = alpha_cumprod_t.sqrt()
        sqrt_one_minus_alpha_cumprod_t = (1. - alpha_cumprod_t).sqrt()

        if scheduler.prediction_type=='v_prediction':
            noise_pred = (noise_pred + sqrt_one_minus_alpha_cumprod_t*z) / sqrt_alpha_cumprod_t


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
        self.scheduler_vae = torch.optim.lr_scheduler.StepLR(self.optim_vae, step_size=600, gamma=1.0)
        self.optim_diffusion = torch.optim.Adam(self.diffusion_model.parameters(), lr=lr_diffusion)
        self.scheduler_diffusion = torch.optim.lr_scheduler.StepLR(self.optim_diffusion, step_size=600, gamma=.6)

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
        elif self.scheduler.prediction_type == 'score':
            sigma = self.scheduler.get_sigma(t)
            z_t, target = self._foward_score(z0, sigma)

        return z_t, target

    def _foward_score(self, z0, sigma):
        noise = torch.randn_like(z0)
        x_sigma = z0 + sigma[:, None] * noise

        # score target   s = (x0 - xσ) / σ²
        target = (z0 - x_sigma) / sigma[:, None]**2
        return x_sigma, target

    def disentangle_latent(self, z0, num_nodes, condition=None, batch=None):
        """
        Disentangle the latent space into edge, position, and semantic components.
        """
        repeated_latent = torch.repeat_interleave(condition, repeats=num_nodes, dim=0)
        z0_e, z0_p, z0_s = self.geomvae.disentangle_latent(z0)
        z0_e = self.geomvae.reparameterize(z0_e, tag='edge', noise_scale=0.0)[0]
        z0_p = self.geomvae.reparameterize(z0_p, tag='coords', noise_scale=0.0)[0]
        z0_s = self.geomvae.reparameterize(z0_s, tag='semantic', noise_scale=0.0)[0]
        if self.is_diff_on_coords:
            z0_p = self.geomvae.decoder.decode_coords(z0_p, None, batch, repeated_latent)
        return z0_e, z0_p, z0_s, 
    
    def disentangle_forward_diffusion(self, z0_e, z0_p, z0_s, repeated_t):
        zt_e, real_noise_e = self._forward_diffusion(z0_e, repeated_t)
        zt_p, real_noise_p = self._forward_diffusion(z0_p, repeated_t)
        zt_s, real_noise_s = self._forward_diffusion(z0_s, repeated_t)

        return zt_e, zt_p, zt_s, real_noise_e.detach(), real_noise_p.detach(), real_noise_s.detach()
    
    def disentangle_noise_pred(self, zt_e, zt_p, zt_s, t, edge_index, batch, num_nodes, condition):
        noise_pred_e = self.diffusion_model('edge', zt_e, edge_index, batch, num_nodes, t, latent_condition=condition)
        # TODO: use predicted edge_index instead of original edge_index
        noise_pred_p = self.diffusion_model('coordinate', zt_p, edge_index, batch, num_nodes, t, latent_condition=condition)
        noise_pred_s = self.diffusion_model('semantic', zt_s, edge_index, batch, num_nodes, t, latent_condition=condition)
        return noise_pred_e, noise_pred_p, noise_pred_s


    def get_batch_geom_latent_norm_from_v(self, t, v_hat, z_t):
        alpha_bar_t  = self.scheduler.alphas_cumprod[t]          # tensor [b*num_nodes,1]
        sqrt_ab      = alpha_bar_t.sqrt()                  # √ᾱ_t
        sqrt_one_mab = (1 - alpha_bar_t).sqrt()            # √(1-ᾱ_t)
        if self.scheduler.prediction_type=='v_prediction':
            z0_hat = (sqrt_ab[:, None] * z_t - sqrt_one_mab[:, None] * v_hat)
        elif self.scheduler.prediction_type=='epsilon':
            z0_hat = (z_t - sqrt_one_mab[:, None] * v_hat) / sqrt_ab[:, None]

        return z0_hat
    
    def _score_loss(self, z0, edge_index, batch, num_nodes, t, condition=None, lambda_e=10.0, lambda_p=100.0, lambda_s=1.0):
        # repeated_t = torch.repeat_interleave(t, repeats=num_nodes, dim=0)
        sigma = self.scheduler.get_sigma(t)
        z0_e, z0_p, z0_s = self.disentangle_latent(z0, num_nodes, condition, batch=batch)
        zt_e, real_noise_e = self._foward_score(z0_e, sigma)
        zt_p, real_noise_p = self._foward_score(z0_p, sigma)
        zt_s, real_noise_s = self._foward_score(z0_s, sigma)

        noise_pred_e, noise_pred_p, noise_pred_s = self.disentangle_noise_pred(zt_e, zt_p, zt_s, sigma, edge_index, batch, num_nodes, condition)
        # EDM weighting
        weight = 0.5 * ((sigma**2 + self.scheduler.sigma_data**2) / (sigma**2))[:, None]
        loss_e = (scatter_mean(torch.mean(weight * (noise_pred_e - real_noise_e)**2, dim=-1), index=batch, dim=0, dim_size=batch.max() + 1)).mean()
        loss_p = (scatter_mean(torch.mean(weight * (noise_pred_p - real_noise_p)**2, dim=-1), index=batch, dim=0, dim_size=batch.max() + 1)).mean()
        loss_s = (scatter_mean(torch.mean(weight * (noise_pred_s - real_noise_s)**2, dim=-1), index=batch, dim=0, dim_size=batch.max() + 1)).mean()

        loss = lambda_e * loss_e + lambda_p * loss_p + lambda_s * loss_s 


        loss_dict = {
            'loss_e': loss_e,
            'loss_p': loss_p,
            'loss_s': loss_s,
            'total_loss': loss,
            'z_norm_reg': torch.tensor(0.0, device=self.device),
            'coords_error': torch.tensor(0.0, device=self.device)
        }
        return loss_dict


    def _ddpm_loss(self, z0, edge_index, batch, num_nodes, t, condition=None, lambda_e=10.0, lambda_p=100.0, lambda_s=1.0):
        repeated_t = torch.repeat_interleave(t, repeats=num_nodes, dim=0)
        
        # diffusion in z_g
        # zt, real_noise= self._forward_diffusion(z0, repeated_t)
        # noise_pred = self.diffusion_model(zt, None, batch, num_nodes, t, latent_condition=condition)
        # loss =  torch.sqrt(scatter_mean((noise_pred - real_noise)**2, index=batch, dim=0, dim_size=batch.max() + 1)).mean()

        # z0_e, dec0_p, z0_s = self.disentangle_latent(z0, num_nodes, condition, batch=batch)
        z0_e, z0_p, z0_s = self.disentangle_latent(z0, num_nodes, condition, batch=batch)
        zt_e, zt_p, zt_s, real_noise_e, real_noise_p, real_noise_s = self.disentangle_forward_diffusion(z0_e, z0_p, z0_s, repeated_t)
        # zt_e, dect_p, zt_s, real_noise_e, real_noise_dec_p, real_noise_s = self.disentangle_forward_diffusion(z0_e, dec0_p, z0_s, repeated_t)
        # if v_prediction, noise_pred is v
        # noise_pred_e, noise_pred_p, noise_pred_s = self.disentangle_noise_pred(zt_e, dect_p, zt_s, t, edge_index, batch, num_nodes, condition)
        noise_pred_e, noise_pred_p, noise_pred_s = self.disentangle_noise_pred(zt_e, zt_p, zt_s, t, edge_index, batch, num_nodes, condition)
        

        # snr = self.scheduler.alphas_cumprod[repeated_t] / (1-self.scheduler.alphas_cumprod[repeated_t])
        # gamma = 5.0
        # weight = ((snr + 1) / (snr + gamma))[:,None]
        weight = 1.
        loss_e = (scatter_mean(torch.mean(weight * (noise_pred_e - real_noise_e)**2, dim=-1), index=batch, dim=0, dim_size=batch.max() + 1)).mean()
        loss_p = (scatter_mean(torch.mean(weight * (noise_pred_p - real_noise_p)**2, dim=-1), index=batch, dim=0, dim_size=batch.max() + 1)).mean()
        # loss_p = (scatter_mean(torch.mean((noise_pred_p - real_noise_dec_p)**2, dim=-1), index=batch, dim=0, dim_size=batch.max() + 1)).mean()
        loss_s = (scatter_mean(torch.mean(weight * (noise_pred_s - real_noise_s)**2, dim=-1), index=batch, dim=0, dim_size=batch.max() + 1)).mean()

        z0_p_hat = self.get_batch_geom_latent_norm_from_v(repeated_t, noise_pred_p, zt_p)
        # z0_p_hat = self.get_batch_geom_latent_norm_from_v(repeated_t, noise_pred_p, dect_p)
        z0_e_hat = self.get_batch_geom_latent_norm_from_v(repeated_t, noise_pred_e, zt_e)
        z0_s_hat = self.get_batch_geom_latent_norm_from_v(repeated_t, noise_pred_s, zt_s)


        z_norm_reg = (
            (z0_p_hat - z0_p.detach()).pow(2).mean() + \
                    #   (z0_p_hat - dec0_p.detach()).pow(2).mean() + \
                     (z0_e_hat - z0_e.detach()).pow(2).mean() + \
                     (z0_s_hat - z0_s.detach()).pow(2).mean()) / 3
        
        loss = lambda_e * loss_e + lambda_p * loss_p + lambda_s * loss_s 

        coords_error = torch.sqrt((z0_p.detach() - z0_p_hat.detach()).pow(2).mean()).mean()

        loss_dict = {
            'loss_e': loss_e,
            'loss_p': loss_p,
            'loss_s': loss_s,
            'total_loss': loss,
            'z_norm_reg': z_norm_reg,
            'coords_error': coords_error
        }
        return loss_dict

    def train(self, dataloader, checkpoint_dir='./checkpoints/ldm', num_epochs=10, 
              lambda_e=1.0, lambda_p=100.0, lambda_s=1.0, beta_kl=1.0, 
              train_vae=True, use_wandb=True):
        os.makedirs(checkpoint_dir, exist_ok=True)

        best_ae_loss = float('inf')
        best_denoise_loss = float('inf')
        for epoch in range(num_epochs):
            self.geomvae.train()
            self.diffusion_model.train()
            diff_loss_total = total_loss_e = total_loss_p = total_loss_s = total_loss_znorm = 0.0
            total_coords_error = 0.
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
                t = torch.randint(0, self.scheduler.num_train_timesteps, (batch_size,), device=self.device).long()

                if self.is_condition:
                    latent_condition=torch.cat([latent_lattice, condition], dim=-1) if self.is_condition else latent_lattice
                else:
                    latent_condition=latent_lattice
                # Note: num_atoms != num_nodes
                if self.scheduler.prediction_type == 'score':
                    diff_loss_dict = self._score_loss(zg, batch_data.edge_index, batch_data.batch, batch_data.num_atoms, t, latent_condition,
                                            lambda_e=lambda_e, lambda_p=lambda_p, lambda_s=lambda_s)
                else:
                    diff_loss_dict = self._ddpm_loss(zg, batch_data.edge_index, batch_data.batch, batch_data.num_atoms, t, latent_condition,
                                            lambda_e=lambda_e, lambda_p=lambda_p, lambda_s=lambda_s)
                diff_loss = diff_loss_dict['total_loss']
                diff_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), 1.0)
                self.optim_diffusion.step()
                
                diff_loss_total += diff_loss.item()
                total_loss_e += diff_loss_dict['loss_e'].item()
                total_loss_p += diff_loss_dict['loss_p'].item()
                total_loss_s += diff_loss_dict['loss_s'].item()
                total_loss_znorm += diff_loss_dict['z_norm_reg'].item()
                total_coords_error += diff_loss_dict['coords_error'].item()
                num_batches += 1

            epoch_end_time = time.time()

            if self.scheduler_vae.get_last_lr()[0] > 1e-5:
                        self.scheduler_vae.step()
            if self.scheduler_diffusion.get_last_lr()[0] > 1e-4:
                    self.scheduler_diffusion.step()
    

            avg_diff_loss = diff_loss_total / num_batches
            avg_loss_e = total_loss_e / num_batches
            avg_loss_p = total_loss_p / num_batches
            avg_loss_s = total_loss_s / num_batches
            avg_loss_znorm = total_loss_znorm/ num_batches
            avg_coords_error = total_coords_error / num_batches
            if avg_diff_loss < best_denoise_loss:
                best_denoise_loss = avg_diff_loss
                print("Saving best diffusion model...", os.path.join(checkpoint_dir, 'best_diff_model.pt'))
                torch.save(self.diffusion_model.state_dict(), os.path.join(checkpoint_dir, 'best_diff_model.pt'))
                torch.save(self.geomvae.state_dict(), os.path.join(checkpoint_dir, 'best_ae_model.pt'))


            log_dict = {'Diffusion Loss': avg_diff_loss,
                        'Diffusion Loss Edge': avg_loss_e,
                        'Diffusion Loss Coordinate': avg_loss_p,
                        'Diffusion Loss Semantic': avg_loss_s,
                        'Diffusion Z Norm': avg_loss_znorm,
                        'lr_diffusion': self.scheduler_diffusion.get_last_lr()[0],
                        'coords_error': avg_coords_error
                        }
            print_str = (f"Epoch[{epoch+1}/{num_epochs}] | L_diff={avg_diff_loss:.4f}"
                        f" L_edge={avg_loss_e:.4f}"
                        f" L_coords={avg_loss_p:.4f}"
                        f" L_semantic={avg_loss_s:.4f}"
                        f" L_zNorm={avg_loss_znorm:.4f}"
                        f" Error_coords: {avg_coords_error:.4f}"
            )


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
                    'Node Num Loss': avg_node_num,
                    'lr_vae': self.scheduler_vae.get_last_lr()[0]
                })
                if best_ae_loss > avg_loss:
                        best_ae_loss = avg_loss
                        torch.save(self.geomvae.state_dict(), os.path.join(checkpoint_dir, 'best_ae_model.pt'))

                print_str += (
                    f" || L_VAE ={avg_loss:.4f}"
                    f" | Recon={avg_recon:.4f}"
                    f" KL_lat={avg_kld:.4f}"
                    f" | L_len={avg_len:.4f}"
                    f" L_ang={avg_ang:.4f}"
                    f" L_coords={avg_coord:.4f}"
                    f" L_edge={avg_edge:.4f}"
                    f" L_nnum={avg_node_num:.4f}"
                    f" L_y={avg_y:.4f}"
                )
            if use_wandb:
                wandb.log(log_dict)
            print(print_str + f"  || Time={epoch_end_time - epoch_start_time:.2f}s")
        if use_wandb:
            wandb.finish()

    @torch.no_grad()
    def sample_ddim(self, num_samples=8, ddim_steps=50, eta=0.0, z_lattice=None, z_g=None, batch=None, is_recon=False, condition=None, coords_noise_scale=10., edge_noise_scale=1.0, semantic_noise_scale=.01):
        self.geomvae.eval()
        self.diffusion_model.eval()

        assert (z_lattice == None and z_g == None) or (z_lattice != None and z_g != None), "Either z_lattice or z_g should be provided, not both."

        _, node_num_pred, lengths_pred, angles_pred, z_lattice = self.geomvae.sample_lattice(num_samples, z_lattice, device=self.device, random_scale_latent=1.)

        
        if is_recon:
            # forward diffusion
            print('node_num_pred' ,node_num_pred)
            num_nodes_true = scatter_add(torch.ones_like(batch), batch, dim=0, dim_size=batch.max() + 1)
            # TODO: use node_num_pred instead of num_nodes_true
            t = torch.LongTensor([self.scheduler.num_train_timesteps]).to(self.device)-1
            repeated_t = torch.repeat_interleave(t, repeats=z_g.shape[0], dim=0)
            # TODO: use node_num_pred instead of num_nodes_true

            z0_e, z0_p, z0_s = self.disentangle_latent(z_g, num_nodes_true, z_lattice)
            zt_all = self.disentangle_forward_diffusion(z0_e, z0_p, z0_s, repeated_t)[:3]
            
            num_nodes = node_num_pred

            revised_zt_all = []
            for z_temp in zt_all:
                zt_g_all_samples = []
                for i, nnum in enumerate(num_nodes):
                    if nnum <= z_g.shape[0]:
                        zt_g_all_samples.append(z_temp[:nnum])
                    else:
                        zt_g_all_samples.append(torch.cat([z_temp, coords_noise_scale * torch.randn((nnum - z_temp.shape[0], z_temp.shape[1]), device=self.device)], dim=0))
                zt_i = torch.cat(zt_g_all_samples, dim=0)
                revised_zt_all.append(zt_i)
            revised_zt_all = {
                'edge': revised_zt_all[0],
                'coordinate': revised_zt_all[1],
                'semantic': revised_zt_all[2]
            }
            t = torch.repeat_interleave(t, repeats=num_samples, dim=0)
        else:
            num_nodes = node_num_pred
            # zt_g =  torch.randn(
            #         (num_nodes.sum(), self.geomvae.latent_dim),
            #         device=self.device
            #     )
            revised_zt_all = {
                'edge':  edge_noise_scale * torch.randn((num_nodes.sum(), self.geomvae.latent_dim),device=self.device),
                'coordinate': coords_noise_scale * torch.randn((num_nodes.sum(), self.geomvae.latent_dim),device=self.device),
                'semantic': semantic_noise_scale * torch.randn((num_nodes.sum(), self.geomvae.latent_dim),device=self.device)
            }
    
        batch = torch.arange(num_samples, device=self.device).repeat_interleave(num_nodes)


        # decode global lattice information via vae
        # TODO: add condition
        if self.is_condition:
            if condition is None:
                condition = 0.1 * torch.randn((num_samples, self.condition_dim), device=self.device)
            else:
                condition = condition.expand(num_samples, -1).to(self.device)
            z_lattice = torch.cat([z_lattice, condition], dim=-1)
        
        # node_mask = get_node_mask(node_num_pred, max_num_nodes)

        total_steps = self.scheduler.num_train_timesteps
        # total_steps = 100

        step_interval = max(total_steps // ddim_steps, 1)
        time_sequence = list(range(0, total_steps, step_interval))
        if time_sequence[-1] != (total_steps - 1):
            time_sequence.append(total_steps - 1)
        time_sequence = time_sequence[::-1]


        # z0_p = disentangeled_ddim_denoising_loop('coordinate', revised_zt_all['coordinate'], z_lattice, 
        #                                       None, batch, num_nodes_true,
        #                                        total_steps = total_steps,
        #                                         diffusion_model=LDM.diffusion_model,
        #                                         scheduler=LDM.scheduler,
        #                                         ddim_steps=ddim_steps,
        #                                         eta=eta)
        z0_p = revised_zt_all['coordinate']
        for t_i in time_sequence:
            t_tensor = z0_p.new_full((num_samples,), t_i, dtype=torch.long)
            noise_pred = self.diffusion_model('coordinate', z0_p, None, batch, num_nodes, t_tensor, latent_condition=z_lattice)
            z0_p = self.scheduler.step(noise_pred, timestep=t_i, sample=z0_p).prev_sample
        coords_pred = self.geomvae.decoder.decode_coords(z0_p, None, None)


        # denoising
        # z0_e = disentangeled_ddim_denoising_loop('edge', revised_zt_all['edge'], z_lattice, 
        #                                     None, batch, num_nodes_true,
        #                                     total_steps = total_steps,
        #                                         diffusion_model=LDM.diffusion_model,
        #                                         scheduler=LDM.scheduler,
        #                                         ddim_steps=ddim_steps,
        #                                         eta=eta)
        z0_e = revised_zt_all['edge']
        for t_i in time_sequence:
            t_tensor = z0_p.new_full((num_samples,), t_i, dtype=torch.long)
            noise_pred = self.diffusion_model('edge', z0_e, None, batch, num_nodes, t_tensor, latent_condition=z_lattice)
            z0_e = self.scheduler.step(noise_pred, timestep=t_i, sample=z0_e).prev_sample

        # Decode the latent coordinates to get the reconstructed graph
        _, edge_index_list, _ = self.geomvae.decoder.decode_edges(z0_e, num_nodes, batch, coords_pred)
        edge_index = convert_edge_list_to_edge_index(edge_index_list, num_nodes)



        # z0_s = disentangeled_ddim_denoising_loop('semantic', revised_zt_all['semantic'], z_lattice, 
        #                                       edge_index, batch, num_nodes_true,
        #                                        total_steps = total_steps,
        #                                         diffusion_model=LDM.diffusion_model,
        #                                         scheduler=LDM.scheduler,
        #                                         ddim_steps=ddim_steps,
        #                                         eta=eta)
        z0_s = revised_zt_all['semantic']
        for t_i in time_sequence:
            t_tensor = z0_p.new_full((num_samples,), t_i, dtype=torch.long)
            noise_pred = self.diffusion_model('semantic', z0_s, edge_index, batch, num_nodes, t_tensor, latent_condition=z_lattice)
            z0_s = self.scheduler.step(noise_pred, timestep=t_i, sample=z0_s).prev_sample
        y_pred =  self.geomvae.decoder.decode_semantic(z0_s, batch)

            # obtain batch coords with node mask
        coords_pred_list = []
        nnum_offset = 0
        for i, nnum in enumerate(num_nodes):
            coords_pred_list.append(coords_pred[nnum_offset:nnum_offset + nnum])
            nnum_offset += nnum

        return lengths_pred, angles_pred, num_nodes, coords_pred_list, edge_index_list, y_pred
