import time
import os

import torch
import torch.nn.functional as F
from click.core import batch
from torch import nn

from utils.ldm_utils import padding_graph2max_num_node, compute_edge_bce_loss_with_pos_weight, farthest_point_sampling
from .spherenet import SphereNet, SphereNetNode, radius_graph
from .transformer_backbone import Encoder as TransformerEncoder
import numpy as np
from torch_scatter import scatter_add

from modules.gps_backbone import GPSModel

from modules.submodules import EdgeDecoderDotProduct, SharedDisentangleLayer
from torch_geometric.nn import GraphNorm
import wandb

class EncoderwithPredictionHead(nn.Module):
    def __init__(self, AEmodel, latent_dim=128, condition_dim=12, y_mean=None, y_std=None):
        super().__init__()
        self.condition_dim = condition_dim
        self.AEmodel = AEmodel
        self.Predictor = nn.Sequential(
            nn.Linear(latent_dim*2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, condition_dim),
        )
        
        if y_mean is not None and y_std is not None:
            self.y_mean = nn.Parameter(torch.tensor(y_mean), requires_grad=False)
            self.y_std = nn.Parameter(torch.tensor(y_std), requires_grad=False)
        else:
            self.y_mean = nn.Parameter(torch.zeros((1, condition_dim), dtype=torch.float), requires_grad=False)
            self.y_std = nn.Parameter(torch.ones((1,condition_dim), dtype=torch.float), requires_grad=False)

    def forward(self, z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms, denormalize=True):
        # 1) encoding
        latent_global, encoded_coords = self.AEmodel.encoder(
            z, coords, edge_index, batch,
            lengths_normed, angles_normed, num_atoms, condition=None
        )  
        # print('Encoded_coords.std', encoded_coords.std(dim=0).mean())
        _, z0_s = self.AEmodel.disentangle_latent(encoded_coords, batch)
        if self.AEmodel.is_variational:
            z0_s, _, _ = self.AEmodel.reparameterize(z0_s, tag='semantic', noise_scale=0.)
        cat_semantic_latent = torch.cat([z0_s, latent_global], dim=-1)
        out = self.Predictor(cat_semantic_latent)
        if denormalize:
            # denormalize y_pred
            out = out * self.y_std + self.y_mean
            # if not self.training:
            #     # inplace
            #     out[:, :6] = torch.abs(out[:, :6])
            # else:
            #     y_pred_6 = out[:, :6]
            #     y_pred_post = out[:, 6:]
            #     y_pred_6 = torch.abs(y_pred_6)
            #     out = torch.cat([y_pred_6, y_pred_post], dim=1)
        return out

class GeomEncoder(nn.Module):
    """
    Encode graph to a latent, with node embeddings.
    """
    def __init__(self, latent_dim=128, is_condition=False, condition_dim=12):
        super().__init__()
        self.is_condition = is_condition
        self.condition_dim= condition_dim
        self.GraphNorm = GraphNorm(latent_dim) 
        # self.encoderCoords = TransformerEncoder(d_model=latent_dim, n_head=8, ffn_hidden=latent_dim, n_layers=3,
        #                                 drop_prob=0.1)
        self.encoderCoords = GPSModel(channels=latent_dim, num_layers=4, attn_type='multihead')

        self.proj = nn.Linear(latent_dim, latent_dim)
        self.encoderGeoStruct = SphereNetNode(energy_and_force=False, cutoff=5, num_layers=4,
                  hidden_channels=128, out_channels=latent_dim//2, int_emb_size=64,
                  basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
                  num_spherical=3, num_radial=6, envelope_exponent=5,
                  num_before_skip=1, num_after_skip=2, num_output_layers=3)

        self.encoderCoordsLatent = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim//2)  
        )

        self.encoderGlobal = SphereNet(energy_and_force=False, cutoff=5, num_layers=4,
                  hidden_channels=128, out_channels=latent_dim//2, int_emb_size=64,
                  basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
                  num_spherical=3, num_radial=6, envelope_exponent=5,
                  num_before_skip=1, num_after_skip=2, num_output_layers=3)
        self.encoderLattice = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim//2)  # 对应 lengths+angles
        )
        if is_condition:
            self.combineSemantic = nn.Linear(latent_dim+condition_dim, latent_dim)

        else:
            self.combineSemantic = nn.Linear(latent_dim, latent_dim)
        
    def encode_latent(self, z, pos, edge_index, batch, lengths, angles):
        graph_latent = self.encoderGlobal(z, pos, edge_index, batch)
        lattice_params = torch.cat([lengths, angles], dim=-1)
        lattice_latent = self.encoderLattice(lattice_params)
        semantic_latent = torch.cat([graph_latent, lattice_latent], dim=-1)
        geo_node_latent = self.encoderGeoStruct(z, pos, edge_index, batch)
        coords_latent = self.encoderCoordsLatent(pos)
        geo_coords_latent_concat = torch.cat([coords_latent, geo_node_latent], dim=-1)
        return semantic_latent, geo_coords_latent_concat


    def forward(self, z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms, condition=None):

        max_batch_num_node = num_atoms.max()
        semantic_latent, geo_coords_latent_concat = self.encode_latent(z, coords, edge_index, batch, lengths_normed, angles_normed)
        if self.is_condition:
            if condition is None:
                condition  = torch.randn((semantic_latent.shape[0], self.condition_dim), device=z.device)
            assert condition.shape[0] == semantic_latent.shape[0] and condition.shape[1] == self.condition_dim, f"condition shape {condition.shape} is not valid"
            semantic_latent = torch.cat([semantic_latent, condition], dim=-1)
        semantic_latent = self.combineSemantic(semantic_latent)

        # geo_coords_latent_concat, mask = padding_graph2max_num_node(geo_coords_latent_concat, batch, max_batch_num_node)
        # geo_coords_latent = self.encoderCoords(geo_coords_latent_concat) * mask
        # geo_coords_latent = self.proj(geo_coords_latent) * mask
        geo_coords_latent_concat = self.GraphNorm(geo_coords_latent_concat, batch)
        geo_coords_latent = self.encoderCoords(geo_coords_latent_concat, edge_index, batch)
        geo_coords_latent = self.GraphNorm(geo_coords_latent, batch)
        geo_coords_latent = self.proj(geo_coords_latent)
        return semantic_latent, geo_coords_latent




class GeomDecoder(nn.Module):
    def __init__(self, latent_dim=128, max_node_num=100, edge_sample_threshold=0.5, is_condition=False, condition_dim=12, composition_dim=119):
        super().__init__()
        self.is_condition = is_condition
        self.condition_dim = condition_dim
        self.edge_sample_threshold = edge_sample_threshold
        self.max_node_num = max_node_num
        self.composition_dim = composition_dim

        self.decoderLattice = nn.Sequential(
            nn.Linear(latent_dim//2, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        self.edgeDecoder = EdgeDecoderDotProduct(d_model=latent_dim, use_coords=True, coords_dim=3)
        self.split_size=latent_dim//2

        self.decoderNodenum = nn.Sequential(
            nn.Linear(latent_dim // 2,32),
            nn.ReLU(),
            nn.Linear(32, max_node_num),
        )
        # self.decoderCoords = TransformerEncoder(d_model=latent_dim, n_head=8, ffn_hidden=latent_dim, n_layers=3,
        #                             drop_prob=0.1)
        self.decoderCoords = GPSModel(channels=latent_dim, num_layers=1, attn_type='multihead', conv_type=None)

        self.shared_decoder = SharedDisentangleLayer(latent_dim, layer_num=4, is_condition=is_condition, condition_dim=condition_dim, conv_type=None)
        if not is_condition:
            self.splitGlobal = nn.Linear(latent_dim, latent_dim)
        else:
            self.splitGlobal = nn.Linear(latent_dim+condition_dim, latent_dim)

        self.proj_out_pos = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )

        self.proj_out_semantic_pred = nn.Sequential(
                nn.Linear(latent_dim, latent_dim//2),
                nn.ReLU(),
                nn.Linear(latent_dim//2, condition_dim))

        # Composition head (per-node element logits from geo coords latent)
        self.compositionHead = nn.Sequential(
                nn.Linear(latent_dim, latent_dim//2),
                nn.ReLU(),
                nn.Linear(latent_dim//2, composition_dim)
            )

    def decode_latent_global(self, latent):
        latent = self.splitGlobal(latent)  # shape=[B, latent_dim]
        graph_latent, lattice_latent = latent[:, :self.split_size], latent[:, self.split_size:]

        lattice_out = self.decoderLattice(lattice_latent)  # shape=[B, 6]
        lengths_pred = lattice_out[:, :3]
        angles_pred = lattice_out[:, 3:]

        node_num_logit = self.decoderNodenum(graph_latent)
        node_num_prob = F.softmax(node_num_logit, dim=-1)
        node_num_pred = node_num_prob.argmax(dim=-1) + 1
        return node_num_logit, node_num_pred, lengths_pred, angles_pred

    def decode_edges(self, encoded_coords, num_nodes, batch, coords, return_node_emb=False, repeated_cond=None):
        encoded_coords = self.shared_decoder(encoded_coords, batch, repeated_cond)
        max_batch_num_node = num_nodes.max()
        encoded_coords_padded, node_mask = padding_graph2max_num_node(encoded_coords, batch, max_batch_num_node)
        coords_padded, _ = padding_graph2max_num_node(coords, batch, max_batch_num_node)
        edge_logits, edge_index_list = self.edgeDecoder(encoded_coords_padded, node_mask=node_mask, coords=coords_padded, sample_threshold=self.edge_sample_threshold, return_logits=True)
        if return_node_emb:
            return edge_logits, edge_index_list, encoded_coords
        return edge_logits, edge_index_list, node_mask

    def decode_coords(self, encoded_coords, edge_index, batch, repeated_cond=None, return_node_emb=False):
        encoded_coords = self.shared_decoder(encoded_coords, batch, repeated_cond)
        coords_pred = self.decoderCoords(encoded_coords, edge_index, batch)
        coords_pred = self.proj_out_pos(coords_pred)
        if return_node_emb:
            return coords_pred, encoded_coords
        return coords_pred

    def decode_composition(self, encoded_coords, batch, repeated_cond=None):
        encoded_coords = self.shared_decoder(encoded_coords, batch, repeated_cond)
        return self.compositionHead(encoded_coords)

    def decode_semantic(self, semantic_emb, batch, return_emb=False, repeated_cond=None):
        y_pred = self.proj_out_semantic_pred(semantic_emb)
        if return_emb:
            return y_pred, semantic_emb
        return y_pred

    def forward(self, latent, disentangled_zs, num_nodes, batch, node_mask=None, sample_mode=False, condition=None):
        if self.is_condition:
            if condition is None:
                raise ValueError("Condition is None, but is_condition is True")
            assert condition.shape[0] == latent.shape[0] and condition.shape[1] == self.condition_dim, f"condition shape {condition.shape} is not valid"
            latent = torch.cat([latent, condition], dim=-1)
        node_num_logit, node_num_pred, lengths_pred, angles_pred = self.decode_latent_global(latent)

        repeated_latent = torch.repeat_interleave(latent, repeats=num_nodes, dim=0)
        # edge_index = radius_graph(repeated_latent, r=5.0)
        coords_pred = self.decode_coords(disentangled_zs[0], None, batch, repeated_cond=repeated_latent)

        # edge_logits, edge_index_list, node_mask = self.decode_edges(disentangled_zs[0], num_nodes, batch, coords_pred, repeated_cond=repeated_latent)
        y_pred = self.decode_semantic(disentangled_zs[1], batch, repeated_cond=latent)
        comp_logits = self.decode_composition(disentangled_zs[0], batch, repeated_cond=repeated_latent)

        if sample_mode:
            return node_num_pred, lengths_pred, angles_pred, coords_pred,  y_pred, comp_logits
        else:
            return node_num_logit, lengths_pred, angles_pred, coords_pred, y_pred, comp_logits


class GeomVAE(nn.Module):
    def __init__(self, normalizer, max_node_num=100, latent_dim=128, edge_sample_threshold=0.5, is_variational=False,
                 is_disent_variational=True, is_condition=False, condition_dim=12, disentangle_same_layer=False, composition_dim=119):
        super().__init__()
        self.is_disent_variational = is_disent_variational
        self.max_node_num = max_node_num
        self.is_condition = is_condition
        self.condition_dim = condition_dim
        self.encoder = GeomEncoder(latent_dim=latent_dim, is_condition=False, condition_dim=condition_dim)
        self.decoder = GeomDecoder(latent_dim=latent_dim, max_node_num=max_node_num, edge_sample_threshold=edge_sample_threshold, is_condition=is_condition, condition_dim=condition_dim, composition_dim=composition_dim)
        self.normalizer = normalizer
        self.latent_dim = latent_dim

        self.is_variational = is_variational
        if is_variational:
            self.fc_mu = nn.Linear(latent_dim, latent_dim)
            self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        if disentangle_same_layer:
            shared_layer = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim)
            )
            self.proj_in_edge = shared_layer
            self.proj_in_pos = shared_layer
            self.proj_in_semantic = shared_layer
        else:
            self.proj_in_edge = nn.Linear(latent_dim, latent_dim)
            self.proj_in_pos = nn.Linear(latent_dim, latent_dim)
            self.proj_in_semantic = nn.Linear(latent_dim, latent_dim)
        self.proj_in_semantic1 = nn.Linear(latent_dim, latent_dim)

        if self.is_disent_variational:
            self.fc_mu_coords = nn.Linear(latent_dim, latent_dim)
            self.fc_logvar_coords = nn.Linear(latent_dim, latent_dim)

            self.fc_mu_edge = nn.Linear(latent_dim, latent_dim)
            self.fc_logvar_edge = nn.Linear(latent_dim, latent_dim)

            self.fc_semantic = nn.Linear(latent_dim, latent_dim)
            self.fc_logvar_semantic = nn.Linear(latent_dim, latent_dim)

    def encode(self, z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms):
        semantic_latent, geo_coords_latent = self.encoder(
            z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms
        )
        return semantic_latent, geo_coords_latent


    def reparameterize(self, latent, tag='lattice', noise_scale=1.0):   
        if tag == 'lattice':
            mu = self.fc_mu(latent)  # shape=[B, latent_dim]
            logvar = self.fc_logvar(latent)  # shape=[B, latent_dim]
        elif tag == 'edge':
            mu = self.fc_mu_edge(latent)
            logvar = self.fc_logvar_edge(latent)
        elif tag == 'semantic':
            mu = self.fc_semantic(latent)
            logvar = self.fc_logvar_semantic(latent)
        elif tag == 'coords':
            mu = self.fc_mu_coords(latent)
            logvar = self.fc_logvar_coords(latent)
        else:
            raise ValueError(f"Unknown tag for reparameterize: {tag}")

        logvar = torch.clamp(logvar, -10, 10)
        std = torch.exp(0.5 * logvar)
        # std = F.softplus
        eps = torch.randn_like(std)
        latent = mu + noise_scale * eps * std
        return latent, mu, logvar



    def disentangle_latent(self, z0, batch=None):
        """
        Disentangle the latent space into edge, position, and semantic components.
        """
        assert batch is not None, "batch must be provided for disentangle_latent"
        # z0_e = self.proj_in_edge(z0)
        z0_p = self.proj_in_pos(z0)
        z0_s = self.proj_in_semantic(z0)
        z0_s = scatter_add(z0_s, batch, dim=0)  # shape=[B, max_node_num, latent_dim]
        z0_s = self.proj_in_semantic1(z0_s)

        return z0_p, z0_s

    def forward(self, z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms, condition=None, geo_noise_scale = 1.0):
        # 1) encoding
        latent_global, encoded_coords = self.encoder(
            z, coords, edge_index, batch,
            lengths_normed, angles_normed, num_atoms, condition=None
        )  
        # print('Encoded_coords.std', encoded_coords.std(dim=0).mean())
        if self.is_variational:
            z_latent, mu, logvar = self.reparameterize(latent_global, tag='lattice')  # shape=[B, latent_dim]
        else:
            z_latent = latent_global
            # placeholders for analysis/consistency
            mu = torch.zeros_like(latent_global)
            logvar = torch.zeros_like(latent_global)

        z0_p, z0_s = self.disentangle_latent(encoded_coords, batch)


        # defaults for analysis and optional return
        mu_geo = [torch.zeros_like(z0_p), torch.zeros_like(z0_s)]
        logvar_geo = [torch.zeros_like(z0_p), torch.zeros_like(z0_s)]

        if self.is_disent_variational:
            z0_p, mu_coords, logvar_coords = self.reparameterize(z0_p, tag='coords', noise_scale=geo_noise_scale)
            # z0_e, mu_edge, logvar_edge = self.reparameterize(z0_e, tag='edge', noise_scale=geo_noise_scale)
            z0_s, mu_semantic, logvar_semantic = self.reparameterize(z0_s, tag='semantic', noise_scale=geo_noise_scale)
            mu_geo = [mu_coords, mu_semantic]
            logvar_geo = [logvar_coords, logvar_semantic]
        disent_zs = (z0_p, z0_s)
        
        # print('Decodding...')
        node_num_logit, lengths_pred, angles_pred, coords_pred, y_pred, comp_logits = self.decoder(
            z_latent, disent_zs, num_atoms, batch, node_mask=None, sample_mode=False, condition=condition
        )

        if self.is_variational:
            return (node_num_logit, lengths_pred, angles_pred, coords_pred, y_pred, comp_logits), (mu, logvar, mu_geo, logvar_geo)
        else:
            return node_num_logit, lengths_pred, angles_pred, coords_pred, y_pred, comp_logits


    def compute_loss(self, batch_data, device='cuda', beta=1.0, beta_geo=1.0):
        z, coords, edge_index, batch = batch_data.node_type.to(device), batch_data.cart_coords.to(device), batch_data.edge_index.to(device), batch_data.batch.to(device)
        lengths, angles = batch_data.lengths.to(device), batch_data.angles.to(device)
        num_atoms, num_edges = batch_data.num_atoms.to(device), batch_data.num_edges.to(device)
        y_true = batch_data.y.to(device)
        # placeholders for variational stats to appease static analyzers; used only when flags are True
        mu = torch.zeros((z.size(0), self.latent_dim), device=device)
        logvar = torch.zeros_like(mu)
        mu_geo = [torch.zeros((z.size(0), self.latent_dim), device=device) for _ in range(2)]
        logvar_geo = [torch.zeros_like(mu_geo[0]) for _ in range(2)]

        if self.is_condition:
            condition = y_true
            assert condition.shape[0] == num_atoms.shape[0] and condition.shape[1] == self.condition_dim, f"condition shape {condition.shape} is not valid"
        else:
             condition=None

        lengths_normed, angles_normed = self.normalizer(lengths, angles)

        if self.is_variational:
            (node_num_logit, lengths_pred, angles_pred, coords_pred, y_pred, comp_logits), (mu, logvar, mu_geo, logvar_geo) = self(
                z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms, condition=condition
            )
        else:
            node_num_logit, lengths_pred, angles_pred, coords_pred, y_pred, comp_logits = self(
                z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms, condition=condition
            )

        loss_l = F.mse_loss(lengths_pred, lengths_normed)
        loss_a = F.mse_loss(angles_pred, angles_normed)
        # coords_target, node_mask = padding_graph2max_num_node(coords, batch, max_num_node=num_atoms.max())
        # loss_coords = torch.sum((coords_pred - coords_target) ** 2).sqrt() / node_mask.sum()
        loss_coords = F.mse_loss(coords_pred, coords)
        # loss_edge = compute_edge_bce_loss_with_pos_weight(edge_logits, node_mask, edge_index, num_edges, is_weight=True)
        node_num_target = F.one_hot(num_atoms - 1, num_classes=self.max_node_num).float()
        loss_node_num = F.cross_entropy(node_num_logit, node_num_target)
        loss_y = F.mse_loss(y_pred, y_true)

        # Composition supervision: per-node element classification
        comp_dim = comp_logits.size(-1)
        if z.dim() == 1 or (z.dim() == 2 and z.size(1) == 1):
            # integer labels per node
            z_idx = z.view(-1).long()
            loss_comp = F.cross_entropy(comp_logits, z_idx)
        elif z.dim() == 2 and z.size(1) == comp_dim:
            # one-hot or soft labels per node
            z_sum = z.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            z_prob = z / z_sum
            comp_log_prob = F.log_softmax(comp_logits, dim=-1)
            loss_comp = F.kl_div(comp_log_prob, z_prob, reduction='batchmean')
        else:
            # fallback: argmax to hard labels
            z_idx = torch.argmax(z, dim=-1).long()
            loss_comp = F.cross_entropy(comp_logits, z_idx)

        recon_loss = (loss_l + loss_a) + 1000 * loss_coords  + loss_node_num + loss_comp

        if self.is_variational:
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_divergence = kl_divergence / z.size(0)
        else:
            kl_divergence = torch.tensor(0.0, device=device)

        if self.is_disent_variational:
            kl_divergence_coords = -0.5 * torch.sum(1 + logvar_geo[0] - mu_geo[0].pow(2) - logvar_geo[0].exp())
            kl_divergence_coords = kl_divergence_coords / z.size(0)
            # kl_divergence_edge = -0.5 * torch.sum(1 + logvar_geo[0] - mu_geo[0].pow(2) - logvar_geo[0].exp())
            # kl_divergence_edge = kl_divergence_edge / z.size(0)
            kl_divergence_semantic = -0.5 * torch.sum(1 + logvar_geo[1] - mu_geo[1].pow(2) - logvar_geo[1].exp())
            kl_divergence_semantic = kl_divergence_semantic / z.size(0)
        else:
            kl_divergence_coords = torch.tensor(0.0, device=device)
            # kl_divergence_edge = torch.tensor(0.0, device=device)
            kl_divergence_semantic = torch.tensor(0.0, device=device)

        if recon_loss > 100:
            b1 = 0.01
        else:
            b1 = beta
        loss = recon_loss + loss_y + b1 * kl_divergence  + beta_geo * (kl_divergence_coords + kl_divergence_semantic)

        loss_dict = {
            'total_loss': loss,
            'recon_loss': recon_loss,
            'kl_divergence': kl_divergence,
            'kl_divergence_coords': kl_divergence_coords,
            # 'kl_divergence_edge': kl_divergence_edge,
            'kl_divergence_semantic': kl_divergence_semantic,
            'loss_lattice_len': loss_l,
            'loss_lattice_ang': loss_a,
            'loss_coords': loss_coords,
            # 'loss_edge': loss_edge,
            'loss_node_num': loss_node_num,
            'loss_y': loss_y,
            'loss_comp': loss_comp,
            
        }

        return loss_dict

    def train_model(self, dataloader, optimizer, device='cuda', num_epochs=10, beta=1.0, beta_geo=1.0,
                    scheduler=None, checkpoint_dir='./checkpoints', save_every=5, use_wandb=False):

        os.makedirs(checkpoint_dir, exist_ok=True)
        best_loss = float('inf')
        for epoch in range(num_epochs):
            self.train()

            total_loss = total_kld = total_recon = 0.0
            total_kld_coords = total_kld_edge = total_kld_semantic = 0.0
            total_lattice_len = total_lattice_ang = total_coords = 0.0
            total_edge = total_node_num = total_y = total_comp = 0.0
            num_batches = 0
            epoch_time_start = time.time()
            for batch_data in dataloader:
                loss_dict =  self.compute_loss(batch_data, device=device, beta=beta, beta_geo=beta_geo)
                loss = loss_dict['total_loss']
                recon_loss = loss_dict['recon_loss']
                kl_divergence = loss_dict['kl_divergence']
                kl_divergence_coords = loss_dict['kl_divergence_coords']
                # kl_divergence_edge = loss_dict['kl_divergence_edge']
                kl_divergence_semantic = loss_dict['kl_divergence_semantic']
                loss_l = loss_dict['loss_lattice_len']
                loss_a = loss_dict['loss_lattice_ang']
                loss_coords = loss_dict['loss_coords']
                # loss_edge = loss_dict['loss_edge']
                loss_node_num = loss_dict['loss_node_num']
                loss_y = loss_dict['loss_y']
                loss_comp = loss_dict['loss_comp']

                # 反向传播
                if torch.any(torch.isnan(loss)):
                    print('loss is nan')
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kld += kl_divergence.item()
                # total_kld_edge += kl_divergence_edge.item()
                total_kld_semantic += kl_divergence_semantic.item()
                total_kld_coords += kl_divergence_coords.item()
                total_lattice_len += loss_l.item()
                total_lattice_ang += loss_a.item()
                total_coords += loss_coords.item()
                # total_edge += loss_edge.item()
                total_node_num += loss_node_num.item()
                total_y += loss_y.item()
                total_comp += loss_comp.item()
                num_batches += 1


            # 更新学习率调度器
            if scheduler is not None:
                try:
                    scheduler.step()
                except Exception:
                    pass

            epoch_time_end = time.time()
            # 平均损失
            avg_loss = total_loss / num_batches
            avg_recon = total_recon / num_batches
            avg_kld = total_kld / num_batches
            avg_kld_coords = total_kld_coords / num_batches
            # avg_kld_edge = total_kld_edge / num_batches
            avg_kld_semantic = total_kld_semantic / num_batches
            avg_len = total_lattice_len / num_batches
            avg_ang = total_lattice_ang / num_batches
            avg_coord = total_coords / num_batches
            # avg_edge = total_edge / num_batches
            avg_node_num = total_node_num / num_batches
            avg_y = total_y / num_batches
            avg_comp = total_comp / num_batches

            ptr_str = (f"Epoch [{epoch + 1:02d}/{num_epochs}]"
                  f" | Loss={avg_loss:.4f}"
                  f" | Recon={avg_recon:.4f}"
                  f" || KL_lat={avg_kld:.4f}"
                  f"  KL_coords={avg_kld_coords:.4f}"
                    # f"  KL_edge={avg_kld_edge:.4f}"
                    f"  KL_semantic={avg_kld_semantic:.4f}"
                  f" || L_len={avg_len:.4f}"
                  f"  L_ang={avg_ang:.4f}"
                  f"  L_coords={avg_coord:.4f}"
                #   f"  L_edge={avg_edge:.4f}"
                  f"  L_nnum={avg_node_num:.4f}"
                                    f"  L-y={avg_y:.4f}"
                                    f"  L_comp={avg_comp:.4f}"
                  f" | Time={(epoch_time_end-epoch_time_start):4f}s")
            print(ptr_str)
            if use_wandb:
                curr_lr = None
                if scheduler is not None:
                    try:
                        curr_lr = scheduler.get_last_lr()[0]
                    except Exception:
                        curr_lr = optimizer.param_groups[0]['lr']
                else:
                    curr_lr = optimizer.param_groups[0]['lr']
                log_vals = {
                    'epoch': epoch + 1,
                    'avg_loss': avg_loss,
                    'avg_recon': avg_recon,
                    'avg_kld': avg_kld,
                    'avg_kld_coords': avg_kld_coords,
                    # 'avg_kld_edge': avg_kld_edge,
                    'avg_kld_semantic': avg_kld_semantic,
                    'avg_len': avg_len,
                    'avg_ang': avg_ang,
                    'avg_coord': avg_coord,
                    # 'avg_edge': avg_edge,
                    'avg_node_num': avg_node_num,
                    'avg_y': avg_y,
                    'avg_comp': avg_comp,
                    'lr': curr_lr,
                }
                wandb.log(log_vals)
            if avg_loss < best_loss:
                best_loss = avg_loss
                print('Saving checkpoint to ', os.path.join(checkpoint_dir, 'best_ae_model.pt'))
                torch.save(self.state_dict(), os.path.join(checkpoint_dir, 'best_ae_model.pt'))

            if (epoch + 1) % save_every == 0:
                torch.save(self.state_dict(), os.path.join(checkpoint_dir, f'model_epoch{epoch + 1}.pt'))


    @torch.no_grad()
    def sample_lattice(self, num_samples=4, z_latent=None, condition=None, device='cuda', random_scale_latent=1., random_scale_cond=1e-3):
        self.eval()
        self.to(device)

        # sample z_vae ∼ N(0, I)
        if z_latent is None:
            z_latent = random_scale_latent * torch.randn((num_samples, self.latent_dim), device=device)
            if self.is_condition:
                if condition is None:
                    condition = random_scale_cond * torch.randn((num_samples, self.condition_dim), device=device)
        else:
            z_latent = z_latent.to(device)
            z_latent = torch.repeat_interleave(z_latent, repeats=num_samples, dim=0)
            z_latent = self.reparameterize(z_latent)[0]  # shape=[num_samples, latent_dim]
    
        if self.is_condition:
            assert condition is not None, "Condition must be provided when is_condition=True"
            z_latent = torch.cat([z_latent, condition], dim=-1)
            assert z_latent.shape[1] == self.latent_dim + self.condition_dim, f"z_latent shape {z_latent.shape} is not valid"
        else:
            assert z_latent.shape[1] == self.latent_dim, f"z_latent shape {z_latent.shape} is not valid"

        # Predict lattice
        node_num_logit, node_num_pred, lengths_pred, angles_pred = self.decoder.decode_latent_global(
            z_latent,
        )

        return node_num_logit, node_num_pred, lengths_pred, angles_pred, z_latent


    @torch.no_grad()
    def sample(self, num_samples=1, z_latent=None, encoded_coords=None, batch=None, device='cuda', condition=None, random_scale_geo=0.1, random_scale_latent=1.0, random_scale_cond=1e-3):
        self.eval()
        self.to(device)
        if z_latent is not None or encoded_coords is not None:
            assert z_latent is not None or encoded_coords is not None, "z_latent and encoded_coords are both None"

        node_num_logit, node_num_pred, lengths_pred, angles_pred, latent = self.sample_lattice(num_samples=num_samples, z_latent=z_latent, condition=condition, device=device, random_scale_latent=random_scale_latent, random_scale_cond=random_scale_cond)
        if batch is None:
            batch = torch.repeat_interleave(torch.arange(num_samples, device=device), node_num_pred, dim=0)
        if encoded_coords is not None:
            if node_num_pred.sum() != encoded_coords.shape[0]:
                raise ValueError(f"node_num_pred {node_num_pred.sum()} does not match encoded_coords {encoded_coords.shape[0]}")
            z0_p, z0_s = self.disentangle_latent(encoded_coords, batch)
        else:
            total_nodes = int(node_num_pred.sum().item())
            z0_p = torch.randn((total_nodes, self.latent_dim), device=device)
            z0_s = torch.randn((total_nodes, self.latent_dim), device=device)

        if self.is_disent_variational:
            z0_p, mu_coords, logvar_coords = self.reparameterize(z0_p, tag='coords', noise_scale=random_scale_geo)
            # z0_e, mu_edge, logvar_edge = self.reparameterize(z0_e, tag='edge', noise_scale=random_scale_geo)
            z0_s, mu_semantic, logvar_semantic = self.reparameterize(z0_s, tag='semantic', noise_scale=random_scale_geo)
            mu_geo = [mu_coords, mu_semantic]
            logvar_geo = [logvar_coords, logvar_semantic]
        disentangled_zs = (z0_p, z0_s)

        # print('Decodding...')

        ## GPS
        repeated_latent = torch.repeat_interleave(latent, repeats=node_num_pred, dim=0)
        assert disentangled_zs[1].shape[0] == repeated_latent.shape[0], f"disentangled_zs[1] shape {disentangled_zs[1].shape} does not match latent shape {repeated_latent.shape}"
        repeated_latent = torch.repeat_interleave(latent, repeats=node_num_pred, dim=0)
        coords_pred = self.decoder.decode_coords(disentangled_zs[0], None, batch, repeated_cond=repeated_latent)
        # edge_logits, edge_index_list, node_mask = self.decoder.decode_edges(disentangled_zs[0], node_num_pred, batch, coords_pred,  repeated_cond=repeated_latent)
        y_pred = self.decoder.decode_semantic(disentangled_zs[1], batch, repeated_cond=latent)

        coords_pred_list = []
        nnum_offset = 0
        for i, nnum in enumerate(node_num_pred):
            coords_pred_list.append(coords_pred[nnum_offset:nnum_offset + nnum])
            nnum_offset += nnum

        return node_num_pred, lengths_pred, angles_pred, coords_pred_list, None, y_pred

    @torch.no_grad()
    def decode_from_encoded(self, new_geo_latent, new_lattice_latent, batch, condition, random_scale_geo=1, random_scale_latent=1.0, random_scale_cond=1e-3):
        num_samples = new_lattice_latent.shape[0]
        assert new_geo_latent.shape[1] == self.latent_dim, f"new_geo_latent shape {new_geo_latent.shape} is not valid"
        device = new_geo_latent.device
        if condition is None:
            condition = random_scale_cond * torch.randn((num_samples, self.condition_dim), device=device)
        self.eval()
        z_latent = new_lattice_latent
        # z_latent = semantic_latent
        # self.decoder.edge_sample_threshold = 0.5
        num_atoms = torch.LongTensor([new_geo_latent.shape[0]]).to(device)

        z0_p_new, z0_s_new = self.disentangle_latent(new_geo_latent, batch=batch)
        # z_e_new, mu_e_new, logvar_e_new = self.reparameterize(z0_e_new, tag='edge', noise_scale=random_scale_geo)
        if self.is_disent_variational:
            z_p_new, mu_p_new, logvar_p_new = self.reparameterize(z0_p_new, tag='coords', noise_scale=random_scale_geo)
            z_s_new, mu_s_new, logvar_s_new = self.reparameterize(z0_s_new, tag='semantic', noise_scale=random_scale_geo)
        else:
            z_p_new = z0_p_new
            z_s_new = z0_s_new

        
        # new z_latent from sample_lattice is the reparameterized and concatenated with condition
        node_num_logit, node_num_pred, lengths_pred, angles_pred, z_latent = self.sample_lattice(num_samples=num_samples, z_latent=z_latent, condition=condition, device=device, random_scale_latent=1.0, random_scale_cond=1.0)
        lengths_pred, angles_pred = self.normalizer.denormalize(lengths_pred, angles_pred)
        # print(f'node_num_pred:{node_num_pred}, lengths_pred:{lengths_pred}, angles_pred:{angles_pred}')
        if node_num_pred.item() > num_atoms.item():
            node_num_pred = num_atoms # directly use the original node num
            # print(f'predicted node num {node_num_pred} is larger than original node num {num_atoms}, padding the latent')
            # z_p_new = torch.cat([z_p_new, torch.randn((node_num_pred.item() - num_atoms, self.latent_dim), device=device) * random_scale_geo], dim=0)
            # z_e_new = torch.cat([z_e_new, torch.randn((node_num_pred.item() - num_atoms, self.latent_dim), device=device) * random_scale_geo], dim=0)
            # z_s_new = torch.cat([z_s_new, torch.randn((node_num_pred.item() - num_atoms, self.latent_dim), device=device) * random_scale_geo], dim=0)
            # batch = torch.cat([batch, torch.full((node_num_pred.item() - num_atoms,), 0, device=device)])
        elif node_num_pred.item() < num_atoms.item():
            # node_num_pred = num_atoms # directly use the original node num
            selected_index = farthest_point_sampling(z_p_new, node_num_pred.item())
            # selected_index = torch.randperm(num_atoms)[:node_num_pred] # randomly select node_num_pred nodes
            # print(z_s_new.pow(2).mean(dim=1, keepdim=True).shape, node_num_pred.item())
            # selected_index = torch.topk(z_s_new.pow(2).mean(dim=1, keepdim=True), k=node_num_pred.item(), largest=True).indices # select the top node_num_pred nodes based on the semantic latent
            z_p_new = z_p_new[selected_index]
            # z_e_new = z_e_new[selected_index]
            batch = batch[selected_index]

        disentangled_zs = (z_p_new, z_s_new)

        repeated_latent = torch.repeat_interleave(z_latent, repeats=node_num_pred, dim=0)
        assert disentangled_zs[0].shape[0] == repeated_latent.shape[0], f"disentangled_zs[1] shape {disentangled_zs[1].shape} does not match latent shape {repeated_latent.shape}"
        coords_pred = self.decoder.decode_coords(disentangled_zs[0], None, batch, repeated_cond=repeated_latent)
        # edge_logits, edge_index_list, node_mask = self.decoder.decode_edges(disentangled_zs[0], node_num_pred, batch, coords_pred,  repeated_cond=repeated_latent)
        y_pred = self.decoder.decode_semantic(disentangled_zs[1], batch, repeated_cond=repeated_latent)

        coords_pred_list = []
        nnum_offset = 0
        # print(f'node_num_pred:{node_num_pred}')
        for i, nnum in enumerate(node_num_pred):
            coords_pred_list.append(coords_pred[nnum_offset:nnum_offset + nnum])
            nnum_offset += nnum
        return node_num_pred, lengths_pred, angles_pred, coords_pred_list, None, y_pred

        # visualizeLattice(coords_pred_list[0].detach().cpu().numpy(), edge_index_list[0].cpu().numpy())

