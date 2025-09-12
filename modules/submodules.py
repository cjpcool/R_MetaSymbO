import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean, scatter_add

from modules.transformer_backbone import Encoder as TransformerEncoder
from modules.gps_backbone import GPSModel
import math
from torch_geometric.nn import GraphNorm

class EdgeDecoderDotProduct(nn.Module):
    def __init__(self, d_model, use_coords=True, coords_dim=128):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        # self.weight = nn.Parameter(torch.empty(d_model, d_model))
        # nn.init.xavier_uniform_(self.weight)
        # nn.init.normal_(self.weight, mean=1.0, std=0.01)
        self.use_coords = use_coords
        if use_coords:
            self.line = nn.Linear(d_model+coords_dim, d_model, bias=False)
        else:
            self.line = nn.Linear(d_model, d_model, bias=False)
	
    def forward(
        self, 
        node_embeds, 
        sample_threshold=0.5, 
        node_mask=None,
        return_logits=False, 
        coords = None
    ):
        B, N, d_model = node_embeds.shape

        if node_mask is not None:
            node_embeds = node_embeds * node_mask

        if self.use_coords and coords is not None:
            node_embeds = self.line(torch.cat([node_embeds, coords], dim=-1))
        else:
            node_embeds = self.line(node_embeds)
        if node_mask is not None:
            node_embeds = node_embeds * node_mask
        left = node_embeds.unsqueeze(2)   
        right = node_embeds.unsqueeze(1) 
        logits = (left * right).sum(dim=-1) * self.scale + self.bias

        if node_mask is not None:
            mask_2d = node_mask.squeeze(-1).unsqueeze(2) & node_mask.squeeze(-1).unsqueeze(1)
            logits = logits.masked_fill(~mask_2d, float('-inf'))

        edge_probs = torch.sigmoid(logits)

        edge_index_list = []
        for b in range(B):
            adjacency = (edge_probs[b] > sample_threshold)
            adjacency = torch.triu(adjacency, diagonal=1)
            rows, cols = adjacency.nonzero(as_tuple=True) 
            edges = torch.stack([rows, cols], dim=0)
            edge_index_list.append(edges)

        if return_logits:
            return logits, edge_index_list
        return edge_probs, edge_index_list



class EdgeDecoderBilinear(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.weight = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self, 
        node_embeds, 
        sample_threshold=0.5, 
        node_mask=None,
        return_logits=False, 
    ):
        B, N, d_model = node_embeds.shape

        if node_mask is not None:
            node_embeds = node_embeds * node_mask

        hw = torch.matmul(node_embeds, self.weight)

        left = node_embeds.unsqueeze(2)   
        right = node_embeds.unsqueeze(1) 
        logits = (left * right).sum(dim=-1) * self.scale + self.bias

        if node_mask is not None:
            mask_2d = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)  # [B, N, N]
            logits = logits.masked_fill(~mask_2d, float('-inf'))

        edge_probs = torch.sigmoid(logits)  # [B, N, N]

        edge_index_list = []
        for b in range(B):
            adjacency = (edge_probs[b] > sample_threshold)
            adjacency = torch.triu(adjacency, diagonal=1)
            rows, cols = adjacency.nonzero(as_tuple=True) 
            edges = torch.stack([rows, cols], dim=0)
            edge_index_list.append(edges)

        if return_logits:
            return logits, edge_index_list
        return edge_probs, edge_index_list



class LatticeNormalizer(nn.Module):
    def __init__(self, lengths=None, angles=None, eps=1e-8):
        super().__init__()
        self.register_buffer("mean_len", torch.zeros(3))
        self.register_buffer("std_len", torch.ones(3))
        self.register_buffer("mean_ang", torch.zeros(3))
        self.register_buffer("std_ang", torch.ones(3))
        self.eps = eps
        if lengths is not None and angles is not None:
            self.fit(lengths, angles)

    def fit(self, lengths, angles):
        if isinstance(lengths, list):
            lengths = torch.cat(lengths, dim=0)
        if isinstance(angles, list):
            angles = torch.cat(angles, dim=0)

        mean_len = lengths.mean(dim=0)  # shape=[3]
        std_len = lengths.std(dim=0)  # shape=[3]

        mean_ang = angles.mean(dim=0)
        std_ang = angles.std(dim=0)

        self.mean_len = mean_len
        self.std_len = std_len
        self.mean_ang = mean_ang
        self.std_ang = std_ang

    def forward(self, lengths, angles):
        if isinstance(lengths, list):
            lengths = torch.cat(lengths, dim=0)
        if isinstance(angles, list):
            angles = torch.cat(angles, dim=0)
        lengths_normed = (lengths - self.mean_len) / (self.std_len + self.eps)
        angles_normed = (angles - self.mean_ang) / (self.std_ang + self.eps)
        return lengths_normed, angles_normed

    def denormalize(self, lengths_normed, angles_normed):
        if isinstance(lengths_normed, list):
            lengths_normed = torch.cat(lengths_normed, dim=0)
        if isinstance(angles_normed, list):
            angles_normed = torch.cat(angles_normed, dim=0)
        lengths = lengths_normed * (self.std_len + self.eps) + self.mean_len
        angles = angles_normed * (self.std_ang + self.eps) + self.mean_ang
        return lengths, angles











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
    def __init__(self, latent_dim=128, time_emb_dim=128, is_condition=False, condition_dim=12,num_layer=8, conv_type=None):
        super().__init__()
        d_model= latent_dim * 2 + condition_dim if is_condition else latent_dim * 2
        self.d_model = d_model
        self.is_condition = is_condition
        self.condition_dim= condition_dim

        self.in_proj = nn.Linear(d_model, latent_dim)  # 

        self.GPSModel = GPSModel(channels=latent_dim, num_layers=num_layer, attn_type='multihead', conv_type='GINConv')
        # self.GPSModel = nn.Sequential(
        #     nn.Linear(latent_dim, latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(latent_dim, latent_dim),
        # )
        # A simple MLP for time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, d_model),
        )

        self.in_proj = nn.Linear(d_model, latent_dim)  # 
        
        self.out_proj = nn.Linear(latent_dim, latent_dim)

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


        
        latent_condition = torch.repeat_interleave(latent_condition, repeats=num_nodes, dim=0)
        x_in = torch.cat([x_in, latent_condition], dim=-1)
        # 2) Combine x_t with time embedding
        x_in = x_t + t_emb_expanded

        x_in = self.in_proj(x_in)  # [B, N, latent]
        encoded = self.GPSModel(x_in, edge_index, batch)
        # encoded = self.GPSModel(x_in, batch=batch)

        noise_pred = self.out_proj(encoded) 

        return noise_pred


class HeadSemantic(nn.Module):
    def __init__(self, latent_dim, out_dim=128):
        super().__init__()
        self.layers = nn.ModuleList()
        self.proj_in = nn.Linear(latent_dim, latent_dim)

        self.proj_out = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, out_dim)
        )
    def forward(self, x, batch):
        x = self.proj_in(x)
        x = scatter_add(x, batch, dim=0, dim_size=batch.max() + 1)
        x = self.proj_out(x)
        return x

class DisentangledDenoise(nn.Module):
    def __init__(self, latent_dim=128, time_emb_dim=128, is_condition=False, condition_dim=12, is_diff_on_coords=False):
        super().__init__()
        self.shared_backbone_e = DenoiseGPS(latent_dim=latent_dim, time_emb_dim=time_emb_dim, num_layer=1, is_condition=is_condition, condition_dim=condition_dim, conv_type=None)
        self.shared_backbone_p = DenoiseGPS(latent_dim=latent_dim, time_emb_dim=time_emb_dim, num_layer=1, is_condition=is_condition, condition_dim=condition_dim, conv_type=None)
        self.shared_backbone_s = DenoiseGPS(latent_dim=latent_dim, time_emb_dim=time_emb_dim, num_layer=1, is_condition=is_condition, condition_dim=condition_dim, conv_type=None)
        # self.shared_backbone = SharedDisentangleLayer(latent_dim=latent_dim, layer_num=3, is_condition=is_condition, condition_dim=condition_dim, conv_type=None)
        # self.shared_backbone = nn.Sequential(
        #     nn.Linear(latent_dim*2, latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(latent_dim, latent_dim)
        # )
        self.is_diff_on_coords = is_diff_on_coords
        self.head_edge = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, latent_dim)
        )
        # self.head_edge = nn.Linear(latent_dim, latent_dim)
        # self.head_edge = GPSModel(channels=latent_dim, num_layers=1, attn_type='multihead', conv_type=None)
        if self.is_diff_on_coords:
            self.head_node_in = nn.Linear(3, latent_dim)
        # self.head_node = GPSModel(channels=latent_dim, num_layers=1, attn_type='multihead', conv_type=None)
        # self.head_node1 = DenoiseGPS(latent_dim=latent_dim, time_emb_dim=time_emb_dim, is_condition=is_condition, num_layer=2, condition_dim=condition_dim, conv_type=None)
        # self.head_node_out = nn.Sequential(nn.Linear(latent_dim, latent_dim//2), nn.ReLU(), nn.Linear(latent_dim//2, 3))
        if self.is_diff_on_coords:
            self.head_node = nn.Sequential(
                nn.Linear(latent_dim, latent_dim*2),
                nn.ReLU(),
                nn.Linear(latent_dim*2, 3)
            )
        else:
            self.head_node = nn.Sequential(
                nn.Linear(latent_dim, latent_dim*2),
                nn.ReLU(),
                nn.Linear(latent_dim*2, latent_dim)
            )
        # self.head_semantics = nn.ModuleList(
        #     [GPSModel(channels=latent_dim, num_layers=1, attn_type='multihead', conv_type=None), 
        #      nn.Sequential(
        #         nn.Linear(latent_dim, latent_dim*2),
        #         nn.ReLU(),
        #         nn.Linear(latent_dim*2, latent_dim)
        #     )]
        # )
        self.head_semantic = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, latent_dim)
        )
    def shared_noise_pred(self, x_t, batch, num_nodes, t,  latent_condition):
        feat = self.shared_backbone(x_t, None, batch, num_nodes, t, latent_condition)
        # repated_cond = torch.repeat_interleave(latent_condition, num_nodes, dim=0)
        # feat = torch.cat([x_t, repated_cond], dim=-1)
        # feat = self.shared_backbone(feat)
        return feat
    
    def forward(self, denoise_type, x_t, edge_index, batch, num_nodes, t, latent_condition):
        if denoise_type == 'edge':
            # feature = self.shared_noise_pred(x_t, batch, num_nodes, t, latent_condition)
            feature = self.shared_backbone_e(x_t, edge_index, batch, num_nodes, t, latent_condition)
            noise_pred = self.head_edge(feature)
        elif denoise_type == 'coordinate':
            if self.is_diff_on_coords:
                x_t = self.head_node_in(x_t)            
            # feature = self.shared_noise_pred(x_t, batch, num_nodes, t, latent_condition)
            feature = self.shared_backbone_p(x_t, edge_index, batch, num_nodes, t, latent_condition)
            # feature = self.head_node1(feature, None, batch, num_nodes, t, latent_condition)
            noise_pred = self.head_node(feature)

        elif denoise_type == 'semantic':
            # feature = self.shared_noise_pred(x_t, batch, num_nodes, t, latent_condition)
            feature = self.shared_backbone_s(x_t, edge_index, batch, num_nodes, t, latent_condition)
            noise_pred = self.head_semantic(feature)
            # for semantic in self.head_semantics:
            #     feature = semantic(feature)
            # noise_pred = feature
        return noise_pred
    


class SharedDisentangleLayer(nn.Module):
    def __init__(self, latent_dim=128, layer_num=3, is_condition=False, condition_dim=12, conv_type=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.is_condition = is_condition
        self.condition_dim = condition_dim
        if self.is_condition:
            conv_dim = latent_dim * 2 + condition_dim
        else:
            conv_dim = latent_dim * 2
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(
                GPSModel(channels=latent_dim, num_layers=1, attn_type='multihead', conv_type=None)
                )
        self.layers = layers
        self.proj_in_layers = nn.ModuleList()
        for i in range(layer_num):
            self.proj_in_layers.append(
                nn.Linear(conv_dim, latent_dim)
                )
        self.layer_num = layer_num
        
        self.graph_norm = GraphNorm(latent_dim)
        self.act = nn.Softplus()
        self.layer_out= nn.Linear(latent_dim, latent_dim)


    def forward(self, x, batch, latent_cond):
        h = x.clone()
        for i in range(self.layer_num):
            h = torch.cat([h, latent_cond], dim=-1)
            h = self.proj_in_layers[i](h)
            h = self.layers[i](h, None, batch) + h
            h = self.graph_norm(h)
            h = self.act(h)

        out = self.layer_out(h) + x
        return out



from diffusers.models.embeddings import TimestepEmbedding, Timesteps


class ScoreGPS(nn.Module):
    def __init__(self,
                 latent_dim      = 128,
                 time_emb_dim    = 128,          # kept for σ-embedding
                 is_condition    = False,
                 condition_dim   = 12,
                 num_layer       = 8,
                 conv_type       = "GINConv"):

        super().__init__()
        self.is_condition = is_condition
        d_model = latent_dim * 2 + condition_dim if is_condition else latent_dim * 2

        # --------- σ (time) embedding ----------
        self.sinu_emb = Timesteps(num_channels=time_emb_dim, flip_sin_to_cos=False, downscale_freq_shift=0 )   # from diffusers
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, d_model),
        )

        # ---------- backbone ----------
        self.in_proj  = nn.Linear(d_model, latent_dim)
        self.GPSModel = GPSModel(channels=latent_dim,
                                 num_layers=num_layer,
                                 attn_type='multihead',
                                 conv_type=conv_type)
        self.out_proj = nn.Linear(latent_dim, latent_dim)       # predict score

    # ---------------------------------------------------------
    def forward(self,
                x_sigma: torch.Tensor,         # (∑N_i, d_model)
                edge_index: torch.Tensor,
                batch: torch.Tensor,
                num_nodes: torch.Tensor,
                sigma: torch.Tensor,           # (B,)  continuous σ
                latent_condition: torch.Tensor = None):

        # 1) σ-embedding  (use log σ like EDM & Karras)
        sigma_emb = self.sinu_emb(torch.log(sigma))
        sigma_emb = self.time_mlp(sigma_emb)                   # (B, latent)

        # expand to node-level
        sigma_emb = torch.repeat_interleave(sigma_emb, repeats=num_nodes, dim=0)

        # 2) prepare input
        if self.is_condition and latent_condition is not None:
            latent_condition = torch.repeat_interleave(latent_condition, repeats=num_nodes, dim=0)
            x_sigma = torch.cat([x_sigma, latent_condition], dim=-1)

        x_in = self.in_proj(x_sigma + sigma_emb)               # add embedding

        # 3) shared GPS backbone
        h = self.GPSModel(x_in, edge_index, batch)

        # 4) predict score  ŝ(xσ,σ)
        score_hat = self.out_proj(h)
        return score_hat




class DisentangledScore(nn.Module):
    def __init__(self, latent_dim=128, time_emb_dim=128, is_condition=False, condition_dim=12, is_diff_on_coords=False):
        super().__init__()
        self.shared_backbone_e = ScoreGPS(latent_dim=latent_dim, time_emb_dim=time_emb_dim, num_layer=1, is_condition=is_condition, condition_dim=condition_dim, conv_type=None)
        self.shared_backbone_p = ScoreGPS(latent_dim=latent_dim, time_emb_dim=time_emb_dim, num_layer=1, is_condition=is_condition, condition_dim=condition_dim, conv_type=None)
        self.shared_backbone_s = ScoreGPS(latent_dim=latent_dim, time_emb_dim=time_emb_dim, num_layer=1, is_condition=is_condition, condition_dim=condition_dim, conv_type=None)
        # self.shared_backbone = SharedDisentangleLayer(latent_dim=latent_dim, layer_num=3, is_condition=is_condition, condition_dim=condition_dim, conv_type=None)
        # self.shared_backbone = nn.Sequential(
        #     nn.Linear(latent_dim*2, latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(latent_dim, latent_dim)
        # )
        self.is_diff_on_coords = is_diff_on_coords
        self.head_edge = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, latent_dim)
        )
        # self.head_edge = nn.Linear(latent_dim, latent_dim)
        # self.head_edge = GPSModel(channels=latent_dim, num_layers=1, attn_type='multihead', conv_type=None)
        if self.is_diff_on_coords:
            self.head_node_in = nn.Linear(3, latent_dim)
        # self.head_node = GPSModel(channels=latent_dim, num_layers=1, attn_type='multihead', conv_type=None)
        # self.head_node1 = DenoiseGPS(latent_dim=latent_dim, time_emb_dim=time_emb_dim, is_condition=is_condition, num_layer=2, condition_dim=condition_dim, conv_type=None)
        # self.head_node_out = nn.Sequential(nn.Linear(latent_dim, latent_dim//2), nn.ReLU(), nn.Linear(latent_dim//2, 3))
        if self.is_diff_on_coords:
            self.head_node = nn.Sequential(
                nn.Linear(latent_dim, latent_dim*2),
                nn.ReLU(),
                nn.Linear(latent_dim*2, 3)
            )
        else:
            self.head_node = nn.Sequential(
                nn.Linear(latent_dim, latent_dim*2),
                nn.ReLU(),
                nn.Linear(latent_dim*2, latent_dim)
            )
        # self.head_semantics = nn.ModuleList(
        #     [GPSModel(channels=latent_dim, num_layers=1, attn_type='multihead', conv_type=None), 
        #      nn.Sequential(
        #         nn.Linear(latent_dim, latent_dim*2),
        #         nn.ReLU(),
        #         nn.Linear(latent_dim*2, latent_dim)
        #     )]
        # )
        self.head_semantic = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, latent_dim)
        )
    def shared_noise_pred(self, x_t, batch, num_nodes, t,  latent_condition):
        feat = self.shared_backbone(x_t, None, batch, num_nodes, t, latent_condition)
        # repated_cond = torch.repeat_interleave(latent_condition, num_nodes, dim=0)
        # feat = torch.cat([x_t, repated_cond], dim=-1)
        # feat = self.shared_backbone(feat)
        return feat
    
    def forward(self, denoise_type, x_t, edge_index, batch, num_nodes, t, latent_condition):
        if denoise_type == 'edge':
            feature = self.shared_noise_pred(x_t, batch, num_nodes, t, latent_condition)
            # feature = self.shared_backbone_e(x_t, edge_index, batch, num_nodes, t, latent_condition)
            noise_pred = self.head_edge(feature)
        elif denoise_type == 'coordinate':
            if self.is_diff_on_coords:
                x_t = self.head_node_in(x_t)            
            feature = self.shared_noise_pred(x_t, batch, num_nodes, t, latent_condition)
            # feature = self.shared_backbone_p(x_t, edge_index, batch, num_nodes, t, latent_condition)
            # feature = self.head_node1(feature, None, batch, num_nodes, t, latent_condition)
            noise_pred = self.head_node(feature)

        elif denoise_type == 'semantic':
            # feature = self.shared_noise_pred(x_t, batch, num_nodes, t, latent_condition)
            # feature = self.shared_backbone_s(x_t, edge_index, batch, num_nodes, t, latent_condition)
            noise_pred = self.head_semantic(feature)
            # for semantic in self.head_semantics:
            #     feature = semantic(feature)
            # noise_pred = feature
        return noise_pred
