import torch.nn as nn
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torch_geometric.nn import GPSConv, GINEConv, LayerNorm, GINConv 
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)



class GPSModel(nn.Module):
    def __init__(self, channels=128, num_layers=8, attn_type: str ='multihead', attn_kwargs: Dict[str, Any] = None, conv_type: str='GINConv',heads=8):
        super().__init__()

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            if conv_type == 'GINConv':
                conv = GPSConv(channels, GINConv(nn), heads=heads,
                            attn_type=attn_type, attn_kwargs=attn_kwargs)
            elif conv_type == 'GINEConv':
                conv = GPSConv(channels, GINEConv(nn), heads=heads,
                            attn_type=attn_type, attn_kwargs=attn_kwargs)
            elif conv_type == None:
                conv = GPSConv(channels, None, heads=heads,
                            attn_type=attn_type, attn_kwargs=attn_kwargs)
            
            self.convs.append(conv)

       
    def forward(self, x, edge_index=None, batch=None):
        """
        x: node features, shape [num_nodes, d_model]
        edge_index: [2, num_edges]
        batch: [num_nodes] -> graph batch index
        """
        for conv in self.convs:
            x = conv(x, edge_index, batch)
        return x
