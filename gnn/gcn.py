"""GCN model implementation using composition-based convolution."""

import torch
import torch.nn as nn
from torch import Tensor

from loader import VectorizedQueryBatch
from gnn.compgcn_conv import CompGCNConv
from gnn.message_passing import MessagePassing
from gnn.pooling import *


class GCNModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        stop_at_diameter: bool = True,
        pool: str = 'max',
        comp: str = 'mult',
        use_bias: bool = True,
        use_bn: bool = True,
        dropout: float = 0.0,
        device=None
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.stop_at_diameter = stop_at_diameter
        assert pool in ['max', 'sum', 'tm'], f"Readout function {pool} is not implemented."
        self.pool_str = pool
        assert comp in ['sub', 'mult', 'cmult', 'cconv', 'ccorr', 'crot'], f"Composition operation {comp} is not implemented."
        self.comp_str = comp
        self.use_bias = use_bias
        self.dropout = dropout
        self.device = device       
        
        # create message passing layers (sharing weights)
        layers = []
        conv = CompGCNConv(
            self.embed_dim,
            self.embed_dim,
            comp=comp,
            use_bias=use_bias,
            use_bn=use_bn,
            dropout=dropout
            )
        for _ in range(num_layers - 1):
            layers += [conv, nn.ReLU()]
        layers += [conv]
        self.layers = nn.ModuleList(layers)

        # define readout function
        if pool == 'max':
            self.pool = MaxGraphPooling()
        elif pool == 'sum':
            self.pool = SumGraphPooling()
        elif pool == 'tm':
            self.pool = TargetPooling()
        else:
            raise NotImplementedError


    def forward(self, data: VectorizedQueryBatch) -> Tensor:
        
        ent_embed, rel_embed, diameters = data.ent_embed, data.rel_embed, data.q_diameters

        # perform message passing
        convs = 0
        for layer in self.layers:
            
            # convolution layer
            if isinstance(layer, MessagePassing):
                
                # include boolean mask for entities where query diameter <= conv step
                ent_mask = torch.le(diameters, convs) if self.stop_at_diameter else None
                ent_embed, rel_embed = layer(ent_embed, rel_embed, data.edge_index, data.edge_type, ent_mask=ent_mask)
                convs += 1
            
            # otherwise its activation function
            else:
                ent_embed = layer(ent_embed)

        # aggregate node embeddings
        out = self.pool(
            embs        =   ent_embed,
            batch_idx   =   data.batch_idx,
            target_idx  =   data.target_idx
            )
        return out
