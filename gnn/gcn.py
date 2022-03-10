"""GCN model implementation using composition-based convolution."""

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_add, scatter_max

from gnn.compgcn_conv import CompGCNConv
from gnn.message_passing import MessagePassing
from loader import VectorizedQueryBatch

class GCNModel(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_layers,
        readout,
        use_bias=True,
        opn='corr',
        dropout=0,
        device=None
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.device = device
        self.use_bias = use_bias
        assert opn in ['corr', 'sub', 'mult'], f"Composition operation {opn} is not implemented."
        self.opn = opn
        self.dropout = dropout
        assert readout in ['max', 'sum', 'TM'], f"Readout function {readout} is not implemented."
        self.readout_str = readout
        
        # create message passing layers
        layers = []
        conv = CompGCNConv(
            self.embed_dim,
            self.embed_dim, 
            use_bias=use_bias,
            opn=opn,
            dropout=dropout
            )
        for _ in range(num_layers - 1):
            layers += [conv, nn.ReLU()]
        layers += [conv]
        self.layers = nn.ModuleList(layers)

        # define readout function
        if readout == 'max':
            self.readout = self.max_readout
        elif readout == 'sum':
            self.readout = self.sum_readout
        elif readout == 'TM':
            self.readout = self.target_message_readout

    def max_readout(self, embs, batch_idx, **kwargs):
        out, argmax = scatter_max(embs, batch_idx, dim=0)
        return out

    def sum_readout(self, embs, batch_idx, **kwargs):
        return scatter_add(embs, batch_idx, dim=0)

    def target_message_readout(self, embs, target_idx, **kwargs):
        return embs[target_idx]

    def forward(self, data: VectorizedQueryBatch) -> Tensor:
        
        ent_embed, rel_embed, diameters = data.ent_embed, data.rel_embed, data.q_diameters

        # perform message passing
        convs = 1
        for layer in self.layers:
            if isinstance(layer, MessagePassing):
                
                # include binary mask for entities where conv step is > query diameter
                ent_mask = torch.tensor((convs > diameters).float(), dtype=torch.long)
                
                ent_embed, rel_embed = layer(ent_embed, rel_embed, data.edge_index, data.edge_type, ent_mask=ent_mask)
                convs += 1
            else:
                ent_embed = layer(ent_embed)

        # aggregate node embeddings
        out = self.readout(
            embs        =   ent_embed,
            batch_idx   =   data.batch_idx,
            target_id   =   data.target_idx
            )
        return out
