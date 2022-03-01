"""GCN model implementation using composition-based convolution"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_add, scatter_max, scatter_mean, scatter_min

from CompGCN.compgcn_conv import CompGCNConv
from CompGCN.message_passing import MessagePassing


class GCNModel(torch.nn.Module):
    def __init__(self, embed_dims, num_layers, num_rels, readout):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_rels = num_rels
        assert readout in ['max', 'sum', 'TM']
        
        layers = []
        for _ in range(num_layers - 1):
            layers += [
                CompGCNConv(embed_dims, embed_dims, num_rels), # find out what the 'params' arg needs!
                nn.ReLU()
            ]
        layers += [
            CompGCNConv(embed_dims, embed_dims, num_rels)
        ]
        self.layers = nn.ModuleList(layers)

        self.readout_str = readout
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

    def target_message_readout(self, embs, batch_size, num_nodes, num_anchors,
                                **kwargs):
        device = embs.device

        non_target_idx = torch.ones(num_nodes, dtype=torch.bool)
        non_target_idx[num_anchors] = 0
        non_target_idx.to(device)

        embs = embs.reshape(batch_size, num_nodes, -1)
        targets = embs[:, ~non_target_idx].reshape(batch_size, -1)

        return targets

    def forward(self, x: Tensor, edge_index: Tensor, edge_type, rel_embed) -> Tensor:       
        for layer in self.layers:
            if isinstance(layer, MessagePassing):
                x = layer(x, edge_index, edge_type, rel_embed)
            else: # ReLU
                x = layer(x)
        out = self.readout(x) # fix arguments
        return out


# test function
if __name__ == "__main__":

    # load data: try batched input
    edge_index, edge_type, rel_embed = None, None, None

    model = GCNModel(
        embed_dims=128,
        num_layers=2,
        num_rels=100,
        readout='sum'
    )

    out = model(edge_index, edge_type, rel_embed)