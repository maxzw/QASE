"""GCN model implementation using composition-based convolution."""

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_add, scatter_max

from CompGCN.compgcn_conv import CompGCNConv
from CompGCN.message_passing import MessagePassing
from dataclass import VectorizedQueryBatch

class GCNModel(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_layers,
        readout,
        device=None,
        use_bias=True,
        opn='corr',
        dropout=0
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.device = device
        self.use_bias = use_bias
        self.opn = opn
        self.dropout = dropout
        assert readout in ['max', 'sum', 'TM']
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

    def target_message_readout(self, embs, batch_size, num_nodes, target_id, **kwargs):
        device = embs.device

        non_target_idx = torch.ones(num_nodes, dtype=torch.bool)
        non_target_idx[target_id] = 0
        non_target_idx.to(device)

        embs = embs.reshape(batch_size, num_nodes, -1)
        targets = embs[:, ~non_target_idx].reshape(batch_size, -1)

        return targets

    def forward(self, data: VectorizedQueryBatch) -> Tensor:

        ent_e, rel_e = data.ent_e, data.rel_e

        # perform message passing
        for layer in self.layers:
            if isinstance(layer, MessagePassing):
                ent_e, rel_e = layer(ent_e, data.edge_index, data.edge_type, rel_e)
            else:
                ent_e = layer(ent_e)

        # aggregate node embeddings
        out = self.readout(
            embs        =   ent_e,
            batch_idx   =   data.batch_idx,
            batch_size  =   data.batch_size,
            num_nodes   =   data.num_nodes,
            target_id   =   data.target_idx
            )
        return out
