"""GCN model implementation using composition-based convolution."""

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_add, scatter_max

from CompGCN.compgcn_conv import CompGCNConv
from CompGCN.message_passing import MessagePassing
from loader import QueryBatch, VectorizedQueryBatch


class GCNModel(nn.Module):
    def __init__(self, embed_dim, num_layers, readout, device=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.device = device
        assert readout in ['max', 'sum', 'TM']

        # get node embeddings
        # create relation embeddings (normal + inverse relations)
        
        # create message passing layers
        layers = []
        for _ in range(num_layers - 1):
            layers += [
                CompGCNConv(
                    self.embed_dim,
                    self.embed_dim,
                    ),
                nn.ReLU()
            ]
        layers += [
            CompGCNConv(
                self.embed_dim,
                self.embed_dim,
                )
        ]
        self.layers = nn.ModuleList(layers)

        # define readout
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

    def target_message_readout(self, embs, batch_size, num_nodes, num_anchors, **kwargs):
        device = embs.device

        non_target_idx = torch.ones(num_nodes, dtype=torch.bool)
        non_target_idx[num_anchors] = 0
        non_target_idx.to(device)

        embs = embs.reshape(batch_size, num_nodes, -1)
        targets = embs[:, ~non_target_idx].reshape(batch_size, -1)

        return targets

    def vectorize_batch(self, batch: QueryBatch) -> VectorizedQueryBatch:
        """Converts IDs to embeddings and vectorizes query graphs."""
        
        # empty tensors
        ent_e       = torch.empty(batch.batch_size, batch.num_entities, self.embed_dim).to(self.device)
        edge_index  = torch.empty(batch.batch_size, 2, batch.num_edges).to(self.device)
        edge_type   = torch.empty(batch.batch_size, batch.num_edges).to(self.device)
        rel_e       = torch.empty(batch.batch_size, 2 * batch.num_relations, self.embed_dim).to(self.device)

        # fill tensors
        # ...
        
        return VectorizedQueryBatch(
            ent_e=ent_e,
            edge_index=edge_index,
            edge_type=edge_type,
            rel_e=rel_e
        )

    def forward(self, batch: QueryBatch) -> Tensor:

        # get and unpack vectorized data
        data: VectorizedQueryBatch = self.vectorize_batch(batch)
        ent_e, edge_index, edge_type, rel_e = data.ent_e, data.edge_index, data.edge_type, data.rel_e

        # perform message passing
        for layer in self.layers:
            if isinstance(layer, MessagePassing):
                ent_e, rel_e = layer(ent_e, edge_index, edge_type, rel_e)
            else:
                ent_e = layer(ent_e)

        # aggregate node embeddings
        out = self.readout(
            embs        =   ent_e,
            batch_idx   =   data.batch_idx, # TODO: include these in either batch or data.
            batch_size  =   data.batch_size,
            num_nodes   =   data.num_nodes,
            num_anchors =   data.num_anchors
            )
        return out

