"""GCN model implementation using composition-based convolution."""

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_add, scatter_max

from CompGCN.compgcn_conv import CompGCNConv
from CompGCN.message_passing import MessagePassing
from loader import QueryBatch, VectorizedQueryBatch


class GCNModel(nn.Module):
    def __init__(self, data_dir, embed_dim, num_layers, readout, device=None):
        super().__init__()
        self.data_dir = data_dir
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.device = device
        assert readout in ['max', 'sum', 'TM']

        # create embeddings and lookup functions
        self.ent_features, self.rel_features, self.embed_ents, self.embed_rels = \
            self.build_embeddings(data_dir, embed_dim)
        
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

    def build_embeddings():
        """Builds embeddings for both entities (including variables) and relations."""
        return

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
        """Converts batch data with global IDs to embeddings."""
        
        batch_idx   = torch.Tensor(batch.batch_idx).to(self.device)
        edge_index  = torch.Tensor(batch.edge_index).to(self.device)
        edge_type   = torch.Tensor(batch.edge_type).to(self.device)
        # extract entity and relation embeddings using lookup functions
        ent_e       = torch.Tensor(self.embed_ents(batch.entity_id, batch.entity_type)).to(self.device)
        rel_e       = torch.Tensor(self.embed_rels(batch.edge_type)).to(self.device)
        
        return VectorizedQueryBatch(
            batch_idx=batch_idx,
            ent_e=ent_e,
            edge_index=edge_index,
            edge_type=edge_type,
            rel_e=rel_e
        )

    def forward(self, batch: QueryBatch) -> Tensor:

        # get and unpack vectorized data
        data: VectorizedQueryBatch = self.vectorize_batch(batch)
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
            batch_size  =   batch.batch_size,  
            num_nodes   =   batch.num_entities,
            num_anchors =   data.num_anchors # don't know...
            )
        return out

