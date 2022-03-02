"""Metamodel implementation."""

import torch

from gcn import GCNModel


class MetaModel(torch.nn.Module):
    def __init__(
        self,
        num_buckets:    int,    # number of buckets
        num_planes:     int,    # numbef of hyperplanes (GCNModels) per bucket
        ):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_planes = num_planes

        # initiate GCNModels
        self.submodels = [
            GCNModel(
                self.embed_dims,
                self.gcn_layers,
                self.num_rels,
                self.gcn_readout
                ) for _ in range(self.num_buckets * self.num_planes)
        ]

    def forward(self, query_graph_batch):
        
        x, edge_index, edge_type, rel_embed, graph_ids = query_graph_batch

        # GCNModel output is matrix with hyperplanes in size [batchsize * embed_dim],
        # we concatenate these along the first axis.
        hyp_matrix = torch.cat((
            [gcn(x, edge_index, edge_type, rel_embed, graph_ids) for gcn in self.submodels]
            ), 1)
        return hyp_matrix
