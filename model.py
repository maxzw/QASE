"""Metamodel implementation."""

import torch
import torch.nn as nn
from torch import Tensor

from gnn.gcn import GCNModel
from loader import QueryBatch


class MetaModel(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        num_bands=4,
        num_hyperplanes=4,
        gcn_layers=2,
        gcn_readout='sum',
        gcn_use_bias=True,
        gcn_opn='corr',
        gcn_dropout=0,
        device=None
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bands = num_bands
        self.num_hyperplanes = num_hyperplanes
        self.gcn_layers = gcn_layers
        self.gcn_readout = gcn_readout
        self.gcn_use_bias = gcn_use_bias
        self.gcn_opn = gcn_opn
        self.gcn_dropout = gcn_dropout
        self.device = device

        # instantiate GCN models
        self.submodels = nn.ModuleList([
            GCNModel(
                self.embed_dim,
                self.gcn_layers,
                self.gcn_readout
                ) for _ in range(self.num_bands * self.num_hyperplanes)
        ])

    def forward(self, data: QueryBatch) -> Tensor:
        """
        Forwards the query graph batch through the GCN submodels.

        Args:
            data (QueryBatch):
                Contains all information needed for message passing and readout.

        Returns:
            Tensor: Shape (batch_size, num_bands, num_hyperplanes, embed_dim)
                Collection of hyperplanes that demarcate the answer space.
        """
        return torch.cat([gcn(data) for gcn in self.submodels], dim=1).reshape(
            data.batch_size, self.num_bands, self.num_hyperplanes, self.embed_dim)
