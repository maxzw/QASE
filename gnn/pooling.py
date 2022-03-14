"""Graph pooling operations."""
from abc import abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn
from torch_scatter import scatter_add, scatter_max


class GraphPooling(nn.Module):
    """A module for graph pooling."""

    @abstractmethod
    def forward(
        self,
        embs: Tensor,
        batch_idx: Optional[Tensor], 
        target_idx: Optional[Tensor],
        ) -> Tensor:
        """Obtain graph representations by aggregating node representations.
        
        Args:
            embs (_type_): _description_
            batch_idx (Optional[Tensor]): _description_
            target_idx (Optional[Tensor]): _description_

        Returns:
            Tensor: _description_
        """
        raise NotImplementedError


class SumGraphPooling(GraphPooling):
    """Aggregation by taking sum."""

    def forward(
        self,
        embs,
        batch_idx,
        **kwargs
        ) -> Tensor:
        return scatter_add(embs, index=batch_idx, dim=0)


class MaxGraphPooling(GraphPooling):
    """Aggregation by taking maximum."""

    def forward(
        self,
        embs,
        batch_idx,
        **kwargs
        ) -> Tensor:
        out, argmax = scatter_max(embs, batch_idx, dim=0)
        return out


class TargetPooling(GraphPooling):
    """Aggregation by taking target node."""

    def forward(
        self,
        embs: Tensor,
        target_idx: Tensor,
        **kwargs
        ) -> Tensor:
        return embs[target_idx.bool()]
