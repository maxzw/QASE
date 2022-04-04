"""Loss functions for hyperplane configurations"""
import random
from typing import Sequence
from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class AnswerSpaceLoss(nn.Module):
    """An LSH-inspired loss for query answer spaces."""
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x: Tensor
        ) -> Tensor:
        """Calculates the loss for an answer space.

        Args:
            x (Tensor): Shape (batch_size, num_bands, num_hyperplanes).
                Dot product between hyperplanes and entity embedding.

        Returns:
            Tensor: Shape (batch_size, num_bands).
                Band-wise distance value.
        """
        raise NotImplementedError


class QASEAnserSpaceLoss(AnswerSpaceLoss):
    def __init__(self, pos_w: float = 1.0, neg_w: float = 1.0, div_w: float = 1.0) -> None:
        super().__init__()
        self.pos_w = pos_w
        self.neg_w = neg_w
        self.div_w = div_w

    def _shuffled_indices(self, max_index: int) -> Sequence[int]:
        out = []
        for idx in range(max_index):
            out.append(random.choice([x for x in range(max_index) if not x == idx]))
        return out

    def forward(
        self,
        hyp: Tensor,
        pos_embeds: Tensor,
        neg_embeds: Tensor
        ) -> Tensor:

        # hyp shape: (batch_size, num_bands, num_hyp, embed_dim)
        
        pos_ = pos_embeds.reshape(pos_embeds.size(0), 1, 1, pos_embeds.size(1))     # shape: (batch, 1, 1, embed_dim)
        pos_cos_sim = torch.cosine_similarity(pos_, hyp, dim=-1)                    # shape: (batch, num_bands, num_hyp)

        neg_ = neg_embeds.reshape(neg_embeds.size(0), 1, 1, neg_embeds.size(1))     # shape: (batch, 1, 1, embed_dim)
        neg_cos_sim = torch.cosine_similarity(neg_, hyp, dim=-1)                    # shape: (batch, num_bands, num_hyp)

        neg_hyp = hyp[:, :, self._shuffled_indices(hyp.size(2)), :]                 # shape: (batch_size, num_bands, num_hyp, embed_dim)
        div_cos_sim = torch.cosine_similarity(hyp, neg_hyp, dim=-1)                 # shape: (batch, num_bands, num_hyp)

        # calculate band distance:
        band_pos_loss = -torch.mean(pos_cos_sim, dim=-1)    # shape (batch, num_bands)
        band_neg_loss =  torch.mean(neg_cos_sim, dim=-1)    # shape (batch, num_bands)
        band_div_loss =  torch.mean(div_cos_sim, dim=-1)    # shape (batch, num_bands)

        # aggregate band distances:
        pos_w = torch.softmax(-band_pos_loss, dim=-1)
        batch_pos_loss = torch.sum(band_pos_loss * pos_w, dim=-1)   # shape (batch)
        batch_neg_loss = torch.mean(band_neg_loss, dim=-1)          # shape (batch)
        batch_div_loss = torch.sum(band_div_loss* pos_w, dim=-1)    # shape (batch)

        # calculate margin loss
        batch_loss = batch_pos_loss * self.pos_w + batch_neg_loss * self.neg_w + batch_div_loss * self.div_w

        loss = torch.mean(batch_loss)
        p = torch.mean(batch_pos_loss.detach(), dim=-1).item()
        n = torch.mean(batch_neg_loss.detach(), dim=-1).item()
        d = torch.mean(batch_div_loss.detach(), dim=-1).item()
        return loss, p, n, d
