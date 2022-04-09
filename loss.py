"""Loss functions for hyperplane configurations"""
import random
from typing import Tuple
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


class QASEAnswerSpaceLoss(AnswerSpaceLoss):
    def __init__(self, pos_w: float = 1.0, neg_w: float = 1.0, div_w: float = 1.0, norm_w: float = 1.0) -> None:
        super().__init__()
        self.pos_w = pos_w
        self.neg_w = neg_w
        self.div_w = div_w
        self.norm_w = norm_w

    def shuffled_indices(self, max_index):
        out = []
        for idx in range(max_index):
            out.append(random.choice([x for x in range(max_index) if not x == idx]))
        return out

    def forward(
        self,
        hyp: Tensor,
        pos_embeds: Tensor,
        neg_embeds: Tensor
        ) -> Tuple[Tensor, float, float, float]:

        # hyp shape: (batch_size, num_bands, num_hyp, embed_dim)
        
        # positive cosine distance
        hyp = torch.nn.functional.normalize(hyp, dim=-1)

        # ----- positive cosine distance -----
        pos_ = pos_embeds.reshape(pos_embeds.size(0), 1, 1, pos_embeds.size(1))     # shape: (batch, 1, 1, embed_dim)
        pos_cos_sim = torch.cosine_similarity(pos_, hyp, dim=-1)                    # shape: (batch, num_bands, num_hyp)

        # ----- negative cosine distance -----
        neg_ = neg_embeds.reshape(neg_embeds.size(0), 1, 1, neg_embeds.size(1))     # shape: (batch, 1, 1, embed_dim)
        neg_cos_sim = torch.cosine_similarity(neg_, hyp, dim=-1)                    # shape: (batch, num_bands, num_hyp)

        # ----- hyperplane diversity -----

        # get projection of normal vectors onto average normal vector
        # avg_nv = torch.mean(hyp, dim=-2).reshape(hyp.size(0), hyp.size(1), 1, hyp.size(3))  # shape: (batch, num_bands, 1, embed_dim)
        # numer = torch.sum(avg_nv * hyp, dim=-1)
        # denom = torch.sum(hyp * hyp, dim=-1)
        # hyp_proj = hyp - (numer / denom).reshape(hyp.size(0), hyp.size(1), hyp.size(2), 1) * hyp

        # angle diversity
        # get random normal vector to be reference vector for every band
        # ref_norm_idx = torch.randint(0, hyp.size(2)-1, (hyp.size(0), hyp.size(1), 1,))\
        #     .repeat(1, 1, hyp.size(3)).view(hyp.size(0), hyp.size(1), 1, hyp.size(3))
        # ref_norms = torch.gather(hyp, 1, ref_norm_idx)
        # calculate angle between reference vector and all other normal vectors (0, x, y, z, ...)
        # sort angles
        # define target angles as range with len(hyperplanes) and steps 2pi/len(hyperplanes)
        # calculate difference in angle and take mean

        # div_cos_sim = torch.cosine_similarity(hyp_proj, hyp_proj[:, :, self.shuffled_indices(hyp.size(2)), :], dim=-1) # shape: (batch, num_bands, num_hyp)

        # angle from average normal vector (with norm as proxy)
        # we want the norm to be 1, so we minimize on -norm
        # hyp_proj_norm = -torch.norm(hyp_proj, dim=-1) # shape: (batch, num_bands, num_hyp)

        # calculate band distance:
        band_pos_loss   = -torch.mean(pos_cos_sim, dim=-1)    # shape (batch, num_bands)
        band_neg_loss   =  torch.mean(neg_cos_sim, dim=-1)    # shape (batch, num_bands)
        # band_div_loss   =  torch.mean(div_cos_sim, dim=-1)    # shape (batch, num_bands)
        # band_norm_loss  = -torch.mean(hyp_proj_norm, dim=-1)  # shape (batch, num_bands)

        # aggregate band distances:
        focus           = torch.softmax(-band_pos_loss, dim=-1)
        batch_pos_loss  = torch.sum(band_pos_loss * focus, dim=-1)       # shape (batch)
        batch_neg_loss  = torch.mean(band_neg_loss, dim=-1)              # shape (batch)
        # batch_div_loss  = torch.sum(band_div_loss * focus, dim=-1)       # shape (batch)
        # batch_norm_loss = torch.mean(band_norm_loss * focus, dim=-1)    # shape (batch)

        # calculate margin loss
        batch_loss = batch_pos_loss * self.pos_w + batch_neg_loss * self.neg_w# + batch_div_loss * self.div_w + batch_norm_loss * self.norm_w

        loss = torch.mean(batch_loss, dim=-1) # shape (1)
        p = torch.mean(batch_pos_loss.detach(), dim=-1).item()
        n = torch.mean(batch_neg_loss.detach(), dim=-1).item()
        # d = torch.mean(batch_div_loss.detach(), dim=-1).item()
        # n = torch.mean(batch_norm_loss.detach(), dim=-1).item()
        n = 0
        return loss, p, n, d, n

    def __repr__(self):
        return f"QASEAnswerSpaceLoss(pos_w={self.pos_w}, neg_w={self.neg_w}, div_w={self.div_w}, norm_w={self.norm_w})"
