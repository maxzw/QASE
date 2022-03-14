"""Loss functions for hyperplane configurations"""
import torch
import torch.nn as nn
from torch import Tensor


class MirroredLeakyReLU(nn.Module):
    """Mirrored leaky ReLU implementation."""
    def __init__(self, pos_slope=1e-3):
        super(self.__class__, self).__init__()
        self.pos_slope = pos_slope
    
    def forward(self, y: Tensor) -> Tensor:
        return -torch.minimum(y, torch.tensor(0)) + self.pos_slope * torch.maximum(torch.tensor(0), y)


class AnswerSpaceLoss(nn.Module):
    """A loss for answer space."""
    def __init__(self, pos_slope=1e-3):
        super(self.__class__, self).__init__()
        self.pos_slope = pos_slope
        self.act = MirroredLeakyReLU(pos_slope)

    def _band_distance(self, hyp: Tensor, y: Tensor):
        # extend y for calculating dot product
        # from: (batch_size, embed_dim)
        # to:   (batch_size, num_bands, num_hyperplanes, embed_dim)
        y = y.reshape(y.size(0), 1, 1, y.size(1)).expand(-1, hyp.size(1), hyp.size(2), -1)
        # calculate dot product with hyperplanes
        # to:   (batch_size, num_bands, num_hyperplanes)
        dot = torch.mul(hyp, y).sum(dim=-1)
        # get approximate signature using mirrored leaky ReLU activation
        s = self.act(dot)
        # calculate band-wise distance with perfect score: [>1, >1, ..., >1]
        # to:   (batch_size, num_bands)
        sign_distance = torch.sum(s, dim=-1)
        return sign_distance

    def forward(
        self,
        hyp: Tensor,
        pos_embeds: Tensor,
        neg_embeds: Tensor
        ) -> Tensor:

        # calculate distance per band for true and false samples:
        # (batch_size, num_bands)
        d_true = self._band_distance(hyp, pos_embeds)
        d_false = self._band_distance(hyp, neg_embeds)
        
        d_true_selection = torch.argmin(d_true, dim=-1)

        # batch-wise loss
        a = torch.sum(d_true[d_true_selection], dim=-1)
        b = torch.mean(d_false, dim=-1)
        loss =  a + b
        return torch.mean(loss)
