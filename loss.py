"""Loss functions for hyperplane configurations"""
from abc import abstractmethod
import torch
import torch.nn as nn
from torch import Tensor


# ---------- Classes for calculating distance ----------

class BandDistance(nn.Module):
    """
    Distance function between answer space and entity embedding.
    """
    
    @abstractmethod
    def forward(
        self,
        x: Tensor
        ) -> Tensor:
        """_summary_

        Args:
            x (Tensor): Shape (batch_size, num_bands, num_hyperplanes).
                Dot product between hyperplanes and entity embedding.

        Returns:
            Tensor: Shape (batch_size, num_bands).
                Band-wise distance value.
        """
        raise NotImplementedError


class SigmoidDistance(BandDistance):

    def forward(self, x: Tensor, pos: bool = True) -> Tensor:
        
        # apply sigmoid to map values to range [0, 1]
        a = torch.sigmoid(x)

        # For true samples we want all dot products to be positive, for simplicity
        # we want them to be much higher mapping them to 1. The band-wise distance 
        # can therefore be calculated as the difference between perfect score 
        # [1, 1, ..., 1] and the activated dot product
        if pos:
            return 1 - torch.mean(a, dim=-1)
        else:
            return 1 - torch.min(a, dim=-1)


# ---------- Classes for calculating hyperplane diversity ----------

# TODO: build function to measure diversity...


# ---------- Main loss class ----------


class AnswerSpaceLoss(nn.Module):
    """A loss for answer space."""
    def __init__(self, aggr: str = 'softmin'):
        super().__init__()
        assert aggr in ['min', 'mean', 'softmin']
        self.aggr = aggr

        # Define softmin if required
        self.distance = SigmoidDistance()
        if aggr == 'softmin':
            self.sm = nn.Softmin(dim=-1)


    def _calc_dot(self, hyp: Tensor, y: Tensor):
        """Calculated dot product between hyperplanes and entity."""
        
        # add extra dimensions to y for broadcasting:  
        # from  (batch_size, embed_dim)
        # to    (batch_size, 1, 1, embed_dim)
        y = y.reshape(y.size(0), 1, 1, y.size(1))
        
        # calculate dot product with hyperplanes
        # to    (batch_size, num_bands, num_hyperplanes)
        dot = torch.mul(hyp, y).sum(dim=-1)
        
        return dot


    def forward(
        self,
        hyp: Tensor,
        pos_embeds: Tensor,
        neg_embeds: Tensor
        ) -> Tensor:

        # calculate distance per band for true and false samples:
        # (batch_size, num_bands)
        d_true = self.distance(self._calc_dot(hyp, pos_embeds))
        d_false = self.distance(self._calc_dot(hyp, neg_embeds), pos=False)
        
        # aggregate the band-wise losses for positive samples. We only need one band to 
        # contain the answer, so we use a trade-off between exploration and exploitation.
        if self.aggr == 'min':
            p = torch.min(d_true, dim=-1).values
        elif self.aggr == 'mean':
            p = torch.mean(d_true, dim=-1)
        elif self.aggr == 'softmin':
            p = torch.sum(d_true * self.sm(d_true), dim=-1)
        
        # aggregate the band-wise losses for negative samples. we want all bands to always 
        # move away from non-target entities, we even prefer them to contain no answers at 
        # all in many cases. So we aggregate them uniformly.
        n = torch.mean(d_false, dim=-1)

        # we want to minimize the distance for positive samples and maximize it for 
        # negative samples:
        loss =  p - n

        # return mean for batch
        return torch.mean(loss), torch.mean(p.detach()).item(), torch.mean(n.detach()).item()

    def __repr__(self):
	    return '{}(Dist={}, aggr={})'.format(self.__class__.__name__, self.distance, self.aggr)