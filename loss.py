"""Loss functions for hyperplane configurations."""
from abc import abstractmethod

import torch
from torch import Tensor, nn

class AnswerSpaceLoss(nn.Module):
    """A loss for answer space."""

    def _signature_loss(self, hyp: Tensor, y: Tensor):
        # reshape y for calculating dot product
        y = y.reshape(y.size(0), 1, 1, -1).expand(y.size(0), hyp.size(1), hyp.size(2), -1)
        # calculate dot product with hyperplanes
        dot = torch.mul(hyp, y).sum(dim=-1)
        # get approximate signature using sigmoid function
        s = torch.sigmoid(dot)
        # calculate band-wise distance with perfect score [1, 1, ..., 1]
        sign_distance = (hyp.size(1) - torch.sum(s, dim=-1))/hyp.size(1)
        return sign_distance

    def forward(
        self,
        hyp: Tensor,
        pos_embeds: Tensor,
        neg_embeds: Tensor
        ) -> Tensor:
        """
        Computes the loss for a collection of hyperplanes.

        Args:
            hyp (Tensor): Shape (batch_size, num_bands, num_hyperplanes, embed_dim)
                A collection of hyperplanes grouped per band.
            pos_embeds (Tensor): Shape (batch_size, embed_dim)
                The answer entity embeddings.
            neg_embeds (Tensor): Shape (batch_size, embed_dim)
                The negative sample entity embeddings.

        Returns:
            Tensor: The loss value
        """
        d_true = self._signature_loss(hyp, pos_embeds)
        d_false = self._signature_loss(hyp, neg_embeds)

        loss = d_true + (1- d_false)
        return torch.mean(loss)

"""
Loss ideas considering 1 entity, 1 band:

First calculate loss for true answer:
    1. For each hyperplane calculate dot product with answer entity embedding.
    2. Apply tanh/sigmoid activation function to map to [0, 1] and get signature. --> fix saturated neurons!
    3. Loss = distance between signature and [1, 1, 1, 1] -> is what we want.
        - We can use loss = l - sum(signature) where l = length of signature.

Second calculate loss for negative sample:
    1. For each hyperplane calculate dot product with negative sample embedding.
    2. Apply tanh/sigmoid activation function to map to [0, 1] and get signature.
    3. Loss = 1 - distance between signature and [1, 1, 1, 1] -> is what we DONT want.

Thirdly calculate loss for hyperplane diversity. Intuition is that vectors that point
in the same direction have a high dot product -> minimize this.
    1. Calculate sum of dot products of every combination of 2 hyperplanes in a band. 
    This is called “n choose r” and its equation is n!/r!(n-r)!
    For a band of 4 hyperplanes this is (4*3*2*1)/((2*1)(2*1)) = 24/4 = 6 combinations (reasonable!)

The loss for one band can be calculated as follows:
    - band_loss = loss_true + loss_false + loss_diversity*C where C is a hyperparameter

Not every band needs backpropagation! Only backpropagate on bands that already contain the answer.
If multiple bands contain the answer, these bands get backpropagated. 
If no band contains the answer, all bands get backpropagated:
    - total_loss = band_loss_i + band_loss_i+1 + ... 
"""