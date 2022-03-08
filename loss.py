"""Loss functions for hyperplane configurations."""
from abc import abstractmethod

from torch import Tensor, nn

class AnswerSpaceLoss(nn.Module):
    """A loss for answer space."""

    @abstractmethod
    def forward(
        self,
        hyps: Tensor,
        pos_embeds: Tensor,
        neg_embeds: Tensor
    ) -> Tensor:
        """
        Computes the loss for a collection of hyperplanes.

        Args:
            hyps (Tensor): Shape (batch_size, num_bands, num_hyperplanes, embed_dim)
                A collection of hyperplanes grouped per band.
            pos_embeds (Tensor): Shape (batch_size, embed_dim)
                The answer entity embeddings.
            neg_embeds (Tensor): Shape (batch_size, embed_dim)
                The negative sample entity embeddings.

        Returns:
            Tensor: The loss value
        """
        raise NotImplementedError

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

# def return_answers(self, hyp: Tensor) -> Sequence:
#     """
#     Returns answer set per batch item.
#     Used to compute metrics such as accuracy, precision and recall.
#     """
#     pass

# def signature_loss(self, hyp: Tensor, y: Tensor) -> Tensor:
#     """
#     Calculates the distance between the preferred signature [1,1,..,1] 
#     and the signature of the entity.
#     """
#     # reshape y for calculating dot product
#     y = y.reshape(y.size(0), 1, 1, -1).expand(y.size(0), self.num_bands, self.num_hyperplanes, -1)
#     # calculate dot product with hyperplanes
#     dot = torch.mul(hyp, y).sum(dim=-1)
#     # get approximate signature using sigmoid function
#     s = torch.sigmoid(dot)
#     # calculate bucket-wise distance with perfect score [1, 1, ..., 1]
#     sign_distance = (self.num_hyperplanes - torch.sum(s, dim=-1))/self.num_hyperplanes
#     return sign_distance

# def diversity_loss(self, hyp: Tensor) -> Tensor:
#     """
#     Calculates the diversity loss for a set of hyperplanes
#     """
#     return torch.Tensor(0)

# def calc_loss(self, x: QueryBatch, y: Tensor, y_neg: Tensor, return_answers: bool = False) -> Tuple[Tensor, List]:
#     hyp = self.forward(x)

#     d_true = self.signature_loss(hyp, y)
#     d_false = 1 - self.signature_loss(hyp, y_neg)

#     # only use loss for buckets that contain the answer
#     # if none contain the answer, we use all buckets.
#     d_true_ = d_true.clone().detach()
#     indicator = torch.tensor(d_true_ > .5).float()
#     indicator[(indicator == 0).all(dim=-1)] = 1
#     ind_sums = indicator.sum(dim=-1)
#     # we combine the bucket losses into an average loss instead of sum
#     loss_true = torch.mul(d_true, indicator).sum(dim=-1)/ind_sums
#     # and average over the batch size
#     loss_true = loss_true.mean()

#     # TODO: implement loss_false!
#     loss_false = 0

#     hyp_loss = self.diversity_loss(hyp)
#     loss = loss_true + loss_false + hyp_loss
#     answers = self.return_answers() if return_answers else [None]
#     return loss, answers