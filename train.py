"""Training script."""

from data_utils import load_queries_by_formula
from metamodel import MetaModel
from loader import *
    
data_dir = "./data/AIFB/processed/"
embed_dim = 128
num_hyp = 16
num_buckets = 4
assert num_hyp % num_buckets == 0
bucket_size = num_hyp / num_buckets

train_queries = load_queries_by_formula(data_dir + "/train_edges.pkl")
# for i in range(2, 4):
#     train_queries.update(load_queries_by_formula(data_dir + "/train_queries_{:d}.pkl".format(i)))

chainqueries = train_queries['1-chain']
first_formula = list(chainqueries.keys())[0]
formqueries = chainqueries[first_formula]

model = MetaModel(
    data_dir=data_dir,
    embed_dim=embed_dim,
    num_hyperplanes=num_hyp,
    gcn_readout='sum'
)

x, y = vectorize_batch(first_formula, formqueries, model)

print(y)
print(y.size())

out = model(x)
print(out.size())

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
    For a band of 4 hyperplanes this is (4*3*2*1)/((2*1)(2*1)) = 24/4 = 6

The loss for one band can be calculated as follows:
    - band_loss = loss_true + loss_false + loss_diversity*C where C is a hyperparameter

Not every band needs backpropagation! Only backpropagate on bands that already contain the answer.
If multiple bands contain the answer, these bands get backpropagated. 
If no band contains the answer, all bands get backpropagated:
    - total_loss = band_loss_i + band_loss_i+1 + ... 
"""