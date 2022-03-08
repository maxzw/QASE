from data.data_utils import load_queries_by_formula
from model import MetaModel
from loader import *
from embedder import GraphEmbedder
    
data_dir = "./data/AIFB/processed/"
embed_dim = 128

train_queries = load_queries_by_formula(data_dir + "/train_edges.pkl")
# for i in range(2, 4):
#     train_queries.update(load_queries_by_formula(data_dir + "/train_queries_{:d}.pkl".format(i)))

chainqueries = train_queries['1-chain']
first_formula = list(chainqueries.keys())[0]
formqueries = chainqueries[first_formula]

embedder = GraphEmbedder(
    data_dir=data_dir,
    embed_dim=embed_dim
    )

model = MetaModel(
    embed_dim=embed_dim
)

x, y_ids, y, y_neg = vectorize_batch(first_formula, formqueries, embedder, device=model.device)

out = model(x)
print(out.size())
print(out)