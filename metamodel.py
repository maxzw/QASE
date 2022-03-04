"""Metamodel implementation."""

import numpy as np
import pickle
import torch
import torch.nn as nn
from torch import Tensor, var
from typing import Sequence
from torch_geometric.data import Data, Batch

import loader
from gcn import GCNModel
from graph import Formula, Query, _reverse_relation
from loader import VectorizedQueryBatch


class MetaModel(torch.nn.Module):
    def __init__(
        self,
        data_dir,
        embed_dim=128,
        num_hyperplanes=4,
        gcn_layers=2,
        gcn_readout='sum',
        gcn_use_bias=True,
        gcn_opn='corr',
        gcn_dropout=0,
        device=None
        ):
        super().__init__()
        self.data_dir = data_dir
        self.embed_dim = embed_dim
        self.num_hyperplanes = num_hyperplanes
        self.gcn_layers = gcn_layers
        self.gcn_readout = gcn_readout
        self.gcn_use_bias = gcn_use_bias
        self.gcn_opn = gcn_opn
        self.gcn_dropout = gcn_dropout
        self.device = device

        # create embeddings and lookup functions
        self._build_embeddings()

        # initiate GCNModels
        self.submodels = nn.ModuleList([
            GCNModel(
                self.embed_dim,
                self.gcn_layers,
                self.gcn_readout
                ) for _ in range(self.num_hyperplanes)
        ])

    def _build_embeddings(self):
        """
        Builds embeddings for both entities (including variables) and relations.

        Embeddings for entities are stored in a dict of embeddings:
        self.ent_features = {
            'type1': nn.Embedding(num_ents, embed_dim),
            'type2': nn.Embedding(num_ents, embed_dim),
            ...
        }

        Embeddings for relations are stored directly as embedding:
        self.rel_features = nn.Embedding(num_rels, embed_dim)
        """
        
        # load data and statistics
        rels, _, node_maps = pickle.load(open(self.data_dir+"/graph_data.pkl", "rb"))
        node_mode_counts = {mode: len(node_maps[mode]) for mode in node_maps}
        num_nodes = sum(node_mode_counts.values())
        
        # create and initialize entity embeddings. For each type: (num_nodes + 1, embed_dim)
        self.ent_features = nn.ModuleDict()
        self.var_features = nn.ModuleDict()
        for mode in rels:
            self.ent_features[mode] = torch.nn.Embedding(node_mode_counts[mode] + 1, self.embed_dim)
            self.ent_features[mode].weight.data.normal_(0, 1./self.embed_dim)
            self.var_features[mode] = torch.nn.Embedding(1, self.embed_dim)
            self.var_features[mode].weight.data.normal_(0, 1./self.embed_dim)
        print("\nCreated entity embeddings:")
        for m, e in self.ent_features.items():
            print(f"    {m}: {e}")
        print("\nCreated variable embeddings:")
        for m, e in self.var_features.items():
            print(f"    {m}: {e}")
        
        # create mapping from global id to type-specific id
        new_node_maps = torch.ones(num_nodes, dtype=torch.long).fill_(-1)
        for mode, id_list in node_maps.items():
            for i, n in enumerate(id_list):
                assert new_node_maps[n] == -1
                new_node_maps[n] = i
        self.node_maps = new_node_maps

        # create entity lookup function
        def _ent_lookup(nodes: Tensor, mode: str) -> Tensor:
            return self.ent_features[mode](self.node_maps[nodes])
        self.embed_ents = _ent_lookup

        # create variable lookup function
        def _var_lookup(mode: str) -> Tensor:
            return self.var_features[mode](torch.tensor([0]))
        self.embed_vars = _var_lookup

        # create mapping from rel str to rel ID
        rel_maps = {}
        rel_counter = 0
        for fr in list(rels.keys()):
            for to_r in rels[fr]:
                to, r = to_r
                rel_id = (fr, r, to)
                if rel_id not in rel_maps:
                    rel_maps[rel_id] = rel_counter
                    rel_counter += 1
        self.rel_maps = rel_maps

        # create relation and initialize relation embeddings: (num_types, embed_dim)
        self.rel_features = torch.nn.Embedding(len(rel_maps), self.embed_dim)
        self.rel_features.weight.data.normal_(0, 1./self.embed_dim)
        print("\nCreated relation embeddings:")
        print(f"    {self.rel_features}")

        # create relation lookup function
        def _rel_lookup(rel_types: Tensor) -> Tensor:
            return self.rel_features(rel_types)
        self.embed_rels = _rel_lookup
    
    def vectorize_batch(self, formula: Formula, queries: Sequence[Query]) -> VectorizedQueryBatch:
        """Converts batch data with global IDs to embeddings."""
        # general info
        batch_size = len(queries)
        var_idx = loader.variable_node_idx[formula.query_type]
        num_anchors = len(formula.anchor_modes)
        num_nodes = num_anchors + len(var_idx)
        batch_idx = torch.arange(0, batch_size).repeat_interleave(num_nodes)
        target_idx = torch.full((batch_size,), num_nodes-1)

        # get edge index
        edge_index = torch.tensor(loader.query_edge_indices[formula.query_type], dtype=torch.long)
        edge_data = Data(edge_index=edge_index)
        edge_data.num_nodes = num_nodes
        batch = Batch.from_data_list([edge_data for i in range(batch_size)])

        # get relation types
        rels = formula.get_rels()
        rel_idx = loader.query_edge_label_idx[formula.query_type]
        edge_embs = torch.empty((len(rel_idx), self.embed_dim))
        edge_type = torch.empty((len(rel_idx)), dtype=torch.int64)
        for i in rel_idx:
            global_rel_id = self.rel_maps[_reverse_relation(rels[i])]
            edge_embs[i] = self.embed_rels(torch.tensor([global_rel_id]))
            edge_type[i] = i
        edge_type = edge_type.repeat(batch_size)
        
        # get entity embeddings
        anchor_ids = np.empty([batch_size, num_anchors]).astype(int)
        # First rows of x contain embeddings of all anchor nodes
        for i, anchor_mode in enumerate(formula.anchor_modes):
            anchors = [q.anchor_nodes[i] for q in queries]
            anchor_ids[:, i] = anchors
        ent_e = torch.empty(batch_size, num_nodes, self.embed_dim)
        for i, anchor_mode in enumerate(formula.anchor_modes):
            ent_e[:, i] = self.embed_ents(anchor_ids[:, i], anchor_mode)
        # all other rows contain variable embeddings
        all_nodes = formula.get_nodes()
        for i, var_id in enumerate(var_idx):
            var_type = all_nodes[var_id]
            ent_e[:, num_anchors+i] = self.embed_vars(var_type)
        # then we reshape to feature matrix shape
        ent_e = ent_e.reshape(-1, self.embed_dim)

        # TODO: add all to device...
        return VectorizedQueryBatch(
            batch_size=batch_size,
            num_nodes=edge_data.num_nodes,
            target_idx=target_idx,
            batch_idx=batch_idx,
            ent_e=ent_e,
            edge_index=batch.edge_index,
            edge_type=edge_type,
            rel_e=edge_embs
        )

    def forward(self, formula: Formula, queries: Sequence[Query]) -> Tensor:
        data: VectorizedQueryBatch = self.vectorize_batch(formula, queries)
        return torch.cat([gcn(data) for gcn in self.submodels], dim=1)


from data_utils import load_queries_by_formula

if __name__ == "__main__":
    
    data_dir = "./data/AIFB/processed/"
    embed_dim = 128

    train_queries = load_queries_by_formula(data_dir + "/train_edges.pkl")
    # for i in range(2, 4):
    #     train_queries.update(load_queries_by_formula(data_dir + "/train_queries_{:d}.pkl".format(i)))

    chainqueries = train_queries['1-chain']
    first_formula = list(chainqueries.keys())[0]
    formqueries = chainqueries[first_formula]

    model = MetaModel(
        data_dir=data_dir,
        embed_dim=embed_dim,
        num_hyperplanes=16,
        gcn_readout='sum'
    )

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    out = model(formula=first_formula, queries=formqueries)
    print(out)
    print(out.size())