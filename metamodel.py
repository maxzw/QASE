"""Metamodel implementation."""

import pickle
import torch
import torch.nn as nn
from torch import Tensor

from gcn import GCNModel
from dataclass import VectorizedQueryBatch


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

    def forward(self, data: VectorizedQueryBatch) -> Tensor:
        return torch.cat([gcn(data) for gcn in self.submodels], dim=1)
