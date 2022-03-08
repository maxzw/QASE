"""A simple class that holds the graph embeddings."""

import pickle
import torch
from torch import nn, Tensor

class GraphEmbedder:
    def __init__(self, data_dir, embed_dim):
        """
        Builds embeddings for entities, variables and relations.

        Embeddings for entities and variables are stored in a dict of embeddings:
        self.ent_features = {
            'type_1': nn.Embedding(num_ents, embed_dim),
            ...
            'type_n': nn.Embedding(num_ents, embed_dim)
        }
        self.var_features = {
            'type_1': nn.Embedding(1, embed_dim),
            ...
            'type_n': nn.Embedding(1, embed_dim)
        }
        We define self.node_maps as a mapping from global entity ID to typed entity ID.

        Embeddings for relations are stored directly as embedding:
        self.rel_features = nn.Embedding(num_rels, embed_dim)

        We define self.rel_maps as a mapping from tuple (fr, r, to) to rel_id.
        """
        self.data_dir = data_dir
        self.embed_dim = embed_dim
        
        # load data and statistics
        rels, _, node_maps = pickle.load(open(self.data_dir+"/graph_data.pkl", "rb"))
        self.nodes_per_mode = node_maps
        node_mode_counts = {mode: len(node_maps[mode]) for mode in node_maps}
        num_nodes = sum(node_mode_counts.values())

        # create mapping from global id to type-specific id
        new_node_maps = torch.ones(num_nodes, dtype=torch.long).fill_(-1)
        for mode, id_list in node_maps.items():
            for i, n in enumerate(id_list):
                assert new_node_maps[n] == -1
                new_node_maps[n] = i
        self.node_maps = new_node_maps
        
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
        print(f"    {self.rel_features}\n")


    def embed_ents(self, nodes: Tensor, mode: str) -> Tensor:
        return self.ent_features[mode](self.node_maps[nodes])


    def embed_vars(self, mode: str) -> Tensor:
        return self.var_features[mode](torch.tensor([0]))


    def rel_str_to_id(self, rel: str) -> int:
        return self.rel_maps[rel]


    def embed_rels(self, rel_types: Tensor) -> Tensor:
        return self.rel_features(rel_types)
