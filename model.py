"""Metamodel implementation."""

import pickle
import random
import re
from typing import Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from data.graph import _reverse_relation
from gnn.gcn import GCNModel
from loader import QueryBatch, QueryTargetInfo, VectorizedQueryBatch


class MetaModel(torch.nn.Module):
    def __init__(
        self,
        data_dir,
        embed_dim,
        num_bands=4,
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
        self.num_bands = num_bands
        self.num_hyperplanes = num_hyperplanes
        self.gcn_layers = gcn_layers
        self.gcn_readout = gcn_readout
        self.gcn_use_bias = gcn_use_bias
        self.gcn_opn = gcn_opn
        self.gcn_dropout = gcn_dropout
        self.device = device

        # instantiate GCN models
        self.submodels = nn.ModuleList([
            GCNModel(
                self.embed_dim,
                self.gcn_layers,
                self.gcn_readout,
                self.gcn_use_bias,
                self.gcn_opn,
                self.gcn_dropout,
                self.device
                ) for _ in range(self.num_bands * self.num_hyperplanes)
        ])

        # build embeddings
        self._build_embeddings()


    def _build_embeddings(self):
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

        Embeddings for relations are stored as embedding directly:
        self.rel_features = nn.Embedding(num_rels, embed_dim)

        We define self.rel_maps as a mapping from tuple (fr, r, to) to rel_id.
        """
        
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
            self.ent_features[mode] = torch.nn.Embedding(node_mode_counts[mode], self.embed_dim)
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


    def embed_ents(self, mode: str, nodes: Tensor) -> Tensor:
        return self.ent_features[mode](self.node_maps[nodes.long()])


    def embed_vars(self, mode: str) -> Tensor:
        return self.var_features[mode](torch.tensor([0]))


    def embed_rels(self, rel_types: Tensor) -> Tensor:
        return self.rel_features(rel_types)


    def vectorize_batch(self, batch: QueryBatch) -> VectorizedQueryBatch:

        # Embed entities and variables
        ent_embed = torch.empty((batch.ent_ids.size(0), self.embed_dim))
        for i, (id, mode) in enumerate(zip(batch.ent_ids, batch.ent_modes)):
            if id == -1: # means variable
                emb = self.embed_vars(mode)
            else:
                emb = self.embed_ents(mode, id)
            ent_embed[i] = emb

        # Embed relations
        num_unique_edges = len(set(batch.edge_ids))
        reg_rel_embed = torch.empty((num_unique_edges, self.embed_dim))
        inv_rel_embed = torch.empty((num_unique_edges, self.embed_dim))
        edge_type = torch.empty((len(batch.edge_ids),), dtype=torch.int64)
        all_ids = []
        for i, id in enumerate(batch.edge_ids):
            if id not in all_ids:
                # add to regular edges (stored as inverse initially)
                reg_rel_embed[len(all_ids)] = self.embed_rels(
                    torch.tensor(self.rel_maps[_reverse_relation(id)]))
                # add to inverse edges
                inv_rel_embed[len(all_ids)] = self.embed_rels(
                    torch.tensor(self.rel_maps[id]))
                # keep track of known edges
                edge_type[i] = len(all_ids)
                all_ids.append(id)
            else: # if we've already seen the edge, just refer to index
                old_idx = all_ids.index(id)
                edge_type[i] = old_idx
        
        # Combine regular + inverse edges
        rel_embed = torch.cat((reg_rel_embed, inv_rel_embed), dim=0)

        assert ent_embed.size(0) == batch.target_idx.size(0)
        assert rel_embed.size(0)//2 == max(edge_type)+1
        assert num_unique_edges == len(all_ids)
        
        return VectorizedQueryBatch(
            batch_size  = batch.batch_size.to(self.device),
            batch_idx   = batch.batch_idx.to(self.device),
            target_idx  = batch.target_idx.to(self.device),
            ent_embed   = ent_embed.to(self.device),
            rel_embed   = rel_embed.to(self.device),
            edge_index  = batch.edge_index.to(self.device),
            edge_type   = edge_type.to(self.device),
            q_diameters = batch.q_diameters.to(self.device)
        )


    def forward(self, x_batch: QueryBatch) -> Tensor:
        """
        First embeds and then forwards the query graph batch through the GCN submodels.

        Args:
            data (QueryBatch):
                Contains all information needed for message passing and readout.

        Returns:
            Tensor: Shape (batch_size, num_bands, num_hyperplanes, embed_dim)
                Collection of hyperplanes that demarcate the answer space.
        """
        data: VectorizedQueryBatch = self.vectorize_batch(x_batch)

        return torch.cat([gcn(data) for gcn in self.submodels], dim=1).reshape(
            data.batch_size, self.num_bands, self.num_hyperplanes, self.embed_dim)


    def embed_targets(self, y_batch: QueryTargetInfo) -> Tuple[Tensor, Tensor]:

        pos_embs = torch.empty((len(y_batch.pos_ids), self.embed_dim))
        neg_embs = torch.empty((len(y_batch.pos_ids), self.embed_dim))
        
        for i, (p_id, p_m, n_id) in enumerate(zip(y_batch.pos_ids, y_batch.pos_modes, y_batch.neg_ids)):
            pos_embs[i] = self.embed_ents(p_m, p_id)
            
            # no sample found, pick random embedding form target type
            if n_id == -1:
                neg_embs[i] = self.embed_ents(p_m, torch.tensor(random.choice(self.nodes_per_mode[p_m])))
            
            else:
                neg_embs[i] = self.embed_ents(p_m, n_id)
        
        return pos_embs, neg_embs


    def predict(self, hyp: Tensor) -> Sequence[Sequence[int]]:
        raise NotImplementedError
        