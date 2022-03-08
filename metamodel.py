"""Metamodel implementation."""

import pickle
from typing import Sequence, Tuple, List
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

        # create embeddings and lookup functions
        self._build_embeddings()

        # initiate GCNModels
        self.submodels = nn.ModuleList([
            GCNModel(
                self.embed_dim,
                self.gcn_layers,
                self.gcn_readout
                ) for _ in range(self.num_bands * self.num_hyperplanes)
        ]) # TODO: find out how to share weights!

    def _build_embeddings(self):
        """
        Builds embeddings for both entities (including variables) and relations.

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

        We define self.rel_maps as a mapping from tuple (fr, r, to) to rel_id where
        fr, r and to are strings.
        """
        
        # load data and statistics
        rels, _, node_maps = pickle.load(open(self.data_dir+"/graph_data.pkl", "rb"))
        self.nodes_per_mode = node_maps
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
        """
        Forwards the query graph batch through the GCN submodels.
        """
        return torch.cat([gcn(data) for gcn in self.submodels], dim=1).reshape(
            data.batch_size, self.num_bands, self.num_hyperplanes, -1)

    def return_answers(self, hyp: Tensor) -> Sequence:
        """
        Returns answer set per batch item.
        Used to compute metrics such as accuracy, precision and recall.
        """
        pass

    def signature_loss(self, hyp: Tensor, y: Tensor) -> Tensor:
        """
        Calculates the distance between the preferred signature [1,1,..,1] 
        and the signature of the entity.
        """
        # reshape y for calculating dot product
        y = y.reshape(y.size(0), 1, 1, -1).expand(y.size(0), self.num_bands, self.num_hyperplanes, -1)
        # calculate dot product with hyperplanes
        dot = torch.mul(hyp, y).sum(dim=-1)
        # get approximate signature using sigmoid function
        s = torch.sigmoid(dot)
        # calculate bucket-wise distance with perfect score [1, 1, ..., 1]
        sign_distance = (self.num_hyperplanes - torch.sum(s, dim=-1))/self.num_hyperplanes
        return sign_distance

    def diversity_loss(self, hyp: Tensor) -> Tensor:
        """
        Calculates the diversity loss for a set of hyperplanes
        """
        return torch.Tensor(0)

    def calc_loss(self, x: VectorizedQueryBatch, y: Tensor, y_neg: Tensor, return_answers: bool = False) -> Tuple[Tensor, List]:
        hyp = self.forward(x)

        d_true = self.signature_loss(hyp, y)
        d_false = 1 - self.signature_loss(hyp, y_neg)

        # only use loss for buckets that contain the answer
        # if none contain the answer, we use all buckets.
        indicator = torch.tensor(d_true.clone().detach() > .5).float()
        indicator[(indicator == 0).all(dim=-1)] = 1
        ind_sums = indicator.sum(dim=-1)
        # we combine the bucket losses into an average loss instead of sum
        loss_true = torch.mul(d_true, indicator).sum(dim=-1)/ind_sums
        # and average over the batch size
        loss_true = loss_true.mean()

        # TODO: implement loss_false!
        loss_false = 0

        hyp_loss = self.diversity_loss(hyp)
        loss = loss_true + loss_false + hyp_loss
        answers = self.return_answers() if return_answers else [None]
        return loss, answers