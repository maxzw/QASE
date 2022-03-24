"""Metamodel implementation."""

import logging
from abc import abstractmethod
import pickle
import random
import time
from typing import Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from data.graph import _reverse_relation
from gnn.gcn import GCNModel
from loader import QueryBatchInfo, QueryTargetInfo, VectorizedQueryBatch

logger = logging.getLogger(__name__)


class AnswerSpaceModel(nn.Module):
    """Abstract class for embedding queries into sets of hyperplanes."""
    def __init__(self, data_dir, embed_dim, device):
        super().__init__()
        self.data_dir = data_dir
        self.embed_dim = embed_dim
        self.device = device

    def _build_embeddings(self):
        """
        Builds embeddings for entities, variables and relations.

        Embeddings for entities, variables and relations are stored in nn.Embedding:
        self.ent_features = nn.Embedding(num_ents,  embed_dim)
        self.var_features = nn.Embedding(num_types, embed_dim),
        self.rel_features = nn.Embedding(num_rels, embed_dim)

        We define self.var_maps as a mapping from entity mode (type) to var_features index.
        We define self.rel_maps as a mapping from tuple (fr, r, to) to rel_features index.
        """
        
        # load data and statistics
        rels, _, node_maps = pickle.load(open(self.data_dir+"/graph_data.pkl", "rb"))
        self.nodes_per_mode = node_maps
        node_mode_counts = {mode: len(node_maps[mode]) for mode in node_maps}
        self.num_nodes = sum(node_mode_counts.values())

        # create entity and variable embeddings
        self.ent_features = nn.Embedding(self.num_nodes, self.embed_dim)
        self.ent_features.weight.data.normal_(0, 1./self.embed_dim)
        logger.info(f"Created entity embeddings: {self.ent_features}")
        self.var_features = nn.Embedding(len(node_maps), self.embed_dim)
        self.var_features.weight.data.normal_(0, 1./self.embed_dim)
        logger.info(f"Created variable embeddings: {self.var_features}")

        # create mapping from variable mode to variable ID
        self.var_maps = {m:i for i, m in enumerate(node_maps)}

        # create mapping from rel str to rel ID
        rel_maps = {}
        num_rels = 0
        for fr in list(rels.keys()):
            for to_r in rels[fr]:
                to, r = to_r
                rel_id = (fr, r, to)
                if rel_id not in rel_maps:
                    rel_maps[rel_id] = num_rels
                    num_rels += 1
        self.rel_maps = rel_maps

        # create relation embeddings
        self.rel_features = nn.Embedding(num_rels, self.embed_dim)
        self.rel_features.weight.data.normal_(0, 1./self.embed_dim)
        logger.info(f"Created relation embeddings: {self.rel_features}")

    
    def embed_ents(self, nodes: Tensor) -> Tensor:
        return self.ent_features(nodes)


    def embed_vars(self, mode: str) -> Tensor:
        return self.var_features(torch.tensor(self.var_maps[mode]))


    def embed_rels(self, rel_type: Tuple[str, str, str]) -> Tensor:
        return self.rel_features(torch.tensor(self.rel_maps[rel_type]))


    def vectorize_batch(self, batch: QueryBatchInfo) -> VectorizedQueryBatch:
        """Vectorizes batch by converting QueryBatchInfo to VectorizedQueryBatch object.

        Args:
            batch (QueryBatchInfo): Non-vectorized batch information.

        Returns:
            VectorizedQueryBatch: Vectorized batch information.
        """

        # Embed entities and variables
        ent_embed = torch.empty((batch.ent_ids.size(0), self.embed_dim))
        for i, (id, mode) in enumerate(zip(batch.ent_ids, batch.ent_modes)):
            if id == -1: # means variable
                emb = self.embed_vars(mode)
            else:
                emb = self.embed_ents(id)
            ent_embed[i] = emb

        # Embed relations
        num_unique_edges = len(set(batch.edge_ids))
        reg_rel_embed = torch.empty((num_unique_edges, self.embed_dim))
        inv_rel_embed = torch.empty((num_unique_edges, self.embed_dim))
        edge_type = torch.empty((len(batch.edge_ids),), dtype=torch.int64)
        # we keep track of edges already embedded to preserve memory
        all_ids = []
        for i, id in enumerate(batch.edge_ids):
            if id not in all_ids:
                # add to regular edges (stored as inverse initially)
                reg_rel_embed[len(all_ids)] = self.embed_rels(_reverse_relation(id))
                # add to inverse edges
                inv_rel_embed[len(all_ids)] = self.embed_rels(id)
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


    def embed_targets(
        self,
        data: QueryTargetInfo
        ) -> Tuple[Tensor, Tensor]:
        """Returns embeddings for both the positive and negative targets of a query.

        Args:
            data (QueryTargetInfo): Contains all info about the query targets.
                See loader.py for more information.

        Returns:
            Tuple[Tensor, Tensor]: Returns output tensors of shape (batch_size, embed_dim),
                one for both the positive and negative entities.
        """

        pos_embs = torch.empty((len(data.pos_ids), self.embed_dim))
        neg_embs = torch.empty((len(data.pos_ids), self.embed_dim))
        
        for i, (p_id, p_m, n_id, t_nodes) in enumerate(zip(
            data.pos_ids,
            data.pos_modes,
            data.neg_ids,
            data.target_nodes
            )):
            pos_embs[i] = self.embed_ents(p_id)
            
            # no sample found (-1), pick random embedding form target type 
            # which is NOT an answer to the query
            if n_id == -1:
                neg_sample = random.choice([e for e in self.nodes_per_mode[p_m] if e not in t_nodes])
                neg_embs[i] = self.embed_ents(torch.tensor(neg_sample))
            
            else:
                neg_embs[i] = self.embed_ents(n_id)
        
        return pos_embs.to(self.device), neg_embs.to(self.device)


    @abstractmethod
    def predict(
        self,
        hyp: Tensor,
        modes: Sequence[str]
        ) -> Sequence[Sequence[int]]:
        """
        Given a configuration of hyperplanes predict the entity IDs that are 
        part of the answer. These entities are filtered for target mode.

        Args:
            hyp (Tensor): Shape (batch_size, num_bands, band_size, embed_dim).
                The hyperplanes outputted by the model.
            modes (Sequence[str]): Shape (batch_size,).
                The modes belonging to the target nodes, used for filtering.

        Returns:
            Sequence[Sequence[int]]: Nested list of predicted answer entities per batch.
        """
        raise NotImplementedError


    @abstractmethod
    def forward(self, batch: QueryBatchInfo) -> Tensor:
        """
        First embeds and then forwards the query graph batch through the GCN submodels.

        Args:
            batch (QueryBatch):
                Contains all information needed for message passing and readout.

        Returns:
            Tensor: Shape (batch_size, num_bands, num_hyperplanes, embed_dim)
                Collection of hyperplanes that demarcate the answer space.
        """
        data: VectorizedQueryBatch = self.vectorize_batch(batch)
        raise NotImplementedError


class HypewiseGCN(AnswerSpaceModel):
    """Uses one GCN for every hyperplane"""
    def __init__(
        self,
        data_dir: str,
        embed_dim: int,
        device,
        num_bands: int,
        num_hyperplanes: int,
        gcn_layers: int,
        gcn_stop_at_diameter: bool,
        gcn_pool: str,
        gcn_comp: str,
        gcn_use_bias: bool,
        gcn_use_bn: bool,
        gcn_dropout: float,
        gcn_share_weights: bool
        ):
        
        # initiate superclass and build embeddings
        super().__init__(data_dir, embed_dim, device)
        self._build_embeddings()

        # meta info
        self.num_bands = num_bands
        self.num_hyperplanes = num_hyperplanes

        # gcn info
        self.gcn_layers = gcn_layers
        self.gcn_stop_at_diameter = gcn_stop_at_diameter
        self.gcn_pool = gcn_pool
        self.gcn_comp = gcn_comp
        self.gcn_use_bias = gcn_use_bias
        self.gcn_use_bn = gcn_use_bn
        self.gcn_dropout = gcn_dropout
        self.gcn_share_weights = gcn_share_weights

        # instantiate GCN models
        self.submodels = nn.ModuleList([
            GCNModel(
                embed_dim           = self.embed_dim,
                num_layers          = self.gcn_layers,
                stop_at_diameter    = self.gcn_stop_at_diameter,
                pool                = gcn_pool,
                comp                = self.gcn_comp,
                use_bias            = self.gcn_use_bias,
                use_bn              = self.gcn_use_bn,
                dropout             = self.gcn_dropout,
                share_weights       = self.gcn_share_weights,
                device              = self.device
                ) for _ in range(self.num_bands * self.num_hyperplanes)
        ])


    def predict(
        self,
        hyp: Tensor,
        modes: Sequence[str]
        ) -> Sequence[Sequence[int]]:

        answers = [[] for _ in range(len(modes))]
        answers1 = [[] for _ in range(len(modes))]

        with torch.no_grad():

            for mode in set(modes):

                # batch indices that have this target mode
                batch_mode_idx = torch.from_numpy(np.where(np.array(modes) == mode)[0])
                print(batch_mode_idx)

                # select queries with this mode:
                # (batch_size_mode, num_bands, band_size, embed_dim)
                mode_hyp = torch.index_select(hyp, 0, batch_mode_idx) 
                print(mode_hyp.size()) # [28, 6, 4, 128]

                # select entity embeddings with this mode:
                # (num_ents_mode, embed_dim)
                mode_emb = self.ent_features(torch.tensor(self.nodes_per_mode[mode])).to(self.device)
                print(mode_emb.size()) # [146, 128]

                # add extra dimension to hyperplanes for broadcasting:  
                # from  (batch_size_mode, num_bands, band_size, embed_dim)
                # to    (batch_size_mode, 1, num_bands, band_size, embed_dim)       
                mode_hyp_1 = mode_hyp.reshape(mode_hyp.size(0), 1, mode_hyp.size(1), mode_hyp.size(2), mode_hyp.size(3))          
                print(mode_hyp_1.size()) # [28, 1, 6, 4, 128]

                # add extra dimension for broadcasting:  
                # from  (num_ents_mode, embed_dim)
                # to    (1, num_ents_mode, 1, 1, embed_dim)
                mode_emb_1 = mode_emb.reshape(1, mode_emb.size(0), 1, 1, mode_emb.size(1))
                print(mode_emb_1.size()) # [1, 146, 1, 1, 128]

                # calculate dot products of hyperplanes-embeddings
                # and convert positive/negative dot products to binary values
                # shape: (batch_size_mode, num_ents_mode, num_bands, band_size)
                dot_inds = (torch.sum(mode_hyp_1 * mode_emb_1, dim=-1) > 0)
                print(dot_inds.size()) # [28, 146, 6, 4]

                # only if all hyperplanes in a band are positive and the band contains the answer
                # shape: (batch_size_mode, num_ents_mode, num_bands)
                band_inds = torch.all(dot_inds, dim=-1)
                print(band_inds.size()) # [28, 146, 6]

                # if any band contains the entity, then the entity is a predicted answer
                # shape: (batch_size_mode, num_ents_mode)
                ent_inds = torch.any(band_inds, dim=-1)
                print(ent_inds.size()) # [28, 146]

                # iterate through all batches
                for batch_idx, batch in zip(batch_mode_idx, ent_inds):
                    # for all entities for that batch
                    for ent_idx, ent_ind in enumerate(batch):
                        # if the entity indicator is True (is in any band's answer space)
                        if ent_ind:
                            # we add the global(!) entity index (=ID) to the answer list for that batch
                            answers[batch_idx].append(self.nodes_per_mode[mode][ent_idx])
                
        assert len(answers) == len(modes) == hyp.size(0)

        return answers


    def forward(self, x_batch: QueryBatchInfo) -> Tensor:
        # output should be in shape (batch_size, num_bands, num_hyperplanes, embed_dim)

        data: VectorizedQueryBatch = self.vectorize_batch(x_batch)
        
        return torch.cat([gcn(data) for gcn in self.submodels], dim=-1).reshape(
            data.batch_size, self.num_bands, self.num_hyperplanes, self.embed_dim)

        
class BandwiseGCN(AnswerSpaceModel):
    """Uses one GCN for every band."""
    def __init__(
        self,
        data_dir: str,
        embed_dim: int,
        device,
        ):
        
        # initiate superclass and build embeddings
        super().__init__(data_dir, embed_dim, device)
        self._build_embeddings()


    def predict(
        self,
        hyp: Tensor,
        modes: Sequence[str]
        ) -> Sequence[Sequence[int]]:
        raise NotImplementedError


    def forward(self, x_batch: QueryBatchInfo) -> Tensor:
        # output should be in shape (batch_size, num_bands, num_hyperplanes, embed_dim)

        data: VectorizedQueryBatch = self.vectorize_batch(x_batch)

        # TODO: Implement
        raise NotImplementedError
