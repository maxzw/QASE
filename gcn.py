"""GCN model implementation using composition-based convolution."""

import pickle
import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_add, scatter_max

from CompGCN.compgcn_conv import CompGCNConv
from CompGCN.message_passing import MessagePassing
from loader import QueryBatch, VectorizedQueryBatch


class GCNModel(nn.Module):
    def __init__(self, data_dir, embed_dim, num_layers, readout, device=None):
        super().__init__()
        self.data_dir = data_dir
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.device = device
        assert readout in ['max', 'sum', 'TM']
        self.readout_str = readout

        # create embeddings and lookup functions
        # variable embeddings are in their respective mode list with index [-1]
        self.build_embeddings(data_dir, embed_dim)
        
        # create message passing layers
        layers = []
        for _ in range(num_layers - 1):
            layers += [
                CompGCNConv(
                    self.embed_dim,
                    self.embed_dim,
                    ),
                nn.ReLU()
            ]
        layers += [
            CompGCNConv(
                self.embed_dim,
                self.embed_dim,
                )
        ]
        self.layers = nn.ModuleList(layers)

        # define readout function
        if readout == 'max':
            self.readout = self.max_readout
        elif readout == 'sum':
            self.readout = self.sum_readout
        elif readout == 'TM':
            self.readout = self.target_message_readout

    def build_embeddings(self):
        """Builds embeddings for both entities (including variables) and relations."""
        
        # load data and statistics
        rels, adj_lists, node_maps = pickle.load(open(self.data_dir+"/graph_data.pkl", "rb"))
        node_mode_counts = {mode: len(node_maps[mode]) for mode in node_maps}
        num_nodes = sum(node_mode_counts.values())
        
        # create and initialize entity embeddings
        self.ent_features = nn.ParameterDict({
            mode : torch.nn.Embedding(
                node_mode_counts[mode] + 1, 
                self.embed_dim).weight.data.normal_(0, 1./self.embed_dim) \
                 for mode in rels
            })
        
        # create mapping from global id to type-specific id
        new_node_maps = torch.ones(num_nodes + 1, dtype=torch.long).fill_(-1)
        for mode, id_list in node_maps.items():
            for i, n in enumerate(id_list):
                assert new_node_maps[n] == -1
                new_node_maps[n] = i
        self.node_maps = new_node_maps

        # create lookup function
        self.embed_ents = lambda nodes, mode: self.ent_features[mode](self.node_maps[nodes])

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
                self.rel_features = None
                self.embed_rels = None
        self.rel_maps = rel_maps

        # create relation embeddings
        self.rel_features = torch.nn.Embedding(len(rel_maps), self.embed_dim)

    def max_readout(self, embs, batch_idx, **kwargs):
        out, argmax = scatter_max(embs, batch_idx, dim=0)
        return out

    def sum_readout(self, embs, batch_idx, **kwargs):
        return scatter_add(embs, batch_idx, dim=0)

    def target_message_readout(self, embs, batch_size, num_nodes, num_anchors, **kwargs):
        device = embs.device

        non_target_idx = torch.ones(num_nodes, dtype=torch.bool)
        non_target_idx[num_anchors] = 0
        non_target_idx.to(device)

        embs = embs.reshape(batch_size, num_nodes, -1)
        targets = embs[:, ~non_target_idx].reshape(batch_size, -1)

        return targets

    def vectorize_batch(self, batch: QueryBatch) -> VectorizedQueryBatch:
        """Converts batch data with global IDs to embeddings."""
        
        batch_idx   = torch.Tensor(batch.batch_idx).to(self.device)
        edge_index  = torch.Tensor(batch.edge_index).to(self.device)
        edge_type   = torch.Tensor(batch.edge_type).to(self.device)
        # extract entity and relation embeddings using lookup functions
        ent_e       = torch.Tensor(self.embed_ents(batch.entity_id, batch.entity_type)).to(self.device)
        rel_e       = torch.Tensor(self.embed_rels(batch.edge_type)).to(self.device)
        
        return VectorizedQueryBatch(
            batch_idx=batch_idx,
            ent_e=ent_e,
            edge_index=edge_index,
            edge_type=edge_type,
            rel_e=rel_e
        )

    def forward(self, batch: QueryBatch) -> Tensor:

        # get and unpack vectorized data
        data: VectorizedQueryBatch = self.vectorize_batch(batch)
        ent_e, rel_e = data.ent_e, data.rel_e

        # perform message passing
        for layer in self.layers:
            if isinstance(layer, MessagePassing):
                ent_e, rel_e = layer(ent_e, data.edge_index, data.edge_type, rel_e)
            else:
                ent_e = layer(ent_e)

        # aggregate node embeddings
        out = self.readout(
            embs        =   ent_e,
            batch_idx   =   data.batch_idx,
            batch_size  =   batch.batch_size,  
            num_nodes   =   batch.num_entities,
            num_anchors =   data.num_anchors # don't know...
            )
        return out

