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
        self.build_embeddings()
        
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
        
        # create and initialize entity embeddings # TODO: add to parameters
        self.ent_features = {
            mode : torch.nn.Embedding(
                node_mode_counts[mode] + 1, 
                self.embed_dim
                ).weight.data.normal_(0, 1./self.embed_dim) for mode in rels
            }
        print("\nCreated entity embeddings:")
        for t in self.ent_features:
            print(f"{t}:({len(self.ent_features[t])}, {self.embed_dim})")
        
        # create mapping from global id to type-specific id
        new_node_maps = torch.ones(num_nodes + 1, dtype=torch.long).fill_(-1)
        for mode, id_list in node_maps.items():
            for i, n in enumerate(id_list):
                assert new_node_maps[n] == -1
                new_node_maps[n] = i
        self.node_maps = new_node_maps

        # create entity lookup function (global_id: int, ent_type: str)
        def ent_lookup(nodes, mode):
            return self.ent_features[mode][self.node_maps[nodes]]
        self.embed_ents = ent_lookup

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
        self.rel_features = torch.nn.Embedding(len(rel_maps), self.embed_dim)   # TODO: Why is this not initialized?
                                                                                # if init -> need [] instead of () in lookup
        print("\nCreated relation embeddings:")
        print(self.rel_features)

        # create relation lookup function (rel_type: tuple(str, str, str))
        def rel_lookup(rel_types):
            rel_ids = torch.tensor([self.rel_maps[t] for t in rel_types])
            return self.rel_features(rel_ids)
        self.embed_rels = rel_lookup

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


# test function
if __name__ == "__main__":
    
    data_dir = "./data/AIFB/processed/"

    model = GCNModel(
        data_dir = data_dir,
        embed_dim = 128,
        num_layers = 2,
        readout = 'sum'
    )

    # TODO make sure graph features are in module!
    print('')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    
    print('')
    print(model.embed_ents([1, 2, 3], 'person')) # NOTE required only one class, implement formula-specific batches!
    rels = [
        ('publication', 'http://wwww3org/1999/02/22-rdf-syntax-ns#type', 'class'),
        ('publication', 'http://swrcontowareorg/ontology#publishes', 'organization'),
        ('project', 'http://swrcontowareorg/ontology#hasProject', 'publication')
    ]
    print('')
    print(model.embed_rels(rels))