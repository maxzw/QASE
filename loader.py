"""Variables and functions needed for generating training batches."""

import numpy as np
import random
from typing import Sequence, Tuple
import torch
from torch import Tensor
from torch_geometric.data import Data, Batch

from graph import Formula, Query, _reverse_relation
from dataclass import VectorizedQueryBatch
from metamodel import MetaModel

query_edge_indices = {'1-chain': [[0],
                                    [1]],
                        '2-chain': [[0, 2],
                                    [2, 1]],
                        '3-chain': [[0, 3, 2],
                                    [3, 2, 1]],
                        '2-inter': [[0, 1],
                                    [2, 2]],
                        '3-inter': [[0, 1, 2],
                                    [3, 3, 3]],
                        '3-inter_chain': [[0, 1, 3],
                                        [2, 3, 2]],
                        '3-chain_inter': [[0, 1, 3],
                                        [3, 3, 2]]}

query_diameters = {'1-chain': 1,
                    '2-chain': 2,
                    '3-chain': 3,
                    '2-inter': 1,
                    '3-inter': 1,
                    '3-inter_chain': 2,
                    '3-chain_inter': 2}

query_edge_label_idx = {'1-chain': [0],
                        '2-chain': [1, 0],
                        '3-chain': [2, 1, 0],
                        '2-inter': [0, 1],
                        '3-inter': [0, 1, 2],
                        '3-inter_chain': [0, 2, 1],
                        '3-chain_inter': [1, 2, 0]}

variable_node_idx = {'1-chain': [0],
                        '2-chain': [0, 2],
                        '3-chain': [0, 2, 4],
                        '2-inter': [0],
                        '3-inter': [0],
                        '3-chain_inter': [0, 2],
                        '3-inter_chain': [0, 3]}


def vectorize_batch(
    formula: Formula,
    queries: Sequence[Query],
    model: MetaModel
    ) -> Tuple[VectorizedQueryBatch, Tensor]:
    """
    Converts batch data with global IDs to embeddings.
    """
    # general info
    batch_size = torch.tensor(len(queries))
    var_idx = variable_node_idx[formula.query_type]
    num_anchors = len(formula.anchor_modes)
    num_nodes = torch.tensor(num_anchors + len(var_idx))
    batch_idx = torch.arange(0, batch_size).repeat_interleave(num_nodes)
    target_idx = torch.full((batch_size,), num_nodes-1)

    # get edge index
    edge_index = torch.tensor(query_edge_indices[formula.query_type], dtype=torch.long)
    edge_data = Data(edge_index=edge_index)
    edge_data.num_nodes = num_nodes
    batch = Batch.from_data_list([edge_data for i in range(batch_size)])

    # get relation types
    rels = formula.get_rels()
    rel_idx = query_edge_label_idx[formula.query_type]
    edge_embs = torch.empty((len(rel_idx), model.embed_dim))
    edge_type = torch.empty((len(rel_idx)), dtype=torch.int64)
    for i in rel_idx:
        global_rel_id = model.rel_maps[_reverse_relation(rels[i])]
        edge_embs[i] = model.embed_rels(torch.tensor([global_rel_id]))
        edge_type[i] = i
    edge_embs = edge_embs
    edge_type = edge_type.repeat(batch_size)
    
    # get entity embeddings
    ent_e = torch.empty(batch_size, num_nodes, model.embed_dim)
    anchor_ids = np.empty([batch_size, num_anchors]).astype(int)
    # First rows of x contain embeddings of all anchor nodes
    for i, anchor_mode in enumerate(formula.anchor_modes):
        anchors = [q.anchor_nodes[i] for q in queries]
        anchor_ids[:, i] = anchors
    for i, anchor_mode in enumerate(formula.anchor_modes):
        ent_e[:, i] = model.embed_ents(anchor_ids[:, i], anchor_mode)
    # all other rows contain variable embeddings
    all_nodes = formula.get_nodes()
    for i, var_id in enumerate(var_idx):
        var_type = all_nodes[var_id]
        ent_e[:, num_anchors+i] = model.embed_vars(var_type)
    # then we reshape to feature matrix shape
    ent_e = ent_e.reshape(-1, model.embed_dim)

    x = VectorizedQueryBatch(
        batch_size  = batch_size.to(model.device),
        num_nodes   = edge_data.num_nodes.to(model.device),
        target_idx  = target_idx.to(model.device),
        batch_idx   = batch_idx.to(model.device),
        ent_e       = ent_e.to(model.device),
        edge_index  = batch.edge_index.to(model.device),
        edge_type   = edge_type.to(model.device),
        rel_e       = edge_embs.to(model.device)
    )

    # get target node embeddings
    target_mode = formula.target_mode
    y_ids = torch.tensor([q.target_node for q in queries]).to(model.device)
    y = model.embed_ents(y_ids, target_mode).to(model.device)

    # get negative sample embeddings
    if "inter" in formula.query_type: # sample hard negative IDs per query
        neg_nodes = [random.choice(query.hard_neg_samples) for query in queries]
    # 1-chain does not contain negative samples, we sample manually from target mode IDs
    elif formula.query_type == "1-chain": 
        neg_nodes = [random.choice(model.nodes_per_mode[formula.target_mode]) for _ in queries]
    else:
        neg_nodes = [random.choice(query.neg_samples) for query in queries]
    y_neg = model.embed_ents(neg_nodes, target_mode).to(model.device)

    return x, y_ids, y, y_neg
