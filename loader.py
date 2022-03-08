"""Variables and functions needed for generating query batches."""

from dataclasses import dataclass
import numpy as np
import random
from typing import Any, Sequence, Tuple, Mapping
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

from data.graph import Formula, Query, _reverse_relation
from embedder import GraphEmbedder

"""
We have the following data structure:
    - Graph:
        - Data split:
            - query structure:
                - formula:
                    - Query

Questions:
- how do we want the model to be trained?
- how do we vectorize the data efficiently?

"""

@dataclass
class QueryBatch:
    """Batch of vectorized query graphs containing embeddings."""

    # for readout
    batch_size: Tensor      # (1). Number of queries in batch
    num_nodes: Tensor       # (1). Number of nodes per query
    target_idx: Tensor      # (1). Index of target node per query
    batch_idx: Tensor       # (num_entities). Indicates which graph the entities belong to (for readout).

    # for message passing
    ent_embed: Tensor       # (num_nodes, embed_dim). The entity embeddings.
    rel_embed: Tensor       # (2 * num_edges, embed_dim). The relation embeddings.
    edge_index: Tensor      # (2, num_edges). List of batch-local index pairs representing edges in query graph.
                            # Does not contain inverse edges, which are inferred in message passing locally.
    edge_type: Tensor       # (num_edges,). The batch-local relation ID for each edge, referring to rel_embed.


@dataclass
class QueryTargetInfo:
    """
    Additional target information corresponding to QueryBatch object.
    Used for loss function, evaluation and reporting.
    """
    pos_embed: Tensor           # (batch_size, embed_dim). The positive sample entity embeddings.
    neg_embed: Tensor           # (batch_size, embed_dim). The negative sample entity embeddings.
    pos_ids: Tensor             # (batch_size,). The target entity IDs. Used for classification metrics.
    pos_modes: Sequence[str]    # (batch_size,). The target entity modes. Used for classification metrics.
    q_type: Sequence[str]       # (batch_size,). Query types: ['1-chain', '2-chain', ..., '3-inter_chain'].


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
    embedder: GraphEmbedder,
    device
    ) -> Tuple[QueryBatch, Tensor]:
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
    edge_embs = torch.empty((len(rel_idx), embedder.embed_dim))
    edge_type = torch.empty((len(rel_idx)), dtype=torch.int64)
    for i in rel_idx:
        global_rel_id = embedder.rel_maps[_reverse_relation(rels[i])]
        edge_embs[i] = embedder.embed_rels(torch.tensor([global_rel_id]))
        edge_type[i] = i
    edge_embs = edge_embs
    edge_type = edge_type.repeat(batch_size)
    
    # get entity embeddings
    ent_e = torch.empty(batch_size, num_nodes, embedder.embed_dim)
    anchor_ids = np.empty([batch_size, num_anchors]).astype(int)
    # First rows of x contain embeddings of all anchor nodes
    for i, anchor_mode in enumerate(formula.anchor_modes):
        anchors = [q.anchor_nodes[i] for q in queries]
        anchor_ids[:, i] = anchors
    for i, anchor_mode in enumerate(formula.anchor_modes):
        ent_e[:, i] = embedder.embed_ents(anchor_ids[:, i], anchor_mode)
    # all other rows contain variable embeddings
    all_nodes = formula.get_nodes()
    for i, var_id in enumerate(var_idx):
        var_type = all_nodes[var_id]
        ent_e[:, num_anchors+i] = embedder.embed_vars(var_type)
    # then we reshape to feature matrix shape
    ent_e = ent_e.reshape(-1, embedder.embed_dim)

    x = QueryBatch(
        batch_size  = batch_size.to(device),
        num_nodes   = edge_data.num_nodes.to(device),
        target_idx  = target_idx.to(device),
        batch_idx   = batch_idx.to(device),
        ent_embed   = ent_e.to(device),
        rel_embed   = edge_embs.to(device),
        edge_index  = batch.edge_index.to(device),
        edge_type   = edge_type.to(device),
    )

    # get target node embeddings
    target_mode = formula.target_mode
    y_ids = torch.tensor([q.target_node for q in queries]).to(device)
    y = embedder.embed_ents(y_ids, target_mode).to(device)

    # get negative sample embeddings
    if "inter" in formula.query_type: # sample hard negative IDs per query
        neg_nodes = [random.choice(query.hard_neg_samples) for query in queries]
    # 1-chain does not contain negative samples, we sample manually from target mode IDs
    elif formula.query_type == "1-chain": 
        neg_nodes = [random.choice(embedder.nodes_per_mode[formula.target_mode]) for _ in queries]
    else:
        neg_nodes = [random.choice(query.neg_samples) for query in queries]
    y_neg = embedder.embed_ents(neg_nodes, target_mode).to(device)

    return x, y_ids, y, y_neg


def get_datasets():
    """Returns dict of DataSets"""
    # TODO: find out if we want to train structure/formula-wise or completely random.
    raise NotImplementedError


def collate_query_data(
    batch: Sequence[Query],
    embedder: GraphEmbedder,
    ) -> Tuple[QueryBatch, QueryTargetInfo]:
    raise NotImplementedError
    

# hacky way to allow the collator to use the graph embedder
class Collator(object):
    def __init__(self, embedder: GraphEmbedder):
        self.embedder = embedder
    def __call__(self, batch: Sequence[Query]):
        collate_query_data(batch, self.embedder)


def get_dataloaders(
    # ... params to get datasets
    embedder,
    batch_size,
    num_workers
    ):
    """
    Returns a dictionary of dataloaders for train, validation
    and test splits.
    """
    datasets, information = get_datasets()

    # get collator
    collator = Collator(embedder)

    loaders = {
        key: DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=key == "train",
            collate_fn=collator,
            pin_memory=True,
            drop_last=key == "train",
            num_workers=num_workers,
        )
        for key, dataset in datasets.items()
    }
    return loaders, information