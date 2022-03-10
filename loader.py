"""Variables and functions needed for generating query batches."""

from dataclasses import dataclass
import random
from typing import Sequence, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

from data.data_utils import load_queries_by_formula, load_test_queries_by_formula
from data.graph import Formula, _reverse_relation


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


@dataclass
class QueryBatch:
    """Class that holds query info of a batch."""
    batch_size: Tensor          # (1,). Size of the batch
    
    ent_ids: Tensor             # (num_nodes,). Type-specific IDs of the nodes.
    ent_modes: Sequence[str]    # (num_nodes,). Modes of the nodes.
    
    edge_index: Tensor          # (num_edges,). List of batch-local index pairs representing edges in query graph.
    edge_ids: Sequence[Tuple[str, str, str]]    # (num_edges,). IDs of the dges
    
    batch_idx: Tensor           # (num_nodes,). Indicates which graph the entities belong to (for readout).
    target_idx: Tensor          # (num_nodes,). Binary vector indicating target entities. Refers to ent_ids.
    q_diameters: Tensor         # (num_nodes,). Diameter of the query that the node is part of.


@dataclass
class QueryTargetInfo:
    """Class that contains target info of a batch."""
    pos_ids: Tensor             # (batch_size,). IDs of the target entities.
    pos_modes: Sequence[str]    # (batch_size,). Modes of the target entities.
    neg_ids: Tensor             # (batch_size,). IDs of the negative samples, share same mode as real targets.
    q_types: Sequence[str]      # (batch_size,). Query structure: ['1-chain', '2-chain', ..., '3-inter_chain'].


@dataclass
class VectorizedQueryBatch:
    """Batch of vectorized query graphs containing embeddings."""
    batch_size: Tensor      # (1,). Number of queries in the batch.
    
    # for readout
    batch_idx: Tensor       # (num_nodes,). Indicates which graph the entities belong to (for readout).
    target_idx: Tensor      # (num_nodes,). Binary vector indicating target entities. Refers to ent_embed.

    # for message passing
    ent_embed: Tensor       # (num_nodes, embed_dim). The entity embeddings.
    rel_embed: Tensor       # (2 * num_edges, embed_dim). The relation embeddings.
    q_diameters: Tensor     # (num_nodes,). The diameter of the query the node is part of. Used for masking.
    edge_index: Tensor      # (2, num_edges). List of batch-local index pairs representing edges in query graph.
                            # Does not contain inverse edges, which are inferred in message passing locally.
    edge_type: Tensor       # (num_edges,). The batch-local relation ID for each edge, referring to edge_index.



class CompGCNDataset(Dataset):
    def __init__(self, queries):
        self.queries = queries
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        return idx
    
    def collate_fn(self, indices):
        
        # load queries
        queries = [self.queries[i] for i in indices]
        
        # QueryBatch variables
        batch_size      = torch.tensor([len(queries)])
        ent_ids         = []
        ent_modes       = []
        edge_data_list  = [] # proxy for edge_index
        edge_ids        = []
        batch_idx       = torch.tensor([], dtype=torch.int64)
        target_idx      = torch.tensor([], dtype=torch.int64)
        q_diameters     = torch.tensor([], dtype=torch.int64)

        # QueryTargetInfo variables
        pos_ids     = []
        pos_modes   = []
        neg_ids     = []
        q_types     = []
        
        for q_i, query in enumerate(queries):
            form: Formula = query.formula            

            #  --- Collecting QueryBatch data ---
            var_idx = variable_node_idx[form.query_type]
            num_anchors = len(form.anchor_modes)
            num_nodes = torch.tensor(num_anchors + len(var_idx))
            batch_idx = torch.cat((batch_idx, torch.full((num_nodes,), q_i)), dim=-1)
            curr_diameter = torch.full((num_nodes,), query_diameters[form.query_type])
            q_diameters  = torch.cat((q_diameters, curr_diameter), dim=-1)

            # first rows of entities contain embeddings of all anchor nodes
            for n, m in zip(query.anchor_nodes, form.anchor_modes):
                ent_ids     += [n]
                ent_modes   += [m]
            # all other rows contain variable embeddings
            all_curr_nodes = form.get_nodes()
            for var_id in var_idx:
                ent_ids += [-1]
                ent_modes += [f"var_{all_curr_nodes[var_id]}"]
            
            # target_idx is always last node, since we read query from anchors -> targets
            curr_target_idx = torch.zeros(num_nodes)
            curr_target_idx[-1] = 1
            target_idx  = torch.cat((target_idx, curr_target_idx), dim=-1)

            # edge ids
            rels = form.get_rels()  
            rel_idx = query_edge_label_idx[form.query_type]
            for i in rel_idx:
                edge_ids += [_reverse_relation(rels[i])]

            # edge index
            curr_edge_data = Data(edge_index=torch.tensor([query_edge_indices[form.query_type]], dtype=torch.int64))
            curr_edge_data.num_nodes = num_nodes
            edge_data_list.append(curr_edge_data)

            # --- collecting QueryTargetInfo data ---
            pos_ids += [query.target_node]
            pos_modes += [form.target_mode]
            # get negative sample
            if "inter" in form.query_type: # sample hard negative IDs per query
                neg_id = random.choice(query.hard_neg_samples)
            # 1-chain does not contain negative samples, we sample manually from target mode IDs
            elif form.query_type == "1-chain": 
                neg_id = np.nan
            else:
                neg_id = random.choice(query.neg_samples)
            neg_ids += [neg_id]
            q_types += [form.query_type]

        # --- aggregation ---
        ent_ids = torch.tensor(ent_ids, dtype=torch.long)
        pos_ids = torch.tensor(pos_ids)
        neg_ids = torch.tensor(neg_ids)
        # we use the PyG Batch class to get incremented edge indices without overhead
        pyg_batch = Batch.from_data_list(edge_data_list)
        edge_index = pyg_batch.edge_index.reshape((2, -1))

        assert ent_ids.size(0) == len(ent_modes) == batch_idx.size(0) == target_idx.size(0) == q_diameters.size(0)
        assert edge_index.size(1) == len(edge_ids)
        assert batch_size[0] == pos_ids.size(0) == len(pos_modes) == neg_ids.size(0) == len(q_types)

        x = QueryBatch(
            batch_size=batch_size,
            ent_ids=ent_ids,
            ent_modes=ent_modes,
            edge_index=edge_index,
            edge_ids=edge_ids,
            batch_idx=batch_idx,
            target_idx=target_idx,
            q_diameters=q_diameters
        )

        y = QueryTargetInfo(
            pos_ids=pos_ids,
            pos_modes=pos_modes,
            neg_ids=neg_ids,
            q_types=q_types
        )
        
        return x, y


def get_queries(data_dir: str, split: str, exclude: Sequence[str] = []):
    assert split in ['train', 'val', 'test']

    queries = load_queries_by_formula(data_dir + f"/{split}_edges.pkl")
    for i in range(2, 4):
        if split == 'train':
            queries.update(load_queries_by_formula(data_dir + f"/{split}_queries_{i}.pkl"))
        else:
            i_queries = load_test_queries_by_formula(data_dir + f"/{split}_queries_{i}.pkl")
            queries["one_neg"].update(i_queries["one_neg"])
            queries["full_neg"].update(i_queries["full_neg"])

    out_queries = []
    info = {}
    for structure, d in queries.items():
        if structure not in exclude:
            info[structure] = 0
            for _form, queries in d.items():
                out_queries += queries
                info[structure] += len(queries)    

    return out_queries, info


def get_dataloader(
    data,
    batch_size,
    num_workers,
    shuffle,
    ):
    dataset = CompGCNDataset(data)
    return DataLoader(
        dataset     = dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        collate_fn  = dataset.collate_fn,
        pin_memory  = True,
        # num_workers = num_workers,
    )