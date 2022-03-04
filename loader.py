import dataclasses
from torch import Tensor

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

                     
@dataclasses.dataclass
class VectorizedQueryBatch:
    """Batch of vectorized query graphs containing embeddings."""

    # for readout
    batch_size: Tensor      # (1). Number of queries in graph
    num_nodes: Tensor       # (1). Number of nodes per query
    target_idx: Tensor      # (1). Index of target node per query
    batch_idx: Tensor       # (num_entities). Indicates which graph the entities belong to (for readout).

    # for message passing
    ent_e: Tensor           # (num_entities, embed_dim). The entity embeddings (contains duplicate embeddings!).
    edge_index: Tensor      # (2, in_edges + out_edges). List of batch-local index pairs representing edges in query graph.
    edge_type: Tensor       # (in_edges + out_edges). The batch-local relation ID for each edge, refers to rel_e.
    rel_e: Tensor           # (in_edges + out_edges, embed_dim). The relation embeddings.
                            # This includes the relation representation for inverse relations, but does not
                            # include the self-loop relation (which is learned independently for each layer).
                            # Because relations do not overlap like nodes do, it does not contain duplicate embeddings.
