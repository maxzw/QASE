"""Dataclasses used for training."""

import dataclasses
from torch import Tensor

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