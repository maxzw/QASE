import dataclasses
from typing import Sequence
from torch import Tensor

from graph import Query


@dataclasses.dataclass
class VectorizedQueryBatch:
    """Batch of vectorized query graphs containing embeddings."""

    ent_e: Tensor           # (batch_size, num_entities, embed_dim). The entity embeddings.
    edge_index: Tensor      # (batch_size, 2, num_edges). List of index pairs representing the edges in the graph.
    edge_type: Tensor       # (batch_size, num_edges,). The edge type (=relation ID) for each edge.
    rel_e: Tensor           # (batch_size, 2 * num_relations, embed_dim). The relation embeddings.
                            # This includes the relation representation for inverse relations, but does not
                            # include the self-loop relation (which is learned independently for each layer).

    def __post_init__(self):
        assert self.ent_e is not None
        assert self.rel_e is not None
        assert self.edge_index is not None
        assert self.edge_type is not None


@dataclasses.dataclass
class QueryBatch:
    """
    A batch of query graphs with batch info and both global and batch-local IDs.
    Actually everything except the embeddings themselves.
    """

    # batch info
    batch_size: int
    num_entities: int
    num_edges: int
    num_relations: int
    
    # query info
    target_ids: Sequence[tuple]

    def __post_init__(self):
        assert self.target_ids is not None


def collate_query_data(queries: Sequence[Query]) -> QueryBatch:
    """Given a list of graph.Query objects returns a loader.QueryBatch object."""
    
    batch_size = len(queries)

    for query_id, query in enumerate(queries):
        
        formula = query.formula
        # do something...

    return QueryBatch(
        batch_size = batch_size
        )