import numpy as np
import dataclasses
from torch import Tensor
from typing import Sequence

from graph import Query


@dataclasses.dataclass
class VectorizedQueryBatch:
    """Batch of vectorized query graphs containing embeddings."""

    batch_idx: Tensor       # (num_entities). Indicates which graph the entities belong to (for readout).
    ent_e: Tensor           # (num_entities, embed_dim). The entity embeddings (contains duplicate embeddings!).
    edge_index: Tensor      # (2, num_edges). List of batch-local index pairs representing edges in query graph.
    edge_type: Tensor       # (in_edges + out_edges). The batch-local relation ID for each edge, refers to rel_e.
    rel_e: Tensor           # (in_edges + out_edges, embed_dim). The relation embeddings.
                            # This includes the relation representation for inverse relations, but does not
                            # include the self-loop relation (which is learned independently for each layer).
                            # Because relations do not overlap like nodes do, it does not contain duplicate embeddings.

    def __post_init__(self):
        assert self.ent_e is not None
        assert self.rel_e is not None
        assert self.edge_index is not None
        assert self.edge_type is not None


@dataclasses.dataclass
class QueryBatch:
    """
    A batch of query graphs with batch info and both global ids and batch-local pointers.
    Actually everything except the embeddings themselves.
    """

    # batch info
    batch_size: int                 # The number of queries in the batch
    num_entities: int               # The number of entities in the batch (contains duplicate referrals!)
    num_relations: int              # The number of relations in the batch (contains duplicate referrals!)
    
    # query-level info
    query_type: Sequence[str]       # (batch_size). Type of query ('1-chain', etc...)
    target_id: Sequence[int]        # (batch_size). Global ids of the target entities.
    target_type: Sequence[str]      # (batch_size). Types of the target entities.
    neg_id: Sequence[int]           # (batch_size). Global ids of the negative samples.
    hard_neg_id: Sequence[int]      # (batch_size). Global ids of the hard negative samples.

    # subquery-level info (structure)
    entity_id: Sequence[int]                # (num_entities). Global ids of the entities in the batch (all types + variables)
    entity_type: Sequence[str]              # (num_entities). Entity type.
    batch_idx: Sequence[int]                # (num_entities). Indicate the batch(=query)-id of the entity.
    edge_index: Sequence[Sequence[int]]     # (2, num_edges). List of batch-local index pairs representing edges in query graph.
    edge_type: Sequence[str]                # (num_edges). The global ids for relations in the batch.     

    def __post_init__(self):
        assert self.batch_size is not None
        assert self.num_entities is not None
        assert self.num_relations is not None
        assert self.target_id is not None
        assert self.target_type is not None
        assert self.neg_id is not None
        assert self.hard_neg_id is not None
        assert self.entity_id is not None
        assert self.entity_type is not None
        assert self.batch_idx is not None
        assert self.edge_index is not None
        assert self.edge_type is not None


def collate_query_data(queries: Sequence[Query]) -> QueryBatch:
    """Given a list of graph.Query objects returns a QueryBatch dataset."""

    # NOTE: because the formula gets extracted query-wise, the input queries do not
    # have to be of the same query type / formula!
    # I might have to make this function more efficient later.

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
    
    # batch info
    batch_size = len(queries)
    num_entities = 0
    num_relations = 0

    query_type = []
    target_id = []
    target_type = []
    neg_id = []
    hard_neg_id = []

    entity_id = []
    entity_type = []
    batch_idx = [] # NOTE points to entities!
    edge_index = [[],[]]
    edge_type = []

    for q_id, query in enumerate(queries):
        formula = query.formula
        q_type = formula.query_type
        
        # query-level info
        query_type.append(q_type)
        target_id.append(query.target_node)
        target_type.append(formula.target_mode)
        neg_id.append(query.neg_samples)
        hard_neg_id.append(query.hard_neg_samples)

        # subquery-level info (structure)
        
        # ---- entities and edge index:
        # entity_id
        # entity_type
        curr_edge_index = np.array(query_edge_indices[q_type])
        curr_num_entities = len(np.unique(curr_edge_index))
        edge_index = np.concatenate((edge_index, curr_edge_index + num_entities), axis=1)
        batch_idx.extend(np.full(curr_num_entities, q_id))
        num_entities += curr_num_entities
        # ---- edges:
        # num_relations
        # edge_type

    return QueryBatch(
        batch_size=batch_size,
        num_entities=num_entities,
        num_relations=num_relations,
        query_type=query_type,
        target_id=target_id,
        target_type=target_type,
        neg_id=neg_id,
        hard_neg_id=hard_neg_id,
        entity_id=entity_id,
        entity_type=entity_type,
        batch_idx=batch_idx,
        edge_index=edge_index,
        edge_type=edge_type
        )

if __name__ == "__main__":

    from data_utils import load_queries_by_formula

    embed_dim = 128
    data_dir = "./data/AIFB/processed/"

    train_queries = load_queries_by_formula(data_dir + "/train_edges.pkl")
    # for i in range(2, 4):
    #     train_queries.update(load_queries_by_formula(data_dir + "/train_queries_{:d}.pkl".format(i)))

    chainqueries = train_queries['1-chain']
    first_formula = list(chainqueries.keys())[0]
    formqueries = chainqueries[first_formula]
    smallbatch = formqueries[:5]

    b = collate_query_data(smallbatch)
    print(b)