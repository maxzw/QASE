"""Training module."""
import torch.nn.functional as F

def evaluate(model, dataloader):
    x, edge_index, edge_type, rel_embed, graph_ids = vectorize_query_data(dataloader)
    pred = model(x, edge_index, edge_type, rel_embed, graph_ids)
    pass


def train_model(
    model,
    optimizer,
    train_queries,
    val_queries,
    test_queries,
    iterations,
    eval_every_nbatch
    ):

    # prepare all queries in batches using get_query_dataloader()
    train_dataloader = get_query_dataloader(train_queries, num_batches=iterations)
    val_dataloader = get_query_dataloader(val_queries)
    test_dataloader = get_query_dataloader(test_queries)

    for batch_id, (query_data, pos_ans, neg_ans) in enumerate(train_dataloader):

        x, edge_index, edge_type, rel_embed, graph_ids = vectorize_query_data(query_data)

        pred = model(x, edge_index, edge_type, rel_embed, graph_ids)
        loss = model.calculate_loss(pred, pos_ans, neg_ans)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluate every 'eval_every_nbatch' batches on validation set
        if batch_id % eval_every_nbatch == 0:
            evaluate(model, val_dataloader)
    
    # evaluate on test set
    evaluate(model, test_dataloader)