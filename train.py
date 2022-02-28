"""Training module."""
import torch.nn.functional as F


def evaluate(model, dataloader):
    pass


def train_model(
    model,
    optimizer,
    train_queries,
    val_queries,
    test_queries,
    epochs,
    eval_every_nbatch
    ):

    # prepare all queries in batches
    train_batches = None
    val_batches = None
    test_batches = None

    for epoch in range(epochs):

        for batch_id, data in enumerate(train_batches):

            pred = model(data.x, data.edge_index)
            loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # evalaute every 'eval_every_nbatch' batches on validation set
            if batch_id % eval_every_nbatch == 0:
                evaluate(model, val_batches)
    
    # evaluate on test set
    evaluate(model, test_batches)