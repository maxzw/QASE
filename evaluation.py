import torch

def get_predictions():
    pass

def classification_metrics():
    pass

def evaluate(
    model,
    dataloader,
    embedder
    ):

    # put model in evalation mode
    model.eval()
    
    report = {}
    for batch, targets in dataloader:

        # get hyperplanes
        hyps = model(batch)

        # get predicted entity IDs with hyperplanes, embeddings and target modes
        preds = get_predictions(hyps, embedder, targets.modes)

        # get classification metrics for each query type
        results = classification_metrics(preds, targets.y_id, targets.q_type)

        # update report with new results
        report.update(results)
    
    # group batch results together
    results = results
    
    return results