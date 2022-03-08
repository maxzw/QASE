import torch
from torch import Tensor
from torch.utils.data import DataLoader

from evaluation import evaluate
from loss import AnswerSpaceLoss
from model import MetaModel

def _train_epoch(
    model: MetaModel,
    dataloader: DataLoader,
    loss_fn: AnswerSpaceLoss,
    optimizer: torch.optim.Optimizer
    ) -> Tensor:
    """Train the model for one epoch."""
    
    # put the model in train mode
    model.train()
    
    epoch_loss = torch.zeros(size=tuple(), device=model.device)
    for batch, targets in dataloader:
        optimizer.zero_grad()
        hyps = model(batch)
        loss = loss_fn(hyps, targets.pos_embed, targets.neg_embed)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach() * batch.batch_size
    return epoch_loss / len(dataloader)

def train(
    model,
    data_loaders,
    loss_fn,
    optimizer,
    num_epochs,
    eval_freq,
    ):
    
    train_data_loader = data_loaders["train"]
    valid_data_loader = data_loaders["valid"]

    # training
    for epoch in range(num_epochs):
        epoch_loss = _train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer
            )

        # validation
        if (epoch + 1) % eval_freq == 0:
            val_results = evaluate(
                model,
                valid_data_loader,
                loss_fn
            )

    training_report = epoch_loss + val_results
    return training_report