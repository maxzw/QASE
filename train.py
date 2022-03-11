import torch
from torch import Tensor
from torch.utils.data import DataLoader

from evaluation import evaluate
from loader import QueryBatch, QueryTargetInfo
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
    
    x: QueryBatch
    y: QueryTargetInfo
    for x, y in dataloader:
        
        optimizer.zero_grad()
        hyp = model(x)
        loss = loss_fn(hyp, y.pos_embed, y.neg_embed)
        print(loss)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x.batch_size.item()

    return epoch_loss


def train(
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    optimizer,
    num_epochs,
    eval_freq,
    ):
    
    # training
    for epoch in range(num_epochs):
        losses = _train_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimizer
            )
        print(losses)
        
        # validation
        # if (epoch + 1) % eval_freq == 0:
        #     val_results = evaluate(
        #         model,
        #         val_dataloader,
        #         loss_fn
        #     )

    return None