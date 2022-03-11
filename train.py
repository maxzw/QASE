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
    for x_info, y_info in dataloader:
        
        optimizer.zero_grad()
        hyp = model(x_info)
        pos_emb, neg_emb = model.embed_targets(y_info)
        loss = loss_fn(hyp, pos_emb, neg_emb)
        print(loss)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x_info.batch_size.item()

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