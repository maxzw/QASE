"""Training module"""
import logging
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from models import AnswerSpaceModel
from loader import QueryBatchInfo, QueryTargetInfo
from loss import AnswerSpaceLoss
from evaluation import ClassificationData, evaluate

logger = logging.getLogger(__name__)


def _train_epoch(
    model: AnswerSpaceModel,
    dataloader: DataLoader,
    loss_fn: AnswerSpaceLoss,
    optimizer: torch.optim.Optimizer
    ) -> Tensor:
    """Train the model for one epoch."""
    
    # put the model in train mode
    model.train()
    
    epoch_loss = torch.zeros(size=tuple(), device=model.device)
    
    x_info: QueryBatchInfo
    y_info: QueryTargetInfo
    for x_info, y_info in tqdm(dataloader, desc="Epoch", unit="batch", position=1, leave=False):
        
        optimizer.zero_grad()
        hyp = model(x_info)
        pos_emb, neg_emb = model.embed_targets(y_info)
        loss = loss_fn(hyp, pos_emb, neg_emb)
        logger.info(f"Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x_info.batch_size.item()

    return epoch_loss


def train(
    model,
    train_dataloader,
    loss_fn,
    optimizer,
    num_epochs,
    val_dataloader=None,
    val_freq=None,
    ):

    epoch_losses = []
    classification_data = ClassificationData()
    
    # training
    for epoch in tqdm(range(num_epochs), desc="Training", unit="Epoch", position=0):
        epoch_loss = _train_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimizer
            )
        epoch_losses.append(epoch_loss)

        # evaluate every 'val_freq' epochs
        if (epoch + 1) % val_freq == 0:
            eval_report = evaluate(
                model,
                val_dataloader,
                loss_fn
            )
            classification_data.include(eval_report)

    return epoch_losses, classification_data