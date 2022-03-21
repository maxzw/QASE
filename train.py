"""Training module"""
import logging
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from models import AnswerSpaceModel
from loader import QueryBatchInfo, QueryTargetInfo
from loss import AnswerSpaceLoss
from evaluation import ClassificationReport, evaluate

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
    
    # keep track of total loss in this epoch
    batch_losses = []

    # (temporary)
    pos_distances = []
    neg_distances = []
    
    x_info: QueryBatchInfo
    y_info: QueryTargetInfo
    for x_info, y_info in tqdm(dataloader, desc="Epoch", unit="batch", position=1, leave=False):
        
        optimizer.zero_grad()
        hyp = model(x_info)
        pos_emb, neg_emb = model.embed_targets(y_info)
        loss, pos_d, neg_d = loss_fn(hyp, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()
        batch_losses += [loss.item()]
        pos_distances += [pos_d.item()]
        neg_distances += [neg_d.item()]

    logger.info(f"Mean pos: {torch.mean(torch.tensor(pos_distances))} | Mean neg: {torch.mean(torch.tensor(neg_distances))}")
    return torch.mean(torch.tensor(batch_losses))


def train(
    model: AnswerSpaceModel,
    train_dataloader: DataLoader,
    loss_fn: AnswerSpaceLoss,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    val_dataloader: DataLoader,
    val_freq: int,
    ):

    # keep track of total loss during training
    epoch_losses = []

    # keep track of classification statistics during training
    val_report = ClassificationReport()
    
    # train
    for epoch in tqdm(range(num_epochs), desc="Training", unit="Epoch", position=0):
        epoch_loss = _train_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimizer
            )
        logger.info(f"Mean epoch loss: {epoch_loss}")
        epoch_losses.append(epoch_loss)

        # evaluate
        if (epoch + 1) % val_freq == 0:
            val_results = evaluate(
                model,
                val_dataloader
            )
            logger.info(f"Validation results: {val_results}")
            val_report.include(val_results, {'epoch': epoch})

    return epoch_losses, val_report.src