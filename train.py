"""Training module"""
import numpy as np
import wandb
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
    pos_distances = []
    neg_distances = []
    
    x_info: QueryBatchInfo
    y_info: QueryTargetInfo
    for x_info, y_info in tqdm(dataloader, desc="Epoch", unit="batch", position=1, leave=False):
        
        optimizer.zero_grad()
        hyp = model(x_info)
        pos_emb, neg_emb = model.embed_targets(y_info)
        loss, p_dist, n_dist = loss_fn(hyp, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.detach().item()
        wandb.log({"train": {"batch": {"batch_loss": loss_val, "batch_p_dist": p_dist, "batch_n_dist": n_dist}}})

        batch_losses.append(loss_val)
        pos_distances.append(p_dist)
        neg_distances.append(n_dist)

    mean_loss   = np.mean(batch_losses)
    mean_dist_p = np.mean(pos_distances)
    mean_dist_n = np.mean(neg_distances)
    
    logger.info(f"Mean epoch loss: {mean_loss:.5f} ({mean_dist_p:.5f} - {mean_dist_n:.5f})")
    wandb.log({"train": {"epoch": {"mean_loss": mean_loss, "mean_loss_p": mean_dist_p, "mean_loss_n": mean_dist_n}}})
    return mean_loss


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
        epoch_losses.append(epoch_loss)

        # evaluate
        if (epoch + 1) % val_freq == 0:
            val_results = evaluate(
                model,
                val_dataloader
                )
            val_report.include(val_results, {'epoch': epoch})
            logger.info(f"Validation results: {val_results}")
            wandb.log({"val": {**val_results}})

    return epoch_losses, val_report.src