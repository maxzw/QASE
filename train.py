"""Training module"""
import wandb
import optuna
import logging
from tqdm import tqdm

import numpy as np
from pandas import DataFrame
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from helper import EarlyStopping

from models import AnswerSpaceModel
from loader import QueryBatchInfo, QueryTargetInfo
from loss import AnswerSpaceLoss
from evaluation import ClassificationReport, evaluate

logger = logging.getLogger(__name__)


def _train_epoch(
    model: AnswerSpaceModel,
    dataloader: DataLoader,
    loss_fn: AnswerSpaceLoss,
    optimizer: torch.optim.Optimizer,
    epoch: int
    ) -> float:
    """Train the model for one epoch."""
    
    # put the model in train mode
    model.train()
    
    # keep track of total loss in this epoch
    batch_losses = []
    pos_distances = []
    neg_distances = []
    div_distances = []

    mean_foc = torch.empty((len(dataloader),model.num_bands))
    
    x_info: QueryBatchInfo
    y_info: QueryTargetInfo
    for batch_nr, (x_info, y_info) in enumerate(tqdm(dataloader, desc="Epoch", unit="batch", position=1, leave=False)):
        
        optimizer.zero_grad()
        hyp = model(x_info)
        pos_emb, neg_emb = model.embed_targets(y_info)
        loss, p_loss, n_loss, d_loss, _, focus = loss_fn(hyp, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()

        mean_foc[batch_nr] = focus
        
        loss_val = loss.detach().item()
        # Log batch metrics to WandB
        curr_batch = epoch * len(dataloader) + batch_nr
        wandb.log({
            "train": {
                "batch": {
                    "batch_loss": loss_val, "batch_p_dist": p_loss, "batch_n_dist": n_loss, "batch_d_dist": d_loss, "batch_id": curr_batch}}})

        batch_losses.append(loss_val)
        pos_distances.append(p_loss)
        neg_distances.append(n_loss)
        div_distances.append(d_loss)

    # mean focus
    logger.info(f"Mean focus: {torch.mean(focus, dim=0).detach().cpu().numpy()}")
    
    # Log epoch metrics to WandB
    mean_loss   = np.mean(batch_losses)
    mean_dist_p = np.mean(pos_distances)
    mean_dist_n = np.mean(neg_distances)
    mean_dist_d = np.mean(div_distances)
    
    # Log epoch metrics to WandB
    logger.info(f"Mean epoch loss: {mean_loss:.5f} ({mean_dist_p:.5f} + {mean_dist_n:.5f} + {mean_dist_d:.5f})")
    wandb.log({
        "train": {
            "epoch": {
                "mean_loss": mean_loss, "mean_dist_p": mean_dist_p, "mean_dist_n": mean_dist_n, "mean_dist_d": mean_dist_d, "epoch_id": epoch}}})
    
    return mean_loss


def train(
    model: AnswerSpaceModel,
    train_dataloader: DataLoader,
    loss_fn: AnswerSpaceLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    num_epochs: int,
    val_dataloader: DataLoader,
    val_freq: int,
    early_stopper: EarlyStopping = None,
    trial: Optional[optuna.Trial] = None
    ) -> Tuple[Sequence[float], DataFrame]:

    # Keep track of total loss during training
    epoch_losses = []

    # Keep track of classification statistics during training
    val_report = ClassificationReport()
    
    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training", unit="Epoch", position=0):
        
        # Train for one epoch
        epoch_loss = _train_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            epoch)
        epoch_losses.append(epoch_loss)

        # Update learning rate scheduler
        scheduler.step(epoch_loss)

        # Evaluate
        if (epoch + 1) % val_freq == 0:
            val_results = evaluate(
                model,
                val_dataloader,
                epoch) # Means we're tracking answer set size

            # Log val results and include in training report
            val_report.include(val_results, epoch)
            logger.info(f"Validation results: {val_results}")
            wandb.log({"val": {**val_results,  "epoch_id": epoch}})

            # If needed, handle pruning based on the intermediate objective value
            if trial is not None:
                trial.report(val_results['macro']['f1'], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        # Apply early stopping if needed
        if (early_stopper is not None) and (trial is None):
            early_stopper(epoch_loss)
            if early_stopper.early_stop:
                break

    return epoch_losses, val_report.src