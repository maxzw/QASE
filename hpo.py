"""Hyperparameter optimization module"""

import argparse

import wandb
import pickle
import optuna
import logging

from torch.utils.data import DataLoader

from helper import create_logger
from loader import get_dataloader, get_queries
from train import train


def objective(
    trial: optuna.Trial,
    dataset: str,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader
    ) -> float:
    """Objective function to be optimized by Optuna.

    Args:
        trial (optuna.Trial): Optuna trial class.
        dataset (str): The dataset on which we optimize (used for grouping WandB log).
        train_dataloader (DataLoader): The train dataloader.
        val_dataloader (DataLoader): The validation dataloader.

    Returns:
        float: Final model score (macro-f1).
    """

    # Suggest parameters
    args=None
    # all is 3 layers
    # hypewise shared(+/-) + stop_dia(+/-)
    # bandwise (or set shard + stop_dia to false here)

    # Initialize WandB logging
    wandb.login()
    wandb.init(
        project="thesis",
        config=args,
        group=dataset,
        job_type="hpo",
        reinit=True
        )

    # Train
    _, val_report = train(
        model = None,
        train_dataloader = train_dataloader,
        loss_fn = None,
        optimizer = None,
        num_epochs = None,
        val_dataloader = val_dataloader,
        val_freq = 1,
        trial = trial
        )
    
    # Return last validation metric as score
    last_val_result = val_report.iloc[-1].to_dict('records')[0]
    score = last_val_result['macro']['f1']

    return score


# python hpy.py AIFB
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, nargs=1)
    dataset = parser.parse_args().dataset[0]

    # Create logger 
    create_logger(dataset)
    logging.info(f"Training on dataset: {dataset}")

    # Load data
    exclude = ['2-chain', '3-chain', '2-inter', '3-inter', '3-inter_chain', '3-chain_inter']
    train_queries, train_info = get_queries(f"./data/{dataset}/processed/", split='train', exclude=exclude)
    logging.info(f"Train info: {train_info}")
    val_queries, val_info = get_queries(f"./data/{dataset}/processed/", split='val', exclude=exclude)
    logging.info(f"Val info: {val_info}")

    train_dataloader = get_dataloader(train_queries, batch_size = 128, shuffle=True, num_workers=2)
    val_dataloader = get_dataloader(val_queries, batch_size = 128, shuffle=False, num_workers=2)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=123),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
    study.optimize(lambda trial: objective(trial, dataset, train_dataloader, val_dataloader), n_trials=100)
    pickle.dump(study, f"./results/{dataset}/hpo_study.pkl")