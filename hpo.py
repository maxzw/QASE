"""Hyperparameter optimization module"""

import wandb
import pickle
import optuna
import logging
import argparse

import torch
from torch.utils.data import DataLoader

from helper import create_logger
from loader import get_dataloader, get_queries
from loss import AnswerSpaceLoss
from models import AnswerSpaceModel
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Suggest model parameters
    model_str = trial.suggest_categorical("optimizer", ["hypewise", "bandwise"])
    embed_dim = trial.suggest_int("embed_dim", 32, 512, log=True)
    num_bands = trial.suggest_int("num_bands", 4, 12)
    band_size = trial.suggest_int("band_size", 6, 26, step=4)
    gcn_layers = 3
    gcn_stop_dia = trial.suggest_categorical("gcn_stop_dia", [True, False]) if (model == "hypewise") else False
    gcn_pool = trial.suggest_categorical("gcn_pool", ["max", "sum", "tm"])
    gcn_comp = trial.suggest_categorical("gcn_comp", ["sub", "mult", "cmult", "cconv", "ccorr", "crot"])
    gcn_use_bias = True
    gcn_use_bn = True
    gcn_dropout = 0.5
    gcn_share_w = trial.suggest_categorical("gcn_share_w", [True, False]) if (model == "hypewise") else False

    model = AnswerSpaceModel(
        data_dir=dataset,
        method=model_str,
        embed_dim=embed_dim,
        device=device,
        num_bands=num_bands,
        band_size=band_size,
        gcn_layers=gcn_layers,
        gcn_stop_at_diameter=gcn_stop_dia,
        gcn_pool=gcn_pool,
        gcn_comp=gcn_comp,
        gcn_use_bias=gcn_use_bias,
        gcn_use_bn=gcn_use_bn,
        gcn_dropout=gcn_dropout,
        gcn_share_w=gcn_share_w
    )

    # Suggest loss parameters
    loss_pos_aggr = trial.suggest_categorical("optimizer", ["min", "softmin"])
    loss_fn = AnswerSpaceLoss(loss_pos_aggr)

    # Suggest optimizer parameters
    optim_name = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    if optim_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optim_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    config={
        "dataset": dataset,
        "model": model_str,
        "embed_dim": embed_dim,
        "device": device,
        "num_bands": num_bands,
        "band_size": band_size,        
        "gcn_layers": gcn_layers,
        "gcn_stop_dia": gcn_stop_dia,
        "gcn_pool": gcn_pool,
        "gcn_comp": gcn_comp,
        "gcn_use_bias": gcn_use_bias,
        "gcn_use_bn": gcn_use_bn,
        "gcn_dropout": gcn_dropout,
        "gcn_share_w": gcn_share_w,
        "loss_aggr": loss_pos_aggr,
        "optim": optim_name,
        "lr": learning_rate
    }

    # Initialize WandB logging
    wandb.login()
    wandb.init(
        project="thesis",
        config=config,
        group=dataset,
        job_type="hpo",
        reinit=True
        )

    # Train
    _, val_report = train(
        model = model,
        train_dataloader = train_dataloader,
        loss_fn = loss_fn,
        optimizer = optimizer,
        num_epochs = 50,
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

    train_dataloader = get_dataloader(train_queries, batch_size=128, shuffle=True, num_workers=2)
    val_dataloader = get_dataloader(val_queries, batch_size=128, shuffle=False, num_workers=2)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=123),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
    study.optimize(lambda trial: objective(trial, dataset, train_dataloader, val_dataloader), n_trials=100)
    pickle.dump(study, f"./results/{dataset}/hpo_study.pkl")