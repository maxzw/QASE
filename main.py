"""Main script"""
import wandb
import logging
from argparse import ArgumentParser

from helper import create_logger
from loss import AnswerSpaceLoss, SigmoidDistance
from models import HypewiseGCN
from loader import *
from train import train
from evaluation import evaluate

parser = ArgumentParser()

# Dataset & model parameters
parser.add_argument("--dataset",    type=str,       default="AIFB")
parser.add_argument("--model",      type=str,       default="Hypewise")

# Gcn parameters
parser.add_argument("--gcn_layers", type=int,       default=3)

# Training parameters
parser.add_argument("--lr",         type=float,     default=0.01)
args = parser.parse_args()

# Create logger
create_logger(args.dataset)
logging.info(f"Training on dataset: {args.dataset}")

# Login to WandB to track progress
wandb.login()
wandb.init(
    project="thesis",
    config=args
)

# Create model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HypewiseGCN(
    data_dir=f"./data/{args.dataset}/processed/",
    embed_dim=128,
    device=device,
    num_bands=5,
    num_hyperplanes=10,
    gcn_layers=3,
    gcn_stop_at_diameter=True,
    gcn_pool='tm',
    gcn_comp='mult',
    gcn_use_bias=True,
    gcn_use_bn=True,
    gcn_dropout=0.3,
    gcn_share_weights=True)

# Define loss function
loss_fn = AnswerSpaceLoss(
    dist_func=SigmoidDistance(),
    aggr='softmin')
logging.info(f"Loss: {loss_fn}")

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
logging.info(f"Optimizer: {optimizer}")

# Load queries
exclude = ['2-chain', '3-chain', '2-inter', '3-inter', '3-inter_chain', '3-chain_inter']
train_queries, train_info = get_queries(model.data_dir, split='train', exclude=exclude)
logging.info(f"Train info: {train_info}")
val_queries, val_info = get_queries(model.data_dir, split='val', exclude=exclude)
logging.info(f"Val info: {val_info}")
test_queries, test_info = get_queries(model.data_dir, split='test', exclude=exclude)
logging.info(f"Test info: {test_info}")

# Define dataloaders
train_dataloader = get_dataloader(train_queries, batch_size = 128, shuffle=True, num_workers=2)
val_dataloader = get_dataloader(val_queries, batch_size = 128, shuffle=False, num_workers=2)
test_dataloader = get_dataloader(test_queries, batch_size = 128, shuffle=False, num_workers=2)

# Run training
epoch_losses, val_report = train(
    model=model,
    train_dataloader=train_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    num_epochs=50,
    val_dataloader=val_dataloader,
    val_freq=1)
logging.info(epoch_losses)
logging.info(val_report)

# Evaluate on test data
test_report = evaluate(model, test_dataloader)
logging.info(test_report)

# Save model if needed
# remove old...
torch.save(model.state_dict(), f"./results/{args.dataset}/{wandb.run.name}.pt")

# TODO: 
# check if there are previous results if not, or current results are better:
# save all results in ./results/{dataset}/{result_name}.npy and save model.
# if better_than_current(test_report):
    # save...