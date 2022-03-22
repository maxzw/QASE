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
parser.add_argument("--dataset",        type=str,   default="AIFB")
parser.add_argument("--model",          type=str,   default="hypewise")
# TODO: add load_model argument, if True: load from dataset folder and use saved params to initiate model,
# then load model state dict.
parser.add_argument("--embed_dim",      type=int,   default=128)
parser.add_argument("--num_bands",      type=int,   default=8)
parser.add_argument("--band_size",      type=int,   default=6)

# GCN parameters
parser.add_argument("--gcn_layers",     type=int,   default=3)
parser.add_argument("--gcn_stop_dia",   type=bool,  default=True)
parser.add_argument("--gcn_pool",       type=str,   default="tm")
parser.add_argument("--gcn_comp",       type=str,   default="mult")
parser.add_argument("--gcn_use_bias",   type=bool,  default=True)
parser.add_argument("--gcn_use_bn",     type=bool,  default=True)
parser.add_argument("--gcn_dropout",    type=float, default=0.3)
parser.add_argument("--gcn_share_w",    type=bool,  default=True)

# Loss parameters
parser.add_argument("--dist",           type=str,   default="sigm")
parser.add_argument("--loss_aggr",      type=str,   default="softmin")

# Training parameters
# TODO: add do_train
parser.add_argument("--optim",          type=str,   default=0.01)
parser.add_argument("--lr",             type=float, default=0.01)
parser.add_argument("--num_epochs",     type=int,   default=0.01)
parser.add_argument("--val_freq",       type=int,   default=0.01)
parser.add_argument("--early_stop",     type=int,   default=0.01)
# TODO: add do_test
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
if args.model == "hypewise":
    model = HypewiseGCN(
        data_dir=f"./data/{args.dataset}/processed/",
        embed_dim=50,
        device=device,
        num_bands=2,
        num_hyperplanes=4,
        gcn_layers=3,
        gcn_stop_at_diameter=True,
        gcn_pool='tm',
        gcn_comp='mult',
        gcn_use_bias=True,
        gcn_use_bn=True,
        gcn_dropout=0.3,
        gcn_share_weights=True)
else:
    raise NotImplementedError
logging.info(f"Model: {model}")

# Define loss function
loss_fn = AnswerSpaceLoss(
    dist_func=SigmoidDistance(),
    aggr=args.loss_aggr)
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
    num_epochs=2,
    val_dataloader=val_dataloader,
    val_freq=1)
logging.info(epoch_losses)
logging.info(val_report)

# Evaluate on test data
test_results = evaluate(model, test_dataloader)
logging.info(test_results)
wandb.log({"test": {**test_results}})

# Finalize run
weighted_f1 = test_results['weighted']['f1']
wandb.run.summary["weighted_f1"] = weighted_f1

# TODO:
# check if there are previous results if not, or current results are better:
# save all results in ./results/{dataset}/{result_name}.npy and save model AND arguments.
# if better_than_current(test_report):
    # save...

# save
# with open('commandline_args.txt', 'w') as f:
#     json.dump(args.__dict__, f, indent=2)

# load
# parser = ArgumentParser()
# args = parser.parse_args()
# with open('commandline_args.txt', 'r') as f:
#     args.__dict__ = json.load(f)

torch.save(model.state_dict(), f"./results/{args.dataset}/{wandb.run.name}.pt")