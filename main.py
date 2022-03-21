"""Main script"""
import logging
from datetime import datetime
from turtle import pos

from helper import TqdmLoggingHandler
from loss import AnswerSpaceLoss, SigmoidDistance
from models import HypewiseGCN
from loader import *
from train import train
from evaluation import evaluate

dataset = "AIFB"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set log level
dt = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
filename = f"./results/{dataset}/logs/{dt}.txt"
open(filename, "x").close()
logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(name)5s | %(levelname)5s | %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S',
            handlers=[
                logging.FileHandler(filename),
                TqdmLoggingHandler()
            ])

logging.info(f"Training on dataset: {dataset}")

model = HypewiseGCN(
    data_dir=f"./data/{dataset}/processed/",
    embed_dim=128,
    device=None,
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
logging.info(f"Model: {model}")
    
loss_fn = AnswerSpaceLoss(
    dist_func=SigmoidDistance(),
    aggr='softmin')
logging.info(f"Loss: {loss_fn}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
logging.info(f"Optimizer: {optimizer}")

# only train and test on 1-chain queries
exclude = ['2-chain', '3-chain', '2-inter', '3-inter', '3-inter_chain', '3-chain_inter']
train_queries, train_info = get_queries(model.data_dir, split='train', exclude=exclude)
logging.info(f"Train info: {train_info}")
val_queries, val_info = get_queries(model.data_dir, split='val', exclude=exclude)
logging.info(f"Val info: {val_info}")
test_queries, test_info = get_queries(model.data_dir, split='test', exclude=exclude)
logging.info(f"Test info: {test_info}")

train_dataloader = get_dataloader(train_queries, batch_size = 128, shuffle=True, num_workers=2)
val_dataloader = get_dataloader(val_queries, batch_size = 128, shuffle=False, num_workers=2)
test_dataloader = get_dataloader(test_queries, batch_size = 128, shuffle=False, num_workers=2)

epoch_losses, val_report = train(
    model=model,
    train_dataloader=train_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    num_epochs=100,
    val_dataloader=val_dataloader,
    val_freq=1)
logging.info(epoch_losses)
logging.info(val_report)

test_report = evaluate(model, test_dataloader)
logging.info(test_report)

torch.save(model.state_dict(), f"./results/{dataset}/best_model.pt")

# TODO: 
# check if there are previous results if not, or current results are better:
# save all results in ./results/{dataset}/{result_name}.npy and save model.
# if better_than_current(test_report):
    # save...


# TODO: use Colab for GPU
# TODO: use AIM / WandB