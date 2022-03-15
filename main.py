from datetime import datetime
import logging

from helper import TqdmLoggingHandler
from loss import AnswerSpaceLoss
from models import HypewiseGCN
from loader import *
from train import train

# set log level
save_logs = True
dt = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
filename = f"./results/logs/{dt}.txt"
open(filename, "x").close()
logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(name)5s | %(levelname)5s | %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S',
            handlers=[
                logging.FileHandler(filename),
                TqdmLoggingHandler()
            ])

model = HypewiseGCN(
    data_dir="./data/AIFB/processed/",
    embed_dim=128,
    device=None,
    num_bands=4,
    num_hyperplanes=6,
    gcn_layers=3,
    gcn_stop_at_diameter=True,
    gcn_pool='tm',
    gcn_comp='mult',
    gcn_use_bias=True,
    gcn_use_bn=True,
    gcn_dropout=0.0,
    )
    
loss_fn = AnswerSpaceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# only train on 1-chain queries, and test on 1- and 2-diameter queries
train_queries, train_info = get_queries(model.data_dir, split='train', exclude=['2-chain', '3-chain', '2-inter', '3-inter', '3-inter_chain', '3-chain_inter'])
val_queries, val_info = get_queries(model.data_dir, split='val', exclude=['3-inter', '3-inter_chain', '3-chain_inter'])

train_dataloader = get_dataloader(
    train_queries,
    batch_size = 50,
    shuffle=True,
    num_workers=2
    )

val_dataloader = get_dataloader(
    val_queries,
    batch_size = 50,
    shuffle=False,
    num_workers=2
    )

epoch_losses, classification_data = train(
    model=model,
    train_dataloader=train_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    num_epochs=2,
    val_dataloader=val_dataloader,
    val_freq=1
)

print(epoch_losses)