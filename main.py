"""Main script"""
import wandb
import logging
from argparse import ArgumentParser

from helper import EarlyStopping, create_logger
from loss import QASEAnswerSpaceLoss
from models import AnswerSpaceModel
from loader import *
from train import train
from evaluation import evaluate


parser = ArgumentParser()

# Dataset & model parameters
parser.add_argument("--dataset",        type=str,   default="AIFB",     help="Which dataset to use: ['AIFB', 'AM', 'BIO', 'MUTAG']")
parser.add_argument("--model",          type=str,   default="bandwise", help="Which model to use: ['hypewise', 'bandwise']")
parser.add_argument("--embed_dim",      type=int,   default=128,        help="The embedding dimension of the entities and relations")
parser.add_argument("--num_bands",      type=int,   default=8,          help="The number of bands")
parser.add_argument("--band_size",      type=int,   default=16,         help="The size of the bands (number of hyperplanes per band)")

# GCN parameters
parser.add_argument("--gcn_layers",     type=int,   default=3,          help="The number of layers per gcn model: [1, 2, 3]")
parser.add_argument("--gcn_stop_dia",   type=bool,  default=False,       help="If message passing stopping stops when number of passes equals query diameter")
parser.add_argument("--gcn_pool",       type=str,   default="max",       help="Graph pooling operator: ['max', 'sum', 'tm']")
parser.add_argument("--gcn_comp",       type=str,   default="mult",     help="Composition operator: ['sub', 'mult', 'cmult', 'cconv', 'ccorr', 'crot']")
parser.add_argument("--gcn_use_bias",   type=bool,  default=True,       help="If convolution layer contains bias")
parser.add_argument("--gcn_use_bn",     type=bool,  default=True,       help="If convolution layer contains batch normalization")
parser.add_argument("--gcn_dropout",    type=float, default=0.5,        help="If convolution layer contains dropout")
parser.add_argument("--gcn_share_w",    type=bool,  default=False,      help="If the weights of the convolution layer are shared within a GCN")

# Loss parameters
parser.add_argument("--pos_w",          type=float, default=1.0,        help="The weight of the positive loss")
parser.add_argument("--neg_w",          type=float, default=1.0,        help="The weight of the negative loss")
parser.add_argument("--div_w",          type=float, default=0.5,        help="The weight of the diversity loss")

# Optimizer parameters
parser.add_argument("--optim",          type=str,   default="adam",     help="Optimizer: ['adam', 'sgd']")
parser.add_argument("--lr",             type=float, default=1e-3,       help="Learning rate")
parser.add_argument("--use_sched",      type=bool,  default=True,       help="If learning rate should be reduced over time")
parser.add_argument("--sched_pt",       type=int,   default=2,          help="Lr scheduler patience")
parser.add_argument("--sched_f",        type=float, default=0.5,        help="lr scheduler reducing factor")

# Training parameters
parser.add_argument("--num_epochs",     type=int,   default=50,         help="Number of training epochs")
parser.add_argument("--val_freq",       type=int,   default=1,          help="Validation frequency (epochs)")
parser.add_argument("--early_stop",     type=bool,  default=True,       help="If we use early stopping")
parser.add_argument("--stop_pt",        type=int,   default=5,          help="Early stopping patience (epochs)")
parser.add_argument("--stop_delta",     type=float, default=0.0,        help="Early stopping delta")
args = parser.parse_args()

# Create logger
create_logger(args.dataset)
logging.info(f"Training on dataset: {args.dataset}")

# Create model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AnswerSpaceModel(
    data_dir= f"./data/{args.dataset}/processed",
    method = args.model,
    embed_dim = args.embed_dim,
    device = device,
    num_bands = args.num_bands,
    band_size = args.band_size,
    gcn_layers = args.gcn_layers,
    gcn_stop_at_diameter = args.gcn_stop_dia,
    gcn_pool = args.gcn_pool,
    gcn_comp = args.gcn_comp,
    gcn_use_bias = args.gcn_use_bias,
    gcn_use_bn = args.gcn_use_bn,
    gcn_dropout = args.gcn_dropout,
    gcn_share_weights = args.gcn_share_w
)
logging.info(f"Model: {model}")

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

# Define loss function
loss_fn = QASEAnswerSpaceLoss(
    pos_w=args.pos_w,
    neg_w=args.neg_w,
    div_w=args.div_w
)
logging.info(f"Loss: {loss_fn}")

# Define optimizer
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), args.lr)
logging.info(f"Optimizer: {optimizer}")

# Define learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=args.sched_pt,
    factor=args.sched_f,
    verbose=True) if args.use_sched else None
logging.info(f"Scheduler: {scheduler}")

# Initialize early stopper
early_stopper = EarlyStopping(
    patience=args.stop_pt,
    min_delta=args.stop_delta,
) if args.early_stop else None
logging.info(f"Early stopper: {early_stopper}")

# Initialize WandB logging
wandb.login()
wandb.init(
    project="thesis",
    config=args,
    group=args.dataset,
    job_type="run"
    )

# Run training
epoch_losses, val_report = train(
    model=model,
    train_dataloader=train_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=args.num_epochs,
    val_dataloader=val_dataloader,
    val_freq=args.val_freq,
    early_stopper=early_stopper)
logging.info(epoch_losses)
logging.info(val_report)

# Evaluate on test data
test_results = evaluate(model, test_dataloader)
logging.info(test_results)
wandb.log({"test": test_results})

# Finalize run
summary_value = test_results['weighted']['f1']
wandb.run.summary["weighted_f1"] = summary_value

# # Save model if needed
# if args.save_best and is_best(args.dataset, test_results, metric='weighted_f1'):
#     # Archive files of previous best
#     archive_prev_best(args.dataset)
#     # For new best, save test results, model arguments and model weights
#     pickle.dump(test_results, open(f"./results/{args.dataset}/test_results.pkl", 'wb'))
#     pickle.dump(args.__dict__, open(f"./results/{args.dataset}/model_args.pkl", 'wb'))
torch.save(model.state_dict(), f"./results/{args.dataset}/{wandb.run.name}.pt")
