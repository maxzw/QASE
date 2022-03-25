"""Main script"""
import logging

from argparse import ArgumentParser
from torch.profiler import profile, record_function, ProfilerActivity

from helper import create_logger
from models import AnswerSpaceModel
from loader import *

parser = ArgumentParser()

# Dataset & model parameters
parser.add_argument("--dataset",        type=str,   default="AIFB",     help="Which dataset to use: ['AIFB', 'AM', 'BIO', 'MUTAG']")
# parser.add_argument("--load_best",      type=bool,  default=False,      help="If best model for this dataset should be loaded")
parser.add_argument("--model",          type=str,   default="hypewise", help="Which model to use: ['hypewise', 'bandwise']")
parser.add_argument("--embed_dim",      type=int,   default=128,        help="The embedding dimension of the entities and relations")
parser.add_argument("--num_bands",      type=int,   default=6,          help="The number of bands")
parser.add_argument("--band_size",      type=int,   default=4,          help="The size of the bands (number of hyperplanes per band)")

# GCN parameters
parser.add_argument("--gcn_layers",     type=int,   default=3,          help="The number of layers per gcn model: [1, 2, 3]")
parser.add_argument("--gcn_stop_dia",   type=bool,  default=True,      help="If message passing stopping stops when number of passes equals query diameter")
parser.add_argument("--gcn_pool",       type=str,   default="tm",       help="Graph pooling operator: ['max', 'sum', 'tm']")
parser.add_argument("--gcn_comp",       type=str,   default="mult",     help="Composition operator: ['sub', 'mult', 'cmult', 'cconv', 'ccorr', 'crot']")
parser.add_argument("--gcn_use_bias",   type=bool,  default=True,       help="If convolution layer contains bias")
parser.add_argument("--gcn_use_bn",     type=bool,  default=True,       help="If convolution layer contains batch normalization")
parser.add_argument("--gcn_dropout",    type=float, default=0.3,        help="If convolution layer contains dropout")
parser.add_argument("--gcn_share_w",    type=bool,  default=True,       help="If the weights of the convolution layer are shared within a GCN")

# Loss parameters
parser.add_argument("--dist",           type=str,   default="sigm",     help="The distance function used in the loss: ['sigm', 'invlrelu']")
parser.add_argument("--loss_aggr",      type=str,   default="min",      help="The aggregation technique for band distances of positive samples: ['min', 'mean', 'softmin']")

# Optimizer parameters
parser.add_argument("--optim",          type=str,   default="adam",     help="Optimizer: ['adam', 'sgd']")
parser.add_argument("--lr",             type=float, default=1e-3,       help="Learning rate")

# Training parameters
# parser.add_argument("--do_train",       type=bool,  default=True,       help="If we go through training loop (disable for testing loaded model)")
# parser.add_argument("--save_best",      type=bool,  default=True,       help="If model should be saved if it performs better than current best model")
parser.add_argument("--num_epochs",     type=int,   default=50,         help="Number of training epochs")
parser.add_argument("--val_freq",       type=int,   default=1,          help="Validation frequency (epochs)")
parser.add_argument("--min_epochs",     type=int,   default=5,          help="The minimal number of epochs to train (only relevant in combination with early stopping)")
parser.add_argument("--early_stop",     type=int,   default=None,       help="Number of rounds after training is stopped when loss does not go down")
# parser.add_argument("--do_test",        type=bool,  default=True,       help="If we evaluate on the test set")
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

exclude = ['2-chain', '3-chain', '2-inter', '3-inter', '3-inter_chain', '3-chain_inter']
train_queries, train_info = get_queries(model.data_dir, split='train', exclude=exclude)
train_dataloader = get_dataloader(train_queries, batch_size = 128, shuffle=True, num_workers=2)

model.eval()
for x_info, y_info in train_dataloader:
    
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_forward"):
            hyp = model(x_info)
    print(prof.key_averages().table(sort_by="cpu_time_total"))

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_embed"):
            pos_emb, neg_emb = model.embed_targets(y_info)
    print(prof.key_averages().table(sort_by="cpu_time_total"))

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_predict"):
            preds = model.predict(hyp, y_info.pos_modes)
    print(prof.key_averages().table(sort_by="cpu_time_total"))

    break
