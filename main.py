"""Main script."""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',       type=str,   default='./data/AIFB/processed/',   help="")

# architecture parameters
parser.add_argument("--embed_dim",      type=int,   default=128,        help="")
parser.add_argument("--num_buckets",    type=int,   default=128,        help="")
parser.add_argument("--num_planes",     type=int,   default=128,        help="")
parser.add_argument("--gcn_layers",     type=int,   default=128,        help="")
parser.add_argument("--aggr_method",    type=int,   default=128,        help="")

# optimizer parameters
parser.add_argument('--opt',            type=str,   default='adam',     help="")
parser.add_argument('--lr',             type=float, default=1e-5,       help="")

args = parser.parse_args()

# load data (start with AIFB!)
# create model
# create optimizer
# run train