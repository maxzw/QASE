"""GCN model implementation using composition-based convolution."""

from typing import Sequence 

import torch
import torch.nn as nn
from torch import Tensor

from loader import VectorizedQueryBatch
from gnn.compgcn_conv import CompGCNConv
from gnn.message_passing import MessagePassing
from gnn.pooling import *


class GCNModel(nn.Module):
    def __init__(
        self,
        layer_dims: Sequence[int],
        stop_at_diameter: bool = True,
        pool: str = 'max',
        comp: str = 'mult',
        use_bias: bool = True,
        use_bn: bool = True,
        dropout: float = 0.0,
        share_weights: bool = True,
        device = None
        ):
        super().__init__()
        self.layer_dims = layer_dims
        if stop_at_diameter:
            assert len(set(self.layer_dims)) == 1, \
            f"Can only stop at diameter when using constant layer dimensions!"
        self.stop_at_diameter = stop_at_diameter
        assert pool in ['max', 'sum', 'tm'], \
            f"Readout function {pool} is not implemented!"
        self.pool_str = pool
        assert comp in ['sub', 'mult', 'cmult', 'cconv', 'ccorr', 'crot'], \
            f"Composition operation {comp} is not implemented!"
        self.comp = comp
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.dropout = dropout
        if share_weights:
            assert (len(set(self.layer_dims)) == 1), \
            f"Cannot share weights while using varying layer dimensions!"
        self.share_weights = share_weights
        self.device = device       
        
        # Create message passing layers
        layers = []
        conv_args = {
            'comp': self.comp,
            'use_bias': self.use_bias,
            'use_bn': self.use_bn,
            'dropout': self.dropout,
            'device': self.device
            }
        # If we share weights across layers, we assume all layer dimensions are the same
        if share_weights:
            conv_args.update({
                'in_channels': self.layer_dims[0],
                'out_channels': self.layer_dims[0],
            })
            conv = CompGCNConv(**conv_args)
            for i in range(len(self.layer_dims) - 2):
                layers += [
                    conv,
                    nn.ReLU()]
            layers += [conv]

        # If we do not share weights or have different in/output dimensions
        else:
            for i in range(len(self.layer_dims) - 2):
                conv_args.update({
                    'in_channels': self.layer_dims[i],
                    'out_channels': self.layer_dims[i+1],
                })
                layers += [
                    CompGCNConv(**conv_args),
                    nn.ReLU()]
            conv_args.update({
                    'in_channels': self.layer_dims[-2],
                    'out_channels': self.layer_dims[-1],
                })
            layers += [CompGCNConv(**conv_args)]
        
        # Save as ModuleList
        self.layers = nn.ModuleList(layers)

        # Define readout function
        if pool == 'max':
            self.pool = MaxGraphPooling()
        elif pool == 'sum':
            self.pool = SumGraphPooling()
        elif pool == 'tm':
            self.pool = TargetPooling()
        else:
            raise NotImplementedError


    def forward(self, data: VectorizedQueryBatch) -> Tensor:
        
        ent_embed, rel_embed, diameters = data.ent_embed, data.rel_embed, data.q_diameters

        # perform message passing
        convs = 0
        for layer in self.layers:
            
            # convolution layer
            if isinstance(layer, MessagePassing):
                
                # include boolean mask for entities where query diameter <= conv step
                ent_mask = torch.le(diameters, convs) if self.stop_at_diameter else None
                ent_embed, rel_embed = layer(ent_embed, rel_embed, data.edge_index, data.edge_type, ent_mask=ent_mask)
                convs += 1
            
            # otherwise its activation function
            else:
                ent_embed = layer(ent_embed)

        # aggregate node embeddings
        out = self.pool(
            embs        =   ent_embed,
            batch_idx   =   data.batch_idx,
            target_idx  =   data.target_idx
            )
        return out
