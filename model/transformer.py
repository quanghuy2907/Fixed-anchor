# ------------------------------------------------------------- #
# Fixed-anchor                                                  #
# Basic transformer architecture                                #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from model.model_utils import *


class Transformer(nn.Module):
    def __init__(self, 
                 d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False, num_queries=10):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries

        self.pos_embed = PositionEmbeddingSine(num_pos_feats=d_model//2, normalize=True)
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        self.encoder = TransformerEncoder(d_model, dim_feedforward,
                                          dropout, activation,
                                          nhead, num_encoder_layers)

        self.decoder = TransformerDecoder(d_model, dim_feedforward,
                                          dropout, activation,
                                          nhead, num_decoder_layers, return_intermediate_dec)

        

    def forward(self, src):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        
        pos_embed = self.pos_embed(src)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        src = src.flatten(2).permute(2, 0, 1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, pos_embed)
        hs = self.decoder(tgt, memory, pos_embed, query_embed)

        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
             



class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation='relu',
                 n_heads=8):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src
    

class TransformerEncoder(nn.Module):
    def __init__(self, 
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation='relu',
                 n_heads=8, num_layers=4):
        super().__init__()

        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, d_ffn,
                                                             dropout, activation,
                                                             n_heads) 
                                     for _ in range(num_layers)])

    def forward(self, src, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        return output
    


class TransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_heads=8):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos), value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, 
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_heads=8, num_layers=4, return_intermediate=False):
        super().__init__()
          
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, d_ffn,
                                                             dropout, activation,
                                                             n_heads) 
                                     for _ in range(num_layers)])
        
        self.return_intermediate = return_intermediate
        
    def forward(self, tgt, memory, pos=None, query_pos=None):
        inter = []

        output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output.unsqueeze(0)
        
      

            






