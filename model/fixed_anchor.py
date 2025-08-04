# ------------------------------------------------------------- #
# Fixed-anchor                                                  #
# Fixed-anchor model                                            #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #

import torch
import torch.nn.functional as F
from torch import nn

from model.backbone import Backbone
from model.transformer import Transformer
from model.model_utils import MLP


class Fixed_anchor(nn.Module):
    def __init__(self, 
                 d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1,
                 num_encoder_layers=6, num_decoder_layers=6,
                 num_queries=10, activation="relu", return_intermediate_dec=False):
        super().__init__()
       
        self.backbone = Backbone(return_interm_layers=False)
        self.transformer = Transformer(d_model, nhead, 
                                       num_encoder_layers, num_decoder_layers, 
                                       dim_feedforward, dropout, 
                                       activation, return_intermediate_dec, num_queries)
        
        self.input_proj = nn.Conv2d(in_channels=self.backbone.num_channels[-1], out_channels=d_model, kernel_size=1, stride=1, padding=0)

        
        self.conf = MLP(input_dim=d_model, output_dim=1, hidden_dim=d_model)
        self.junc1 = MLP(input_dim=d_model, output_dim=2, hidden_dim=d_model)
        self.junc2 = MLP(input_dim=d_model, output_dim=2, hidden_dim=d_model)
        self.ori1 = MLP(input_dim=d_model, output_dim=2, hidden_dim=d_model)
        self.ori2 = MLP(input_dim=d_model, output_dim=2, hidden_dim=d_model)
        self.slot_type = MLP(input_dim=d_model, output_dim=3, hidden_dim=d_model)
        self.slot_occ = MLP(input_dim=d_model, output_dim=1, hidden_dim=d_model)

    def forward(self, img_input):
        feats = self.backbone(img_input)[-1]
        src = self.input_proj(feats)
        
        hs = self.transformer(src)[0]

        conf = F.sigmoid(self.conf(hs))
        junc1 = F.tanh(self.junc1(hs))
        junc2 = F.tanh(self.junc2(hs))
        ori1 = F.tanh(self.ori1(hs))
        ori2 = F.tanh(self.ori2(hs))
        slot_type = F.softmax(self.slot_type(hs), dim=-1)
        slot_occ = F.sigmoid(self.slot_occ(hs))

        output = torch.cat([conf, junc1, junc2, ori1, ori2, slot_type, slot_occ], dim=-1)

        return output[0]
