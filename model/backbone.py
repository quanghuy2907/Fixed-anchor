# ------------------------------------------------------------- #
# Fixed-anchor                                                  #
# Fixed-anchor backbone                                         #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #

import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter


class Backbone(nn.Module):
    def __init__(self, return_interm_layers: bool):
        super().__init__()
        
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]

        self.body = IntermediateLayerGetter(self.backbone, return_layers=return_layers)

    def forward(self, input):
        xs = self.body(input)

        feat = []
        for name, x in xs.items():
            feat.append(x)

        return feat

        