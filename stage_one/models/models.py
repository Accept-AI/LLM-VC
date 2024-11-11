import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

import time
import numpy as np

from .backbone import build_backbone
from .token_encoder import build_token_encoder


# ResNet18
class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()

        self.num_class = args.num_activities

        # model parameters
        self.num_frame = args.num_frame
        self.hidden_dim = args.hidden_dim

        # feature extraction
        self.backbone = build_backbone(args)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.backbone.num_channels, self.num_class)

        for name, m in self.named_modules():
            if 'backbone' not in name and 'token_encoder' not in name:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        :param x: [B, T, 3, H, W]
        :return:
        """
        #print("----base module----")
        b, t, _, h, w = x.shape
        x = x.reshape(b * t, 3, h, w)
        src, pos = self.backbone(x)                                                             # [B x T, C, H', W']
        #print("src: ", src.shape)    #src:  torch.Size([18, 512, 23, 40])
        _, c, oh, ow = src.shape
        representations = self.avg_pool(src)
        #print("representations1: ", representations.shape)    #representations1:  torch.Size([18, 512, 1, 1])
        representations = representations.reshape(b, t, c)
        #print("representations2: ", representations.shape)   #representations2:  torch.Size([1, 18, 512])

        representations = representations.reshape(b * t, self.backbone.num_channels)        # [B, T, F]
        #print("representations3: ", representations.shape)  # representations3:  torch.Size([18, 512])
        activities_scores = self.classifier(representations)
        #print("activities_scores1: ", activities_scores.shape)  # activities_scores1:  torch.Size([18, 9])
        activities_scores = activities_scores.reshape(b, t, -1).mean(dim=1)
        #print("activities_scores: ", activities_scores.shape)    # activities_scores2:  torch.Size([B, num_class])


        return activities_scores


# ViT


