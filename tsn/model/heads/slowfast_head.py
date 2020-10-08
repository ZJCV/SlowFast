# -*- coding: utf-8 -*-

"""
@date: 2020/10/8 下午2:26
@file: slowfast_head.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from tsn.model import registry


@registry.HEAD.register('SlowFastHead')
class SlowFastHead(nn.Module):

    def __init__(self, cfg):
        super(SlowFastHead, self).__init__()

        in_channels = cfg.MODEL.HEAD.FEATURE_DIMS
        num_classes = cfg.MODEL.HEAD.NUM_CLASSES
        dropout_rate = cfg.MODEL.HEAD.DROPOUT

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x_slow = x[0]
        x_slow = self.avgpool(x_slow)

        x_fast = x[1]
        x_fast = self.avgpool(x_fast)

        x = torch.cat((x_slow, x_fast), dim=1)

        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
