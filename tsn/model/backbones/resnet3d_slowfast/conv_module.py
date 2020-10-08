# -*- coding: utf-8 -*-

"""
@date: 2020/10/7 下午3:39
@file: conv_module.py
@author: zj
@description: 
"""

import torch.nn as nn
from tsn.model.backbones.resnet3d.utility import convTx1x1


class ConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 bias=False,
                 norm_layer=None,
                 act_layer=None):
        super(ConvModule, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if act_layer is None:
            act_layer = nn.ReLU

        self.conv = convTx1x1(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.bn = norm_layer(out_channels)
        self.relu = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
