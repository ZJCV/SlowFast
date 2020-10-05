# -*- coding: utf-8 -*-

"""
@date: 2020/9/28 下午8:59
@file: resnet3d.py
@author: zj
@description: 
"""

import torch

from tsn.model.backbones.resnet3d.build_resnet3d import resnet3d_50
from tsn.config import cfg

if __name__ == '__main__':
    # cfg.merge_from_file('configs/c2d_nl_r3d50_ucf101_rgb_224x224x32.yaml')
    # cfg.merge_from_file('configs/c2d_r3d50_ucf101_rgb_224x224x32.yaml')
    cfg.merge_from_file('configs/i3d_nl_3x1x1_r3d50_ucf101_rgb_224x224x32.yaml')
    cfg.freeze()

    model = resnet3d_50(cfg)
    print(model)

    data = torch.randn(1, 3, 32, 224, 224)
    outputs = model(data)
    print(outputs.shape)
