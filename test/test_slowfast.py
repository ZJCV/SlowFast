# -*- coding: utf-8 -*-

"""
@date: 2020/10/7 下午4:26
@file: slowonly.py
@author: zj
@description: 
"""

import torch
from tsn.config import cfg
from tsn.model.backbones.resnet3d_slowfast.build_slowfast import resnet3d_50_slowfast
from tsn.model.backbones.resnet3d_slowfast.build_slowonly import resnet3d_50_slowonly


def test_slowfast():
    cfg.merge_from_file('configs/slowfast_r3d50_ucf101_rgb_224x32_dense.yaml')
    cfg.freeze()

    model = resnet3d_50_slowfast(cfg)
    print(model)

    data = torch.randn(1, 3, 32, 224, 224)
    outputs = model(data)
    print(len(outputs))
    print(outputs[0].shape)
    print(outputs[1].shape)

    assert len(outputs) == 2
    assert outputs[0].shape == (1, 2048, 4, 7, 7)
    assert outputs[1].shape == (1, 256, 32, 7, 7)


def test_slowonly():
    cfg.merge_from_file('configs/slowonly_r3d50_ucf101_rgb_224x4_dense.yaml')
    cfg.freeze()

    model = resnet3d_50_slowonly(cfg)
    print(model)

    data = torch.randn(1, 3, 4, 224, 224)
    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (1, 2048, 4, 7, 7)


if __name__ == '__main__':
    test_slowonly()
    test_slowfast()
