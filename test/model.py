# -*- coding: utf-8 -*-

"""
@date: 2020/9/10 下午9:10
@file: model.py
@author: zj
@description: 
"""

from tsn.config import cfg
from tsn.model.build import build_model

if __name__ == '__main__':
    cfg.merge_from_file('configs/tsn_resnet50_ucf101_rgb_rgbdiff.yaml')

    model = build_model(cfg)
    print(model)
