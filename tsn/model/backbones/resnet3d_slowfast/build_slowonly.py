# -*- coding: utf-8 -*-

"""
@date: 2020/10/7 下午8:30
@file: build_slowonly.py
@author: zj
@description: 
"""

from torchvision.models.utils import load_state_dict_from_url

from .slowonly import SlowOnly
from tsn.model import registry

__all__ = ['SlowOnly', 'resnet3d_18_slowonly', 'resnet3d_34_slowonly', 'resnet3d_50_slowonly', 'resnet3d_101_slowonly',
           'resnet3d_152_slowonly', ]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def _resnet(arch, cfg, map_location=None):
    pretrained2d = cfg.MODEL.BACKBONE.TORCHVISION_PRETRAINED
    state_dict_2d = None
    if pretrained2d:
        state_dict_2d = load_state_dict_from_url(model_urls[arch],
                                                 progress=True,
                                                 map_location=map_location)

    type = cfg.MODEL.BACKBONE.TYPE
    lateral = cfg.MODEL.BACKBONE.LATERAL
    speed_ratio = cfg.MODEL.BACKBONE.SPEED_RATIO
    channel_ratio = cfg.MODEL.BACKBONE.CHANNEL_RATIO
    fusion_kernel = cfg.MODEL.BACKBONE.FUSION_KERNEL

    slow_in_channels = cfg.MODEL.BACKBONE.SLOW_PATHWAY.IN_CHANNELS
    slow_spatial_strides = cfg.MODEL.BACKBONE.SLOW_PATHWAY.SPATIAL_STRIDES
    slow_temporal_strides = cfg.MODEL.BACKBONE.SLOW_PATHWAY.TEMPORAL_STRIDES
    slow_dilations = cfg.MODEL.BACKBONE.SLOW_PATHWAY.DILATIONS
    slow_base_channel = cfg.MODEL.BACKBONE.SLOW_PATHWAY.BASE_CHANNEL
    slow_conv1_kernel = cfg.MODEL.BACKBONE.SLOW_PATHWAY.CONV1_KERNEL
    slow_conv1_stride_t = cfg.MODEL.BACKBONE.SLOW_PATHWAY.CONV1_STRIDE_T
    slow_pool1_kernel_t = cfg.MODEL.BACKBONE.SLOW_PATHWAY.POOL1_KERNEL_T
    slow_pool1_stride_t = cfg.MODEL.BACKBONE.SLOW_PATHWAY.POOL1_STRIDE_T
    slow_with_pool2 = cfg.MODEL.BACKBONE.SLOW_PATHWAY.WITH_POOL2
    slow_inflates = cfg.MODEL.BACKBONE.SLOW_PATHWAY.INFLATES
    slow_inflate_style = cfg.MODEL.BACKBONE.SLOW_PATHWAY.INFLATE_STYLE
    slow_non_local = cfg.MODEL.BACKBONE.SLOW_PATHWAY.NON_LOCAL
    model = SlowOnly(arch,
                     in_channels=slow_in_channels,
                     spatial_strides=slow_spatial_strides,
                     temporal_strides=slow_temporal_strides,
                     dilations=slow_dilations,
                     base_channel=slow_base_channel,
                     conv1_kernel=slow_conv1_kernel,
                     conv1_stride_t=slow_conv1_stride_t,
                     pool1_kernel_t=slow_pool1_kernel_t,
                     pool1_stride_t=slow_pool1_stride_t,
                     with_pool2=slow_with_pool2,
                     inflates=slow_inflates,
                     inflate_style=slow_inflate_style,
                     non_local=slow_non_local,
                     zero_init_residual=True,
                     lateral=lateral,
                     speed_ratio=speed_ratio,
                     channel_ratio=channel_ratio,
                     fusion_kernel=fusion_kernel,
                     state_dict_2d=state_dict_2d)
    return model


@registry.BACKBONE.register('resnet3d_18_slowonly')
def resnet3d_18_slowonly(cfg, map_location=None):
    return _resnet("resnet18", cfg, map_location=map_location)


@registry.BACKBONE.register('resnet3d_34_slowonly')
def resnet3d_34_slowonly(cfg, map_location=None):
    return _resnet("resnet34", cfg, map_location=map_location)


@registry.BACKBONE.register('resnet3d_50_slowonly')
def resnet3d_50_slowonly(cfg, map_location=None):
    return _resnet("resnet50", cfg, map_location=map_location)


@registry.BACKBONE.register('resnet3d_101_slowonly')
def resnet3d_101_slowonly(cfg, map_location=None):
    return _resnet("resnet101", cfg, map_location=map_location)


@registry.BACKBONE.register('resnet3d_152_slowonly')
def resnet3d_152_slowonly(cfg, map_location=None):
    return _resnet("resnet152", cfg, map_location=map_location)
