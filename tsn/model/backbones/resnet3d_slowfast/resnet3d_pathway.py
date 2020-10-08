# -*- coding: utf-8 -*-

"""
@date: 2020/10/7 下午3:02
@file: resnet3d_pathway.py
@author: zj
@description: 
"""

import torch.nn as nn

from tsn.model.backbones.resnet3d.resnet3d import ResNet3d
from tsn.model.backbones.resnet3d.utility import convTxHxW
from tsn.model.backbones.resnet3d_slowfast.conv_module import ConvModule


class ResNet3dPathway(ResNet3d):
    """A pathway of Slowfast based on ResNet3d.

    Args:
        *args (arguments): Arguments same as :class:``ResNet3d``.
        lateral (bool): Determines whether to enable the lateral connection
            from another pathway. Default: False.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            ``alpha`` in the paper. Default: 8.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to ``beta`` in the paper.
            Default: 8.
        fusion_kernel (int): The kernel size of lateral fusion.
            Default: 5.
        **kwargs (keyword arguments): Keywork arguments for ResNet3d.
    """

    def __init__(self,
                 *args,
                 lateral=False,
                 speed_ratio=8,
                 channel_ratio=8,
                 fusion_kernel=5,
                 **kwargs):
        self.lateral = lateral
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        self.fusion_kernel = fusion_kernel
        super().__init__(*args, **kwargs)

        self.inplanes = self.base_channels
        if self.lateral:
            self.pool1_lateral = ConvModule(self.inplanes // self.channel_ratio,
                                            self.inplanes * 2 // self.channel_ratio,
                                            kernel_size=(fusion_kernel, 1, 1),
                                            stride=(self.speed_ratio, 1, 1),
                                            padding=((fusion_kernel - 1) // 2, 0, 0),
                                            bias=False,
                                            norm_layer=self.norm_layer,
                                            act_layer=self.act_layer)

        self.lateral_connections = []
        for i in range(len(self.stage_blocks)):
            planes = self.base_channels * 2 ** i
            self.inplanes = planes * self.block.expansion

            if lateral and i != self.num_stages - 1:
                # no lateral connection needed in final stage
                lateral_name = f'layer{(i + 1)}_lateral'
                setattr(
                    self, lateral_name,
                    ConvModule(
                        self.inplanes // self.channel_ratio,
                        self.inplanes * 2 // self.channel_ratio,
                        kernel_size=(fusion_kernel, 1, 1),
                        stride=(self.speed_ratio, 1, 1),
                        padding=((fusion_kernel - 1) // 2, 0, 0),
                        bias=False,
                        norm_layer=self.norm_layer,
                        act_layer=self.act_layer))
                self.lateral_connections.append(lateral_name)

    def make_res_layer(self,
                       block,
                       planes,
                       blocks,
                       spatial_stride=1,
                       temporal_stride=1,
                       dilation=1,
                       inflate=1,
                       inflate_style='3x1x1',
                       non_local=0,
                       norm_layer=None,
                       act_layer=None,
                       **kwargs):
        """Build residual layer for ResNet3D.

        Args:
            block (nn.Module): Residual module to be built.
            planes (int): Number of channels for the output feature
                in each block.
            blocks (int): Number of residual blocks.
            spatial_stride (int | Sequence[int]): Spatial strides in
                residual and conv layers. Default: 1.
            temporal_stride (int | Sequence[int]): Temporal strides in
                residual and conv layers. Default: 1.
            dilation (int): Spacing between kernel elements. Default: 1.
            inflate (int | Sequence[int]): Determine whether to inflate
                for each block. Default: 1.
            inflate_style (str): ``3x1x1`` or ``1x1x1``. which determines
                the kernel sizes and padding strides for conv1 and conv2
                in each block. Default: '3x1x1'.
            non_local (int | Sequence[int]): Determine whether to apply
                non-local module in the corresponding block of each stages.
                Default: 0.
            norm_layer (nn.Module): norm layers.
                Default: None.
            act_layer (nn.Module): activation layer.
                Default: None.
        Returns:
            nn.Module: A residual layer for the given config.
        """
        inflate = inflate if not isinstance(inflate, int) else (inflate,) * blocks
        non_local = non_local if not isinstance(non_local, int) else (non_local,) * blocks
        assert len(inflate) == blocks and len(non_local) == blocks

        if self.lateral:
            lateral_inplanes = self.inplanes * 2 // self.channel_ratio
        else:
            lateral_inplanes = 0

        downsample = None
        if spatial_stride != 1 or (self.inplanes + lateral_inplanes) != planes * block.expansion:
            downsample = nn.Sequential(
                convTxHxW(
                    self.inplanes + lateral_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(temporal_stride, spatial_stride, spatial_stride),
                    padding=0,
                    bias=False,
                ),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes + lateral_inplanes,
                planes,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                downsample=downsample,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                non_local=(non_local[0] == 1),
                norm_layer=norm_layer,
                act_layer=act_layer,
            ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    spatial_stride=1,
                    temporal_stride=1,
                    dilation=dilation,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    non_local=(non_local[i] == 1),
                    norm_layer=norm_layer,
                    act_layer=act_layer
                ))

        return nn.Sequential(*layers)
