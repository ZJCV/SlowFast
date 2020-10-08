# -*- coding: utf-8 -*-

"""
@date: 2020/10/7 下午3:02
@file: resnet3d_pathway.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from tsn.model.backbones.resnet3d.basic_block_3d import BasicBlock3d
from tsn.model.backbones.resnet3d.bottleneck_3d import Bottleneck3d
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
                 type='slow',
                 lateral=False,
                 speed_ratio=8,
                 channel_ratio=8,
                 fusion_kernel=5,
                 **kwargs):
        self.type = type
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
        num_stages = len(self.stage_blocks)
        for i in range(len(self.stage_blocks)):
            planes = self.base_channels * 2 ** i
            self.inplanes = planes * self.block.expansion

            if lateral and i != num_stages - 1:
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

    def _init_weights(self, state_dict_2d):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                if (
                        hasattr(m, "transform_final_bn")
                        and m.transform_final_bn
                ):
                    batchnorm_weight = 0.0
                else:
                    batchnorm_weight = 1.0

                nn.init.constant_(m.weight, batchnorm_weight)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3d):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock3d):
                    nn.init.constant_(m.bn2.weight, 0)

        if state_dict_2d and self.type == 'slow':

            def _inflate_conv_params(conv3d, state_dict_2d, module_name_2d,
                                     inflated_param_names):
                """Inflate a conv module from 2d to 3d.

                The differences of conv modules betweene 2d and 3d in Pathway
                mainly lie in the inplanes due to lateral connections. To fit the
                shapes of the lateral connection counterpart, it will expand
                parameters by concatting conv2d parameters and extra zero paddings.

                Args:
                    conv3d (nn.Module): The destination conv3d module.
                    state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
                    module_name_2d (str): The name of corresponding conv module in the
                        2d model.
                    inflated_param_names (list[str]): List of parameters that have been
                        inflated.
                """
                weight_2d_name = module_name_2d + '.weight'
                conv2d_weight = state_dict_2d[weight_2d_name]
                old_shape = conv2d_weight.shape
                new_shape = conv3d.weight.data.shape
                kernel_t = new_shape[2]
                if new_shape[1] != old_shape[1]:
                    # Inplanes may be different due to lateral connections
                    new_channels = new_shape[1] - old_shape[1]
                    pad_shape = old_shape
                    pad_shape = pad_shape[:1] + (new_channels,) + pad_shape[2:]
                    # Expand parameters by concat extra channels
                    conv2d_weight = torch.cat(
                        (conv2d_weight,
                         torch.zeros(pad_shape).type_as(conv2d_weight).to(
                             conv2d_weight.device)),
                        dim=1)
                new_weight = conv2d_weight.data.unsqueeze(2).expand_as(
                    conv3d.weight) / kernel_t
                conv3d.weight.data.copy_(new_weight)
                inflated_param_names.append(weight_2d_name)

                if getattr(conv3d, 'bias') is not None:
                    bias_2d_name = module_name_2d + '.bias'
                    conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
                    inflated_param_names.append(bias_2d_name)

            def _inflate_bn_params(bn3d, state_dict_2d, module_name_2d,
                                   inflated_param_names):
                """Inflate a norm module from 2d to 3d.

                Args:
                    bn3d (nn.Module): The destination bn3d module.
                    state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
                    module_name_2d (str): The name of corresponding bn module in the
                        2d model.
                    inflated_param_names (list[str]): List of parameters that have been
                        inflated.
                """
                for param_name, param in bn3d.named_parameters():
                    param_2d_name = f'{module_name_2d}.{param_name}'
                    param_2d = state_dict_2d[param_2d_name]
                    param.data.copy_(param_2d)
                    inflated_param_names.append(param_2d_name)

                for param_name, param in bn3d.named_buffers():
                    param_2d_name = f'{module_name_2d}.{param_name}'
                    # some buffers like num_batches_tracked may not exist in old
                    # checkpoints
                    if param_2d_name in state_dict_2d:
                        param_2d = state_dict_2d[param_2d_name]
                        param.data.copy_(param_2d)
                        inflated_param_names.append(param_2d_name)

            inflated_param_names = []
            for name, module in self.named_modules():
                if 'non_local' in name:
                    continue
                if isinstance(module, nn.Conv3d):
                    _inflate_conv_params(module, state_dict_2d, name, inflated_param_names)
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    _inflate_bn_params(module, state_dict_2d, name, inflated_param_names)

            # check if any parameters in the 2d checkpoint are not loaded
            remaining_names = set(
                state_dict_2d.keys()) - set(inflated_param_names)
            if remaining_names:
                print(f'These parameters in the 2d checkpoint are not loaded: {remaining_names}')
