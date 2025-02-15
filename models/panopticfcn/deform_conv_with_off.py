#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from detectron2.layers.deform_conv import DeformConv, ModulatedDeformConv


class DeformConvWithOff(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, deformable_groups=1):
        super(DeformConvWithOff, self).__init__()
        self.offset_conv = nn.Conv2d(
            in_channels,
            deformable_groups * 2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.dcn = DeformConv(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            deformable_groups=deformable_groups,
        )

    def forward(self, input):
        offset = self.offset_conv(input)
        output = self.dcn(input, offset)
        return output


class ModulatedDeformConvWithOff(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, deformable_groups=1,
                 bias=True, norm=None, activation=None):
        super(ModulatedDeformConvWithOff, self).__init__()
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            deformable_groups * 3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.dcnv2 = ModulatedDeformConv(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            deformable_groups=deformable_groups,
            bias=bias, norm=norm, activation=activation,
        )

    def forward(self, input):
        x = self.offset_mask_conv(input)
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        output = self.dcnv2(input, offset, mask)
        return output
        
class ModulatedDeformConvWithOffAndMask(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, deformable_groups=1,
                 bias=True, norm=None, activation=None, extra_offset_mask=False):
        super(ModulatedDeformConvWithOffAndMask, self).__init__()
        
        self.extra_offset_mask = extra_offset_mask
        channels_ = deformable_groups * 3 * kernel_size * kernel_size
        self.conv_offset_mask = nn.Conv2d(in_channels, channels_, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.dcnv2 = ModulatedDeformConv(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            deformable_groups=deformable_groups,
            bias=bias, norm=norm, activation=activation,
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        if self.extra_offset_mask:
            out = self.conv_offset_mask(input[1])
            input = input[0]
        else:
            out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)       
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        output = self.dcnv2(input, offset, mask)
        return output