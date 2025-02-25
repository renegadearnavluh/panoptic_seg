from .backbone_utils import LastLevelP6P7fromP5
from .depth_modal import ConcatModality, SAEFusion
from .resnet import BasicBlock, BasicStem, BottleneckBlock, DeformBottleneckBlock, ResNet
from torch import nn
import torch
from detectron2.layers import Conv2d, ShapeSpec, get_norm,  CNNBlockBase
from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling import FPN
import numpy as np
import torch.nn.functional as F

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y

class Excitation(nn.Module):
    # This is taken from https://github.com/PRBonn/PS-res-excite/blob/master/src/models/rgb_depth_fusion.py
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(Excitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = self.fc(x)
        y = x * weighting
        return y
    
class ResidualExciteFusion(nn.Module):
    # This is the fusion module from https://github.com/PRBonn/PS-res-excite/blob/master/src/models/rgb_depth_fusion.py
    
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(ResidualExciteFusion, self).__init__()

        self.se_rgb = Excitation(channels_in,
                                           activation=activation)
        self.se_depth = Excitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        if rgb.sum().item() < 1e-6:
            pass
        else:
            rgb_se = self.se_rgb(rgb)

        if depth.sum().item() < 1e-6:
            pass
        else:
            depth = self.se_depth(depth)

        out = rgb + rgb_se + depth
        return out
    
class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out
    
class ACWModule(nn.Module):
    # This is an implementation of the RGB-D Adaptive Channel Weighting module from the paper
    # Asymmetric Adaptive Fusion in a Two-Stream Network for RGB-D Human Detection
    def __init__(self, in_channels_rgb, in_channels_depth, reduction=16):
        super(ACWModule, self).__init__()
        assert in_channels_rgb == in_channels_depth
        self.in_channels = in_channels_rgb + in_channels_depth
        
        # Bottleneck: Conv -> ReLU -> Conv -> Sigmoid 
        self.excitation = Excitation(self.in_channels, reduction)
        self.out_con = nn.Conv2d(self.in_channels, in_channels_rgb, (1, 1))

    def forward(self, rgb_feat, depth_feat):
        """
        Forward pass for ACW module.
        Args:
            rgb_feat (Tensor): RGB feature map of shape (B, C1, H, W)
            depth_feat (Tensor): Depth feature map of shape (B, C2, H, W)
        Returns:
            Tensor: Output feature map with adaptive channel weighting applied.
        """
        # Step 1: Concatenate RGB and Depth feature maps along the channel dimension
        x = torch.cat((rgb_feat, depth_feat), dim=1)  # Shape: (B, C, H, W)
        return self.out_con(self.excitation(x))
       