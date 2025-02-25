import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

# Todo: Initialization ?
class SqueezeAndExcitation(nn.Module):
    def __init__(
        self,
        n_channels: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()

        n_channels_red = n_channels // reduction
        assert n_channels_red > 0

        self.layers = nn.Sequential(
            nn.Conv2d(n_channels, n_channels_red, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(n_channels_red, n_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.layers(weighting)
        y = x * weighting
        return y

class SAEFusion(nn.Module):

    def __init__(self, feat_dim=256):
        super().__init__()
        self.conv_weight_color = []
        self.conv_weight_depth = []

        self.p2_conv_color = SqueezeAndExcitation(feat_dim)
        self.p3_conv_color = SqueezeAndExcitation(feat_dim)
        self.p4_conv_color = SqueezeAndExcitation(feat_dim)
        self.p5_conv_color = SqueezeAndExcitation(feat_dim)
        self.p6_conv_color = SqueezeAndExcitation(feat_dim)
        self.p7_conv_color = SqueezeAndExcitation(feat_dim)

        self.p2_conv_depth = SqueezeAndExcitation(feat_dim)
        self.p3_conv_depth = SqueezeAndExcitation(feat_dim)
        self.p4_conv_depth = SqueezeAndExcitation(feat_dim)
        self.p5_conv_depth = SqueezeAndExcitation(feat_dim)
        self.p6_conv_depth = SqueezeAndExcitation(feat_dim)
        self.p7_conv_depth = SqueezeAndExcitation(feat_dim)
        
        self.conv_weight_color.append(self.p2_conv_color)
        self.conv_weight_color.append(self.p3_conv_color)
        self.conv_weight_color.append(self.p4_conv_color)
        self.conv_weight_color.append(self.p5_conv_color)
        self.conv_weight_color.append(self.p6_conv_color)
        self.conv_weight_color.append(self.p7_conv_color)

        self.conv_weight_depth.append(self.p2_conv_depth)
        self.conv_weight_depth.append(self.p3_conv_depth)
        self.conv_weight_depth.append(self.p4_conv_depth)
        self.conv_weight_depth.append(self.p5_conv_depth)
        self.conv_weight_depth.append(self.p6_conv_depth)
        self.conv_weight_depth.append(self.p7_conv_depth)

    def forward(self, color_feat, depth_feat):

        res = dict()

        for i, k in enumerate(color_feat.keys()):
            tmp = self.conv_weight_color[i](color_feat[k]) + self.conv_weight_depth[i](depth_feat[k])
            res[k] = tmp
        return res
    
    # TO-DO override this function from base class nn.Module to make it callable from the main model
    def to(self, device):
        
        for conv in self.conv_weight_color:
            conv.to(device)

        for conv in self.conv_weight_depth:
            conv.to(device)

class ConcatModality(nn.Module):

    '''
    sub_feats use the same feature_depth
    '''
    def __init__(self, sub_feats=['p2', 'p3', 'p4', 'p5', 'p6', 'p7'], feature_depth=256):

        super().__init__()
        self.conv_down_depth = []
        
        self.p2_conv = nn.Conv2d(512, 256, (1, 1))
        self.p3_conv = nn.Conv2d(512, 256, (1, 1))
        self.p4_conv = nn.Conv2d(512, 256, (1, 1))
        self.p5_conv = nn.Conv2d(512, 256, (1, 1))
        self.p6_conv = nn.Conv2d(512, 256, (1, 1))
        self.p7_conv = nn.Conv2d(512, 256, (1, 1))
        
        self.conv_down_depth.append(self.p2_conv)
        self.conv_down_depth.append(self.p3_conv)
        self.conv_down_depth.append(self.p4_conv)
        self.conv_down_depth.append(self.p5_conv)
        self.conv_down_depth.append(self.p6_conv)
        self.conv_down_depth.append(self.p7_conv)
        # for sub_ in sub_feats:
        #     # self.conv_down_depth[sub_]= nn.Conv2d(512, 256, (1, 1))
        #     self.conv_down_depth.append(nn.Conv2d(512, 256, (1, 1)))

        for layer in self.conv_down_depth:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, -2.19)

    def forward(self, color_feat, depth_feat):

        res = dict()

        for i, k in enumerate(color_feat.keys()):
            tmp = torch.cat((color_feat[k], depth_feat[k]), 1)
            res[k] = self.conv_down_depth[i](tmp)
        return res
    
    # TO-DO override this function from base class nn.Module to make it callable from the main model
    def to(self, device):
        
        for conv in self.conv_down_depth:
            conv.to(device)

'''
Fusing two feature of color and depth after process FPN
'''
def average_depth_color_feature(color_feat, depth_feat):
    
    for k in color_feat.keys():
        color_feat[k] = color_feat[k]*0.5 + depth_feat[k]*0.5
    return color_feat

# Refer: https://github.com/prismformore/Multi-Task-Transformer/blob/75b90d21d113c9a777e6b22ca1a54955d78ddbfa/TaskPrompter/data/cityscapes3d.py#L153
def depth_norm(x, baseline, fx, type_norm=1, depth_mean=None, depth_std=None, min_depth_noise_threshold=0, max_depth_noise_threshold=500, \
               disparity_sgm_original=False):

    if disparity_sgm_original:
        x[x>0] = (x[x>0]-1) / 256 

    x[x>0] = baseline * fx / x[x>0]

    if type_norm == 1 or type_norm == 2: 
        x /= torch.max(x) # in this case, x[x>0] similar x
    
    if type_norm == 2:
        x *= 255 # in this case, x[x>0] similar x

    if type_norm==3:
        x[torch.logical_or(x<=min_depth_noise_threshold, x>=max_depth_noise_threshold)] = 0
        x[torch.logical_and(min_depth_noise_threshold < x, x<max_depth_noise_threshold)] = \
            (x[torch.logical_and(min_depth_noise_threshold < x, x<max_depth_noise_threshold)]-depth_mean)/depth_std

    return x
