import math
from .backbone_utils import LastLevelP6P7fromP5
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import torch
from torch import nn
import os
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from detectron2.modeling.backbone.backbone import Backbone

from .resnet import build_resnet_backbone
from .rgb_depth_fusion import ResidualExciteFusion, ACWModule
from .fan_backbone_utils import FeatureAlign_V2
import pdb

class AddMeanFusion(nn.Module):
    def __init__(self, factor=1):
        assert factor in [1, 2]
        super(AddMeanFusion, self).__init__()
        self.factor = factor
        
    def forward(self, x, y):
        return (x + y)/self.factor
    
class DepthEnhancedFAN(Backbone):
    def __init__(self, rgb_bottom_up, depth_bottom_up, in_features, in_features_depth, out_channels=256, norm="", top_block=None, fuse_type="sum", should_forward_fused_features=False):
        """
        Args:
            rgb_bottom_up (Backbone): RGB backbone network.
            depth_bottom_up (Backbone): Depth backbone network.
            in_features (list[str]): Feature map names from the backbone networks.
            out_channels (int): Number of channels in the output feature maps.
            norm (str): Normalization type.
            top_block (nn.Module or None): Additional module for extra FPN levels.
            fuse_type (str): Fusion strategy ("sum" or "avg").
        """
        super(DepthEnhancedFAN, self).__init__()
        assert isinstance(rgb_bottom_up, Backbone) and isinstance(depth_bottom_up, Backbone)

        
        input_shapes_rgb = rgb_bottom_up.output_shape()
        strides_rgb = [input_shapes_rgb[f].stride for f in in_features]
        in_channels_per_feature_rgb = [input_shapes_rgb[f].channels for f in in_features]

        input_shapes_depth = depth_bottom_up.output_shape()
        strides_depth = [input_shapes_depth[f].stride for f in in_features_depth]
        in_channels_per_feature_depth = [input_shapes_depth[f].channels for f in in_features_depth]

        self._assert_strides_are_log2_contiguous(strides_rgb)
        self._assert_strides_are_log2_contiguous(strides_depth)

        align_modules_rgb = []
        align_modules_depth = []
        rgb_depth_fusion_modules = []
        output_convs = []
        print(f"FAN {in_channels_per_feature_rgb},{in_channels_per_feature_depth},  {strides_rgb},{strides_depth}")
        assert fuse_type in {"mean", "sum", "residual_excite", "acw"}
        self._fuse_type = fuse_type
        
        use_bias = norm == ""
        for idx, (in_channels_rgb, in_channels_depth)  in enumerate(zip(in_channels_per_feature_rgb[:-1], in_channels_per_feature_depth[:-1])):
            stage_rgb = int(math.log2(strides_rgb[idx]))
            lateral_norm = get_norm(norm, out_channels)
            align_module_rgb = FeatureAlign_V2(in_channels_rgb, out_channels, norm=lateral_norm)  # proposed fapn
            self.add_module("fan_align_rgb{}".format(stage_rgb), align_module_rgb)
            align_modules_rgb.append(align_module_rgb)

            stage_depth = int(math.log2(strides_depth[idx]))
            align_module_depth = FeatureAlign_V2(in_channels_depth, out_channels, norm=lateral_norm)  # proposed fapn
            self.add_module("fan_align_depth{}".format(stage_depth), align_module_depth)
            align_modules_depth.append(align_module_depth)
            assert stage_rgb == stage_depth
            # TODO: experiment with different positions of fusion module here
            if fuse_type == "acw":
                rgb_depth_fusion_module = ACWModule(out_channels, out_channels)
            elif fuse_type == "residual_excite":
                rgb_depth_fusion_module = ResidualExciteFusion(out_channels)
            elif fuse_type == "mean":
                rgb_depth_fusion_module = AddMeanFusion(factor=2)
            elif fuse_type == "sum":
                rgb_depth_fusion_module = AddMeanFusion(factor=1)
            else:
                raise NotImplementedError(f"Unknown fusion type {fuse_type}")

            self.add_module("rgb_depth_fusion{}".format(stage_rgb), rgb_depth_fusion_module)
            rgb_depth_fusion_modules.append(rgb_depth_fusion_module)

            
            output_conv_rgb = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias,
                                 norm=lateral_norm, )
            weight_init.c2_xavier_fill(output_conv_rgb)
            self.add_module("fpn_output_rgb{}".format(stage_rgb), output_conv_rgb)
            output_convs.append(output_conv_rgb)


        stage_rgb = int(math.log2(strides_rgb[len(in_channels_per_feature_rgb) - 1]))
        lateral_conv_rgb = Conv2d(in_channels_per_feature_rgb[-1], out_channels, kernel_size=1, bias=use_bias,
                              norm=get_norm(norm, out_channels))
        align_modules_rgb.append(lateral_conv_rgb)
        self.add_module("fan_align_rgb{}".format(stage_rgb), lateral_conv_rgb)

        stage_depth = int(math.log2(strides_depth[len(in_channels_per_feature_depth) - 1]))
        lateral_conv_depth = Conv2d(in_channels_per_feature_depth[-1], out_channels, kernel_size=1, bias=use_bias,
                                norm=get_norm(norm, out_channels))
        align_modules_depth.append(lateral_conv_depth)
        self.add_module("fan_align_depth{}".format(stage_depth), lateral_conv_depth)
        # Place convs into top-down order (from low to high resolution) to make the top-down computation in forward clearer.
        if fuse_type == "acw":
            rgb_depth_fusion_module = ACWModule(out_channels, out_channels)
        elif fuse_type == "residual_excite":
            rgb_depth_fusion_module = ResidualExciteFusion(out_channels)
        elif fuse_type == "mean":
            rgb_depth_fusion_module = AddMeanFusion(factor=2)
        elif fuse_type == "sum":
            rgb_depth_fusion_module = AddMeanFusion(factor=1)
        else:
            raise NotImplementedError(f"Unknown fusion type {fuse_type}")

        self.add_module("rgb_depth_fusion{}".format(stage_rgb), rgb_depth_fusion_module)
        rgb_depth_fusion_modules.append(rgb_depth_fusion_module)

        self.align_modules_rgb = align_modules_rgb[::-1]
        self.output_convs = output_convs[::-1]
        self.rgb_depth_fusion_modules = rgb_depth_fusion_modules[::-1]

        assert stage_rgb == stage_depth

        self.align_modules_depth = align_modules_depth[::-1]

        self.bottom_up_rgb = rgb_bottom_up
        self.bottom_up_depth = depth_bottom_up

        self.in_features_rgb = in_features
        self.in_features_depth = in_features_depth

        self.top_block = top_block

        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides_rgb}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage_rgb, stage_rgb + 2):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides_rgb[-1]
        self.should_forward_fused_features = should_forward_fused_features
        print(f"FAN set up with {len(self.align_modules_rgb)} ,{len(self.align_modules_depth)}, {self._out_features}, {self.in_features_rgb}, {self.in_features_depth} ")
        
        #FAN [256, 512, 1024, 2048],[256, 512, 1024, 2048],  [4, 8, 16, 32],[4, 8, 16, 32]
        #FAN set up with 4 ,4, ['p2', 'p3', 'p4', 'p5', 'p6', 'p7'], ['res2', 'res3', 'res4', 'res5'], ['res2depth', 'res3depth', 'res4depth', 'res5depth'] 

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x_rgb, x_depth):
        """
        Args:
            x_rgb (dict[str -> Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order..
            x_depth (dict[str -> Tensor]): mapping depth feature map name (e.g., "depth_res5") to
                feature map tensor for each feature level in high to low resolution order..

        Returns:
            dict[str -> Tensor]: Merged feature maps in top-down order.
            mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """

        bottom_up_features_rgb = self.bottom_up_rgb(x_rgb)
        x_rgb = [bottom_up_features_rgb[f] for f in self.in_features_rgb[::-1]]

        bottom_up_features_depth = self.bottom_up_depth(x_depth)
        x_depth = [bottom_up_features_depth[f] for f in self.in_features_depth[::-1]]
        results = []
        prev_features_rgb = self.align_modules_rgb[0](x_rgb[0])
        prev_features_depth = self.align_modules_depth[0](x_depth[0])
        prev_features = self.rgb_depth_fusion_modules[0](prev_features_rgb, prev_features_depth)
        results.append(prev_features)
        for idx, (features, depth_feat, rgb_align_module, depth_align_module, fusion_module ) in enumerate(zip(x_rgb, x_depth, self.align_modules_rgb, self.align_modules_depth, self.rgb_depth_fusion_modules)):
            if idx>0:
                prev_features_rgb = rgb_align_module(features, prev_features_rgb)
                prev_features_depth = depth_align_module(depth_feat, prev_features_depth)
                prev_features = fusion_module(prev_features_rgb, prev_features_depth)
                output_conv = self.output_convs[idx-1]
                results.insert(0, output_conv(prev_features))
                if self.should_forward_fused_features:
                    # feed the fused features to the next bottom-up block
                    prev_features_rgb = prev_features
        
        if self.top_block is not None:
            top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
                for name in self._out_features}

    def _assert_strides_are_log2_contiguous(self, strides):
        """
        Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
        """
        for i, stride in enumerate(strides[1:], 1):
            assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(stride, strides[i - 1])

def build_resnet_depth_enhanced_fan_backbone(cfg, should_forward_fused_feat=False):
    input_shape_rgb = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    input_shape_depth = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    bottom_up_rgb = build_resnet_backbone(cfg,input_shape=input_shape_rgb, modal="color")
    bottom_up_depth = build_resnet_backbone(cfg,input_shape=input_shape_depth, modal= "depth")
    in_features = cfg.MODEL.FPN.IN_FEATURES
    in_features_depth = cfg.MODEL.FPN.DEPTH_IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    norm = cfg.MODEL.FPN.NORM
    in_channels_p6p7 = out_channels
    top_block = LastLevelP6P7fromP5(in_channels_p6p7, out_channels)
    fuse_type = cfg.MODEL.DEPTH_MODALITY.FUSION_TYPE
    return DepthEnhancedFAN(rgb_bottom_up=bottom_up_rgb, 
                            depth_bottom_up=bottom_up_depth, 
                            in_features=in_features, 
                            in_features_depth=in_features_depth, 
                            out_channels=out_channels, 
                            norm=norm, 
                            top_block=top_block, 
                            fuse_type=fuse_type, 
                            should_forward_fused_features=should_forward_fused_feat)