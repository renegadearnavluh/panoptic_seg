from .rgb_depth_fusion import ResidualExciteFusion, SqueezeAndExciteFusionAdd
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


class ResNetWithFusion(Backbone):
    def __init__(self, rgb_stem, depth_stem, rgb_stages, depth_stages, fusion_module_type, out_features, freeze_at=0, num_classes=None, should_forward_fused_features=False):
        """
        Fused ResNet backbone that accepts pre-initialized RGB and Depth stages.

        Args:
            rgb_stem (nn.Module): Stem for the RGB input.
            depth_stem (nn.Module): Stem for the Depth input.
            rgb_stages (List[nn.Sequential]): Stages for RGB input.
            depth_stages (List[nn.Sequential]): Stages for Depth input.
            fusion_module_type (string): Fusion module tye.
            out_features (List[str]): Features to output during the forward pass.
            freeze_at (int): Number of stages to freeze (for fine-tuning).
        """
        super().__init__()

        self.add_module("rgb_stem", rgb_stem)
        self.add_module("depth_stem", depth_stem)
        self.rgb_stem = rgb_stem
        self.depth_stem = depth_stem
        self.num_classes = num_classes

        self.rgb_stages = []
        self.depth_stages = []
        self._out_features = out_features
        self.freeze_at = freeze_at
        self.stage_names = []
        self._out_feature_strides = {"stem": rgb_stem.stride}
        self._out_feature_channels = {"stem": rgb_stem.out_channels}
       
        self.fusion_modules = nn.ModuleDict()
        self.should_forward_fused_features = should_forward_fused_features

        current_stride = rgb_stem.stride

        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            num_stages =  num_stages = max(
                    [{"res2": 1, "res3": 2, "res4": 3, "res5": 4}.get(f, 0) for f in out_features]
                )
            rgb_stages = rgb_stages[:num_stages]
            depth_stages = depth_stages[:num_stages]    
        for i, (rgb_blocks, depth_blocks) in enumerate(zip(rgb_stages, depth_stages)):
            assert len(rgb_blocks) > 0, len(rgb_blocks)
            assert len(depth_blocks) > 0, len(depth_blocks)
            for block in rgb_blocks + depth_blocks:
                assert isinstance(block, CNNBlockBase), block
            stage_name = f"res{i + 2}"
            rgb_stage = nn.Sequential(*rgb_blocks)
            depth_stage = nn.Sequential(*depth_blocks)
            self.add_module(f"{stage_name}", rgb_stage)
            self.add_module(f"depth_{stage_name}", depth_stage)
            self.rgb_stages.append(rgb_stage)
            self.depth_stages.append(depth_stage)
            self.stage_names.append(stage_name)
            
            # Add fusion module for the stage
            in_channels = rgb_blocks[-1].out_channels
            if fusion_module_type == "residual_excite":
                self.fusion_modules[stage_name] = ResidualExciteFusion(in_channels)
            elif fusion_module_type == "sae_add":
                self.fusion_modules[stage_name] = SqueezeAndExciteFusionAdd(in_channels)
            elif fusion_module_type == "mean":
                class MeanFusion(nn.Module):
                    def forward(self, rgb, depth):
                        return (0.5*rgb + 0.5*depth)  # Element-wise mean fusion

                self.fusion_modules[stage_name] = MeanFusion()
            elif fusion_module_type == "concat":
                class ConcatFusion(nn.Module):
                    def __init__(self, in_channels):
                        super().__init__()
                        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)

                    def forward(self, rgb, depth):
                        fused = torch.cat((rgb, depth), dim=1)  # Concatenate along channel dimension
                        return self.conv(fused)  # Reduce channels back to original

                self.fusion_modules[stage_name] = ConcatFusion(in_channels)
            else:
                raise ValueError(f"Unsupported fusion module type: {fusion_module_type}")

            self._out_feature_strides[stage_name] = current_stride = int(
                current_stride * np.prod([k.stride for k in rgb_blocks])
            )
            self._out_feature_channels[stage_name] = curr_channels = rgb_blocks[-1].out_channels
            # TODO: confirm this current_stride *= 2
        self.stage_names = tuple(self.stage_names)  # Make it static for scripting

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            stage_name = "linear"

        if out_features is None:
            out_features = [stage_name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))
        # Freeze initial layers if required
        self.freeze(freeze_at)

    def forward(self, rgb_input, depth_input):
        """
        Forward pass for fused ResNet.

        Args:
            rgb_input: RGB input tensor.
            depth_input: Depth input tensor.

        Returns:
            Dict[str, Tensor]: Fused features at each requested output level.
        """
        rgb_features = self.rgb_stem(rgb_input)
        depth_features = self.depth_stem(depth_input)

        outputs = {}
        if "stem" in self._out_features:
            outputs["stem"] = rgb_features

        for i, (rgb_stage, depth_stage) in enumerate(zip(self.rgb_stages, self.depth_stages)):
            stage_name = f"res{i + 2}"
            rgb_features = rgb_stage(rgb_features)
            depth_features = depth_stage(depth_features)

            # Fuse features
            fused_features = self.fusion_modules[stage_name](rgb_features, depth_features)
            if self.should_forward_fused_features:
                rgb_features = fused_features  # Feed fused features into the next stage


            if stage_name in self._out_features:
                outputs[stage_name] = fused_features
        if self.num_classes is not None:
            x = self.avgpool(rgb_features)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x

        return outputs

    def output_shape(self):
        """
        Returns the output shape for each requested feature.
        """
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first few stages of the network for fine-tuning.
        """
        if freeze_at >= 1:
            self.rgb_stem.freeze()
            self.depth_stem.freeze()
        for idx, (rgb_stage, depth_stage) in enumerate(zip(self.rgb_stages, self.depth_stages), start=2):
            if freeze_at >= idx:
                for block in rgb_stage.children():
                    block.freeze()
                for block in depth_stage.children():
                    block.freeze()

        return self

    @staticmethod
    def make_stage(block_class, num_blocks, *, in_channels, out_channels, norm="BN", **kwargs):
        """
        Creates stages for both RGB and Depth modalities.

        Args:
            block_class (type): Block class (e.g., BasicBlock, BottleneckBlock).
            num_blocks (int): Number of blocks in this stage.
            in_channels (int): Input channels for the stage.
            out_channels (int): Output channels for every block in the stage.
            norm (str): Normalization type.
            kwargs: Additional arguments for block initialization.

        Returns:
            Tuple[List[CNNBlockBase], List[CNNBlockBase]]:
                - RGB stage blocks
                - Depth stage blocks
        """
        rgb_blocks = []
        depth_blocks = []

        for i in range(num_blocks):
            block_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith("_per_block"):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the same length as num_blocks={num_blocks}."
                    )
                    newk = k[: -len("_per_block")]
                    assert newk not in kwargs, f"Cannot call make_stage with both {k} and {newk}!"
                    block_kwargs[newk] = v[i]  # Remove "_per_block" suffix
                else:
                    block_kwargs[k] = v

            rgb_blocks.append(block_class(in_channels=in_channels, out_channels=out_channels, norm=norm, **block_kwargs))
            depth_blocks.append(block_class(in_channels=in_channels, out_channels=out_channels, norm=norm, **block_kwargs))
            in_channels = out_channels

        return rgb_blocks, depth_blocks



#@BACKBONE_REGISTRY.register()
def build_resnet_with_fusion_backbone(cfg, input_shape, should_forward_fused_features=False):
    """
    Build a ResNetWithFusion backbone.

    Args:
        cfg: Detectron2 configuration object.
        input_shape: Input shape specification.

    Returns:
        ResNetWithFusion: A fused ResNet backbone instance.
    """
    # Configuration values
    norm = cfg.MODEL.RESNETS.NORM
    depth = cfg.MODEL.RESNETS.DEPTH
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    stem_out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS

    # Sanity checks for configurations
    assert res5_dilation in {1, 2}, f"Invalid res5_dilation: {res5_dilation}"
    assert depth in [18, 34, 50, 101, 152], f"Unsupported ResNet depth: {depth}"

    # Initialize stems
    rgb_stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=stem_out_channels,
        norm=norm,
    )
    depth_stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=stem_out_channels,
        norm=norm,
    )
    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"


    # Stage parameters
    rgb_stages = []
    depth_stages = []
    in_channels = stem_out_channels
    out_channels = res2_out_channels
    for idx, stage_idx in enumerate(range(2, 6)):
        # res5_dilation is used this way as a convention in R-FCN & Deformable Conv paper
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs["block_class"] = DeformBottleneckBlock
                stage_kargs["deform_modulated"] = deform_modulated
                stage_kargs["deform_num_groups"] = deform_num_groups
            else:
                stage_kargs["block_class"] = BottleneckBlock

        rgb_blocks, depth_blocks = ResNetWithFusion.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        rgb_stages.append(rgb_blocks)
        depth_stages.append(depth_blocks)
    
    return ResNetWithFusion(rgb_stem=rgb_stem, 
                            depth_stem=depth_stem, 
                            rgb_stages=rgb_stages, 
                            depth_stages=depth_stages, 
                            fusion_module_type=cfg.MODEL.DEPTH_MODALITY.FUSION_TYPE, 
                            out_features=out_features, 
                            freeze_at=freeze_at,
                            should_forward_fused_features=should_forward_fused_features)


class FPNWithFusion(FPN):
    def __init__(
        self,
        bottom_up,  # Fused ResNet backbone
        in_features,
        out_channels,
        norm="",
        top_block=None,
        fuse_type="sum",
    ):
        """
        Extended FPN to handle features from a ResNetWithFusion backbone.

        Args:
            bottom_up (Backbone): The fused ResNet backbone.
            in_features (list[str]): Names of the input feature maps from the backbone.
            out_channels (int): Number of output channels for FPN layers.
            norm (str): Normalization for FPN layers (optional).
            top_block (nn.Module, optional): Extra layers for higher-level features (e.g., P6, P7).
            fuse_type (str): Method for fusing lateral and top-down features ("sum" or "avg").
        """
        super().__init__(bottom_up, in_features, out_channels, norm, top_block, fuse_type)

    def forward(self, rgb_inputs, depth_inputs):
        """
        Forward pass for FPN with ResNetWithFusion.

        Args:
            rgb_inputs (Tensor): RGB input tensor.
            depth_inputs (Tensor): Depth input tensor.

        Returns:
            Dict[str, Tensor]: Feature pyramid levels (e.g., "p2", "p3", "p4", "p5").
        """
        # Pass RGB and Depth inputs to the fused ResNet backbone
        bottom_up_features = self.bottom_up(rgb_inputs, depth_inputs)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}



def build_resnet_fusion_fpn_backbone(cfg, input_shape, should_forward_fused_features=True):
    """
    Build ResNet-FPN backbone with P6 and P7 from P5 feature.

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    bottom_up = build_resnet_with_fusion_backbone(cfg, input_shape, should_forward_fused_features=should_forward_fused_features)
    
    in_features = cfg.MODEL.FPN.IN_FEATURES

    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = out_channels
    backbone = FPNWithFusion(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7fromP5(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


