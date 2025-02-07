# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_panopticfcn_config(cfg):
    """
    Add config for Panoptic FCN.
    """
    cfg.MODEL.TENSOR_DIM                 = 100
    cfg.MODEL.IGNORE_VALUE               = 255
    cfg.SOLVER.POLY_LR_POWER             = 0.9
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING   = 0.0

    cfg.MODEL.RESNETS.DEPTH_OUT_FEATURES = ["depth_res2", "depth_res3", "depth_res4", "depth_res5"]
    cfg.MODEL.FPN.DEPTH_IN_FEATURES = ["depth_res2", "depth_res3", "depth_res4", "depth_res5"]

    cfg.MODEL.SEMANTIC_FPN   = CN()
    cfg.MODEL.SEMANTIC_FPN.IN_FEATURES   = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.SEMANTIC_FPN.CONVS_DIM     = 256
    cfg.MODEL.SEMANTIC_FPN.COMMON_STRIDE = 4
    cfg.MODEL.SEMANTIC_FPN.NORM          = "GN"

    cfg.MODEL.POSITION_HEAD   = CN()
    cfg.MODEL.POSITION_HEAD.NUM_CONVS       = 3
    cfg.MODEL.POSITION_HEAD.COORD           = False
    cfg.MODEL.POSITION_HEAD.CONVS_DIM       = 256
    cfg.MODEL.POSITION_HEAD.NORM            = "GN"
    cfg.MODEL.POSITION_HEAD.DEFORM          = True

    cfg.MODEL.POSITION_HEAD.THING = CN()
    cfg.MODEL.POSITION_HEAD.THING.CENTER_TYPE    = "mass"
    cfg.MODEL.POSITION_HEAD.THING.POS_NUM        = 7
    cfg.MODEL.POSITION_HEAD.THING.NUM_CLASSES    = 80
    cfg.MODEL.POSITION_HEAD.THING.BIAS_VALUE     = -2.19
    cfg.MODEL.POSITION_HEAD.THING.MIN_OVERLAP    = 0.7
    cfg.MODEL.POSITION_HEAD.THING.GAUSSIAN_SIGMA = 3
    cfg.MODEL.POSITION_HEAD.THING.THRES          = 0.05
    cfg.MODEL.POSITION_HEAD.THING.TOP_NUM        = 100
    
    cfg.MODEL.POSITION_HEAD.STUFF = CN()
    cfg.MODEL.POSITION_HEAD.STUFF.NUM_CLASSES  = 54
    cfg.MODEL.POSITION_HEAD.STUFF.ALL_CLASSES  = False  # this one should have priority over "WITH_THING"
    cfg.MODEL.POSITION_HEAD.STUFF.WITH_THING   = True
    cfg.MODEL.POSITION_HEAD.STUFF.THRES        = 0.05

    cfg.MODEL.SEM_SEG_HEAD   = CN()
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES  = 54
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255

    cfg.MODEL.KERNEL_HEAD    = CN()
    cfg.MODEL.KERNEL_HEAD.INSTANCE_SCALES = ((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048),)
    cfg.MODEL.KERNEL_HEAD.TEST_SCALES     = ((1, 64), (64, 128), (128, 256), (256, 512), (512, 2048),)
    cfg.MODEL.KERNEL_HEAD.NUM_CONVS       = 3
    cfg.MODEL.KERNEL_HEAD.DEFORM          = False
    cfg.MODEL.KERNEL_HEAD.COORD           = True
    cfg.MODEL.KERNEL_HEAD.CONVS_DIM       = 256
    cfg.MODEL.KERNEL_HEAD.NORM            = "GN"

    cfg.MODEL.FEATURE_ENCODER    = CN()
    cfg.MODEL.FEATURE_ENCODER.IN_FEATURES     = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.FEATURE_ENCODER.NUM_CONVS       = 3
    cfg.MODEL.FEATURE_ENCODER.CONVS_DIM       = 64
    cfg.MODEL.FEATURE_ENCODER.DEFORM          = False
    cfg.MODEL.FEATURE_ENCODER.COORD           = True
    cfg.MODEL.FEATURE_ENCODER.NORM            = ""

    cfg.MODEL.LOSS_WEIGHT    = CN()
    cfg.MODEL.LOSS_WEIGHT.POSITION          = 1.0
    cfg.MODEL.LOSS_WEIGHT.SEGMENT           = 3.0
    cfg.MODEL.LOSS_WEIGHT.FOCAL_LOSS_ALPHA  = 0.25
    cfg.MODEL.LOSS_WEIGHT.FOCAL_LOSS_GAMMA  = 2.0

    cfg.MODEL.INFERENCE      = CN()
    cfg.MODEL.INFERENCE.INST_THRES        = 0.4
    cfg.MODEL.INFERENCE.SIMILAR_THRES     = 0.9
    cfg.MODEL.INFERENCE.SIMILAR_TYPE      = "cosine"
    cfg.MODEL.INFERENCE.CLASS_SPECIFIC    = True

    cfg.MODEL.INFERENCE.COMBINE  = CN()
    cfg.MODEL.INFERENCE.COMBINE.ENABLE           = True
    cfg.MODEL.INFERENCE.COMBINE.NO_OVERLAP       = False
    cfg.MODEL.INFERENCE.COMBINE.OVERLAP_THRESH   = 0.5
    cfg.MODEL.INFERENCE.COMBINE.STUFF_AREA_LIMIT = 4096
    cfg.MODEL.INFERENCE.COMBINE.INST_THRESH      = 0.2

    # Depth config
    cfg.MODEL.DEPTH_MODALITY = CN()
    cfg.MODEL.DEPTH_MODALITY.SHARED_WEIGHT = True
    cfg.MODEL.DEPTH_MODALITY.ENABLED = True
    cfg.MODEL.DEPTH_MODALITY.NORMALIZATION = 1
    cfg.MODEL.DEPTH_MODALITY.DEPTH_MEAN = 0.0
    cfg.MODEL.DEPTH_MODALITY.DEPTH_MAX = 0.0
    cfg.MODEL.DEPTH_MODALITY.DEPTH_STD = 0.0
    cfg.MODEL.DEPTH_MODALITY.FUSION_TYPE = "mean"
    cfg.MODEL.DEPTH_MODALITY.MIN_DEPTH_NOISE_THRESHOLD = 0.8
    cfg.MODEL.DEPTH_MODALITY.MAX_DEPTH_NOISE_THRESHOLD = 200.0
    cfg.MODEL.DEPTH_MODALITY.RAW_IMG_EXTENSION = "npy" # extension of depth image: png, npy, ...
    cfg.MODEL.DEPTH_AWARE_WEIGHT = 3.0

    cfg.MODEL.FREEZE_COLOR_BACKBONE = False

    cfg.DATASETS.NAME = "COCO"

    cfg.INPUT.CROP.MINIMUM_INST_AREA = 0


