from detectron2.config import CfgNode as CN

def add_common_config(cfg):

    # data config
    # select the dataset mapper
    # cfg.INPUT.DATASET_MAPPER_NAME = "oneformer_unified"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    cfg.INPUT.TASK_SEQ_LEN = 77
    cfg.INPUT.MAX_SEQ_LEN = 77

    cfg.INPUT.TASK_PROB = CN()
    cfg.INPUT.TASK_PROB.SEMANTIC = 0.33
    cfg.INPUT.TASK_PROB.INSTANCE = 0.66

    # test dataset
    cfg.DATASETS.TEST_PANOPTIC = ("",)
    cfg.DATASETS.TEST_INSTANCE = ("",)
    cfg.DATASETS.TEST_SEMANTIC = ("",)

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # oneformer inference config
    cfg.MODEL.TEST = CN()
    cfg.MODEL.TEST.SEMANTIC_ON = True
    cfg.MODEL.TEST.INSTANCE_ON = False
    cfg.MODEL.TEST.PANOPTIC_ON = False
    cfg.MODEL.TEST.DETECTION_ON = False
    cfg.MODEL.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.TEST.TASK = "panoptic"

    cfg.DATA_DIR = ""
