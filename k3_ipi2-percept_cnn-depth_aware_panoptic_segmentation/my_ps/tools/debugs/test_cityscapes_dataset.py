import init_path
from my_ps.data.cityscapes.cityscapes_panoptic_separated import register_all_cityscapes_panoptic
from detectron2.data import DatasetCatalog, MetadataCatalog, get_detection_dataset_dicts
from detectron2.data.common import MapDataset
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser
import os
from my_ps.models.panopticfcn import add_panopticfcn_config
from my_ps.utils.configs import add_common_config
from my_ps.data.cityscapes.dataset_mapper import CityscapesPanopticDatasetMapper
import imageio
import numpy as np

def setup_config(config_file):
    cfg = get_cfg()
    add_common_config(cfg)
    add_panopticfcn_config(cfg)
    cfg.merge_from_file(config_file)
    return cfg

if __name__=="__main__":

    args = default_argument_parser().parse_args()
    cfg = setup_config(args.config_file)

    cityscapes_dataset = register_all_cityscapes_panoptic(cfg, "/media/tuan/Daten/project/dataset/cityscapes/")
    mapper = CityscapesPanopticDatasetMapper(cfg)
    print(type(DatasetCatalog.get("my_cityscapes_fine_panoptic_train_separated")))

    dataset = get_detection_dataset_dicts(
            "my_cityscapes_fine_panoptic_train_separated",
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )

    print(type(dataset))
    map_dataset = MapDataset(dataset, mapper)

    print(len(map_dataset))
    print(map_dataset[0]["depth_file_name"])

    # Processing depth information following: https://github.com/prismformore/Multi-Task-Transformer/blob/75b90d21d113c9a777e6b22ca1a54955d78ddbfa/TaskPrompter/data/cityscapes3d.py#L150
    depth = np.array(imageio.imread(map_dataset[0]["depth_file_name"]) , dtype=np.float32) 
    print(depth.shape)

    # Test augmentation
