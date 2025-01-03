import init_path
from my_ps.utils.demo.predictor import VisualizationDemo
import random
import numpy as np 
import torch
from detectron2.config import get_cfg
from my_ps.models.panopticfcn import add_panopticfcn_config
from detectron2.data.detection_utils import read_image
from my_ps.utils.configs import add_common_config
import os

from my_ps.data.cityscapes.cityscapes_panoptic_separated import register_all_cityscapes_panoptic

def setup_config():
    cfg = get_cfg()
    add_common_config(cfg)
    add_panopticfcn_config(cfg)
    # cfg.merge_from_file("../../my_ps/my_ps/configs/PanopticFCN-Star-R50-3x.yaml")
    cfg.merge_from_file("../../my_ps/my_ps/configs/cityscapes/PanopticFCN-R50-cityscapes.yaml")
    return cfg

if __name__=="__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = setup_config()
    demo = VisualizationDemo(cfg)

    cityscapes_dataset = register_all_cityscapes_panoptic(cfg, cfg.DATA_DIR)

    image_path = "/media/tuan/Daten/project/dataset/cityscapes/cityscapes/leftImg8bit/val2/frankfurt/frankfurt_000001_055387_leftImg8bit.png"
    img = read_image(image_path, format="BGR")
    predictions, visualized_output = demo.run_on_image(img, "panoptic")
    for k in visualized_output.keys():
        print(k)
        visualized_output[k].save("{}_test.jpg".format(k))
