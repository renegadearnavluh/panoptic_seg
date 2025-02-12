import os
import init_path
from my_ps.utils.demo.predictor import VisualizationDemo
import random
import numpy as np 
import torch
from detectron2.config import get_cfg
from my_ps.models.panopticfcn import add_panopticfcn_config
from detectron2.data.detection_utils import read_image
from my_ps.utils.configs import add_common_config
import glob

import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--folder_path", type=str, default="/media/tuan/Daten/project/dataset/icsens/Phillip/left", help="path to source image folder")
argParser.add_argument("--ext", type=str, default="png", help="Extention of image")
argParser.add_argument("--result_path", type=str, default="/media/tuan/Daten/project/dataset/icsens/result", help="path to saving result folder")

args = argParser.parse_args()

def setup_config():
    cfg = get_cfg()
    add_common_config(cfg)
    add_panopticfcn_config(cfg)
    cfg.merge_from_file("../../my_ps/my_ps/configs/PanopticFCN-Star-R50-3x.yaml")
    return cfg

if __name__=="__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    cfg = setup_config()
    demo = VisualizationDemo(cfg)

    path_imgs = glob.glob(os.path.join(args.folder_path, "*.{}".format(args.ext)))
    print(path_imgs)

    flag = True

    for image_path in path_imgs:

        image_name = image_path.split("/")[-1].split(".")[0]
        img = read_image(image_path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img, "panoptic")

        if flag:
            flag = False

            for k in visualized_output.keys():

                sub_path = os.path.join(args.result_path, k)
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)

        print(image_path)
        for k in visualized_output.keys():

            sub_path = os.path.join(args.result_path, k)
            visualized_output[k].save(os.path.join(sub_path, "{}_test.{}".format(image_name, args.ext)))
