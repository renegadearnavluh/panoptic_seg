import init_path

from detectron2.config import get_cfg
from my_ps.utils.configs import add_common_config
from detectron2.data import MetadataCatalog, build_detection_train_loader
from my_ps.data.cityscapes.cityscapes_panoptic import register_all_cityscapes_panoptic

def setup_config():
    cfg = get_cfg()
    add_common_config(cfg)
    cfg.merge_from_file("../../my_ps/my_ps/configs/base_citiscapes_panopticsegmentation.yaml")
    return cfg

if __name__=="__main__":
    cityscapes_dataset = register_all_cityscapes_panoptic("/media/tuan/Daten/project/dataset/cityscape/")

    cfg = setup_config()
    train_dataloader = build_detection_train_loader(cfg)

    for i in train_dataloader:
        print(i.keys(), len(i))
        exit()