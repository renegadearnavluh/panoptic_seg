import init_path
from detectron2.data import DatasetCatalog, MetadataCatalog
import os

if __name__=="__main__":
    os.environ['DETECTRON2_DATASETS'] = "/media/tuan/Daten/project/dataset/coco"
    print(DatasetCatalog.get("coco_2017_val_panoptic_separated"))
