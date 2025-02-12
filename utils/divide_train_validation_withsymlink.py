# Using this file to split panoptic annotation for cityscapes.
# Running this file after running panoptic/instance segmentation annotation for cityscapes
# Create a new split from the original train set, with symbolic link. This helps to save memory in the storage 
# Can use this tool to split training set into new_train and new_validation

import json
import os
import shutil
import random
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_root_path', type=str, default="/media/tuan/Daten/project/dataset/cityscapes",
                    help='root path of the source dataset')
parser.add_argument('--dst_root_path', type=str, default="/media/tuan/Daten/project/dataset/cityscapes",
                    help='root path of the destination dataset')
parser.add_argument('--new_split_name', type=str, default="val_split",
                    help='name of new split set splitted from the original training set')
parser.add_argument('--split_city_file', type=str, default="val_city.txt",
                    help='path to file containing list of city name for validating')
parser.add_argument('--disparity_ext', type=str, default="npy",
                    help='extension of disparity data')

if __name__=="__main__":

    args = parser.parse_args()
    disparity_ext = args.disparity_ext

    f = open(args.split_city_file, 'r')
    choosen_city = f.read().splitlines()
    f.close()

    # Source data path
    ROOT_DATASET_PATH=args.src_root_path
    PANOPTIC_JSON_ANN_PATH = "cityscapes/gtFine/cityscapes_panoptic_train.json"
    IMG_FOLDER_PATH = "cityscapes/leftImg8bit/train"
    PANOPTIC_IMG_FOLDER_PATH = "cityscapes/gtFine/cityscapes_panoptic_train"
    GTFINE_ANN_IMG_FOLDER_PATH = "cityscapes/gtFine/train"
    DISPARITY_ANN_IMG_FOLDER_PATH = "cityscapes/disparity/train"
    CAMERA_ANN_IMG_FOLDER_PATH = "cityscapes/camera/train"

    FULL_SRC_IMG_FOLDER_PATH = os.path.join(ROOT_DATASET_PATH, IMG_FOLDER_PATH)
    FULL_SRC_PANOPTIC_IMG_FOLDER_PATH = os.path.join(ROOT_DATASET_PATH, PANOPTIC_IMG_FOLDER_PATH)
    FULL_GTFINE_ANN_IMG_FOLDER_PATH = os.path.join(ROOT_DATASET_PATH, GTFINE_ANN_IMG_FOLDER_PATH)
    FULL_DISPARITY_ANN_IMG_FOLDER_PATH = os.path.join(ROOT_DATASET_PATH, DISPARITY_ANN_IMG_FOLDER_PATH)
    FULL_CAMERA_ANN_IMG_FOLDER_PATH = os.path.join(ROOT_DATASET_PATH, CAMERA_ANN_IMG_FOLDER_PATH)

    # Destimation data path. Create symbolic link for new_train, new_val set to this destination path
    ## For new_train split
    new_split_name = args.new_split_name
    DST_ROOT_DATASET_PATH=args.dst_root_path
    DST_PANOPTIC_JSON_ANN_PATH = "cityscapes/gtFine/cityscapes_panoptic_{}.json".format(new_split_name)
    DST_IMG_FOLDER_PATH = "cityscapes/leftImg8bit/{}".format(new_split_name)
    DST_PANOPTIC_IMG_FOLDER_PATH = "cityscapes/gtFine/cityscapes_panoptic_{}".format(new_split_name)
    DST_GTFINE_ANN_IMG_FOLDER_PATH = "cityscapes/gtFine/{}".format(new_split_name)
    DST_DISPARITY_ANN_IMG_FOLDER_PATH = "cityscapes/disparity/{}".format(new_split_name)
    DST_CAMERA_ANN_IMG_FOLDER_PATH = "cityscapes/camera/{}".format(new_split_name)

    ## Create new file or folder
    FULL_DST_PANOPTIC_JSON_ANN_PATH  = os.path.join(DST_ROOT_DATASET_PATH, DST_PANOPTIC_JSON_ANN_PATH)
    FULL_DST_IMG_FOLDER_PATH = os.path.join(DST_ROOT_DATASET_PATH, DST_IMG_FOLDER_PATH)
    FULL_DST_PANOPTIC_IMG_FOLDER_PATH = os.path.join(DST_ROOT_DATASET_PATH, DST_PANOPTIC_IMG_FOLDER_PATH)
    FULL_DST_GTFINE_ANN_IMG_FOLDER_PATH = os.path.join(DST_ROOT_DATASET_PATH, DST_GTFINE_ANN_IMG_FOLDER_PATH)
    FULL_DST_DISPARITY_ANN_IMG_FOLDER_PATH = os.path.join(DST_ROOT_DATASET_PATH, DST_DISPARITY_ANN_IMG_FOLDER_PATH)
    FULL_DST_CAMERA_ANN_IMG_FOLDER_PATH = os.path.join(DST_ROOT_DATASET_PATH, DST_CAMERA_ANN_IMG_FOLDER_PATH)

    file_dst_panoptic_json_ann = open(FULL_DST_PANOPTIC_JSON_ANN_PATH, 'w')

    if os.path.isdir(FULL_DST_IMG_FOLDER_PATH):
        shutil.rmtree(FULL_DST_IMG_FOLDER_PATH)
    os.makedirs(FULL_DST_IMG_FOLDER_PATH)
    print(FULL_DST_IMG_FOLDER_PATH)

    if os.path.isdir(FULL_DST_PANOPTIC_IMG_FOLDER_PATH):
        shutil.rmtree(FULL_DST_PANOPTIC_IMG_FOLDER_PATH)
    os.makedirs(FULL_DST_PANOPTIC_IMG_FOLDER_PATH)

    if os.path.isdir(FULL_DST_GTFINE_ANN_IMG_FOLDER_PATH):
        shutil.rmtree(FULL_DST_GTFINE_ANN_IMG_FOLDER_PATH)
    os.makedirs(FULL_DST_GTFINE_ANN_IMG_FOLDER_PATH)

    if os.path.isdir(FULL_DST_DISPARITY_ANN_IMG_FOLDER_PATH):
        shutil.rmtree(FULL_DST_DISPARITY_ANN_IMG_FOLDER_PATH)
    os.makedirs(FULL_DST_DISPARITY_ANN_IMG_FOLDER_PATH)

    if os.path.isdir(FULL_DST_CAMERA_ANN_IMG_FOLDER_PATH):
        shutil.rmtree(FULL_DST_CAMERA_ANN_IMG_FOLDER_PATH)
    os.makedirs(FULL_DST_CAMERA_ANN_IMG_FOLDER_PATH)

    json_data = json.load(open(os.path.join(ROOT_DATASET_PATH, PANOPTIC_JSON_ANN_PATH), 'r'))
    json_ann_data = json_data["annotations"]
    json_cate_data = json_data["categories"]
    json_images_data = json_data["images"]

    # Select the files to split based on id
    selected_json_ann_data = []
    selected_json_images_data = []

    for image_data, ann_data in zip(json_images_data, json_ann_data):
        prefix_city = image_data['id'].split("_")[0]
        if prefix_city in choosen_city:
            selected_json_ann_data.append(ann_data)
            selected_json_images_data.append(image_data)

    # Save and copy file, images to new small selection set
    selected_json_data = {}
    selected_json_data["annotations"] = selected_json_ann_data 
    selected_json_data["categories"]  = json_cate_data 
    selected_json_data["images"]  = selected_json_images_data 

    json.dump(selected_json_data, file_dst_panoptic_json_ann); file_dst_panoptic_json_ann.close()

    for img_data in selected_json_images_data:
        file_name_img = img_data["file_name"]
        id_img = img_data["id"]
        print(id_img)

        city_name = id_img.split('_')[0]

        # Copy image from src to dst
        FULL_DST_IMG_FOLDER_PATH_CITY = os.path.join(FULL_DST_IMG_FOLDER_PATH, city_name)
        FULL_SRC_IMG_FOLDER_PATH_CITY = os.path.join(FULL_SRC_IMG_FOLDER_PATH, city_name)

        if not os.path.isdir(FULL_DST_IMG_FOLDER_PATH_CITY): os.makedirs(FULL_DST_IMG_FOLDER_PATH_CITY)
        os.symlink(os.path.join(FULL_SRC_IMG_FOLDER_PATH_CITY, "{}_leftImg8bit.png".format(id_img)), \
                        os.path.join(FULL_DST_IMG_FOLDER_PATH_CITY, "{}_leftImg8bit.png".format(id_img)))
        
        # Copy panoptic ground truth image
        related_panoptic_files = glob.glob(os.path.join(FULL_SRC_PANOPTIC_IMG_FOLDER_PATH, \
                                                        "{}*".format(id_img)))
        for related_panoptic_file in related_panoptic_files:
            file_name = related_panoptic_file.split("/")[-1]
            os.symlink(related_panoptic_file, os.path.join(FULL_DST_PANOPTIC_IMG_FOLDER_PATH, \
                                                    file_name))

        # Copy all related annotation files
        FULL_DST_GTFINE_ANN_IMG_FOLDER_PATH_CITY = os.path.join(FULL_DST_GTFINE_ANN_IMG_FOLDER_PATH, city_name)
        FULL_GTFINE_ANN_IMG_FOLDER_PATH_CITY = os.path.join(FULL_GTFINE_ANN_IMG_FOLDER_PATH, city_name)

        if not os.path.isdir(FULL_DST_GTFINE_ANN_IMG_FOLDER_PATH_CITY): 
            os.makedirs(FULL_DST_GTFINE_ANN_IMG_FOLDER_PATH_CITY)
        
        related_gtfine_files = glob.glob(os.path.join(FULL_GTFINE_ANN_IMG_FOLDER_PATH_CITY, \
                                                        "{}*".format(id_img)))

        for related_gtfine_file in related_gtfine_files:
            file_name = related_gtfine_file.split("/")[-1]
            os.symlink(related_gtfine_file, os.path.join(FULL_DST_GTFINE_ANN_IMG_FOLDER_PATH_CITY, \
                                                    file_name))
            
        # Copy camera parameters
        FULL_DST_CAMERA_ANN_IMG_FOLDER_PATH_CITY = os.path.join(FULL_DST_CAMERA_ANN_IMG_FOLDER_PATH, city_name)
        FULL_CAMERA_ANN_IMG_FOLDER_PATH_CITY = os.path.join(FULL_CAMERA_ANN_IMG_FOLDER_PATH, city_name)

        if not os.path.isdir(FULL_DST_CAMERA_ANN_IMG_FOLDER_PATH_CITY): 
            os.makedirs(FULL_DST_CAMERA_ANN_IMG_FOLDER_PATH_CITY)

        os.symlink(os.path.join(FULL_CAMERA_ANN_IMG_FOLDER_PATH_CITY, "{}_camera.json".format(id_img)), \
                        os.path.join(FULL_DST_CAMERA_ANN_IMG_FOLDER_PATH_CITY , "{}_camera.json".format(id_img)))
        # Copy disparity image
        FULL_DST_DISPARITY_ANN_IMG_FOLDER_PATH_CITY = os.path.join(FULL_DST_DISPARITY_ANN_IMG_FOLDER_PATH, city_name)
        FULL_DISPARITY_ANN_IMG_FOLDER_PATH_CITY = os.path.join(FULL_DISPARITY_ANN_IMG_FOLDER_PATH, city_name)

        if not os.path.isdir(FULL_DST_DISPARITY_ANN_IMG_FOLDER_PATH_CITY): os.makedirs(FULL_DST_DISPARITY_ANN_IMG_FOLDER_PATH_CITY)
        os.symlink(os.path.join(FULL_DISPARITY_ANN_IMG_FOLDER_PATH_CITY, "{}_disparity.{}".format(id_img, disparity_ext)), \
                        os.path.join(FULL_DST_DISPARITY_ANN_IMG_FOLDER_PATH_CITY, "{}_disparity.{}".format(id_img, disparity_ext)))

