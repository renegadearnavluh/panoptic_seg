import cv2
import numpy as np 

if __name__=="__main__":
    
    IMG_PATH = "/media/tuan/Daten/project/dataset/cityscapes/cityscapes/disparity/train/hanover/hanover_000000_008017_disparity.png"
    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (int(1024*1.5), int(512*1.5)))
    # img = 255-img
    cv2.imshow("img", img)
    cv2.waitKey(0)