from config import *
from utils.color_map import label2color
from tqdm import tqdm
import cv2
import numpy as np

dst_dir = config.dataset_root + 'overlap_val_4/'
for i in tqdm(range(len( val_img_list ))):
    name = val_img_list[i]
    img_ads = dir_val_imgs + name
    gt_ads = '/home/cx/Dataset/seg/LIP/TrainVal_parsing_annotations/val_segmentations_4/' +name.replace('jpg', 'png')
    save_ads = dst_dir + name

    img = cv2.imread(img_ads)
    gt = cv2.imread(gt_ads)
    #print(np.unique(gt))




    color = label2color(gt, 6)
    overlap = color * 0.6 + img * 0.4
    cv2.imwrite(save_ads,overlap)