from tqdm import tqdm
import cv2
from config import *
import numpy as np

dst_dir = '/home/cx/Dataset/seg/LIP/TrainVal_parsing_annotations/train_segmentations_5/'



for i in tqdm(range(len( train_img_lst))):
    name = train_img_lst[i]
    gt_ads = dir_train_gt +name.replace('jpg', 'png')
    save_ads = dst_dir + name.replace('jpg', 'png')
    gt = cv2.imread(gt_ads)
    w,h,c = gt.shape
    gt_4 = np.zeros((w,h,c))
    gt_4[gt == 1] = 1
    gt_4[gt == 2] = 1
    gt_4[gt == 3] = 2
    gt_4[gt == 4] = 1
    gt_4[gt == 5] = 2
    gt_4[gt == 6] = 2
    gt_4[gt == 7] = 2
    gt_4[gt == 8] = 5
    gt_4[gt == 9] = 3
    gt_4[gt == 10] = 2
    gt_4[gt == 11] = 2
    gt_4[gt == 12] = 3
    gt_4[gt == 13] = 1
    gt_4[gt == 14] = 4
    gt_4[gt == 15] = 4
    gt_4[gt == 16] = 3
    gt_4[gt == 17] = 3
    gt_4[gt == 18] = 5
    gt_4[gt == 19] = 5

    #print(np.unique(gt_4))
    cv2.imwrite(save_ads,gt_4)



