import os
import cv2
import numpy as np

dir_deeplab = 'pre _deeplab/'
dir_tiny = 'pre_tiny/'
dir_EDA = 'pre_EDA/'
dir_gt = '/home/cx/Dataset/seg/LIP/overlap_val_4/'

dir_dst = 'concat/'
img_list = os.listdir(dir_deeplab)

for i in range(len(img_list)):
    name = img_list[i]
    img_gt = cv2.imread(dir_gt + name)
    img_gt = cv2.resize(img_gt, (128,256))
    img_deeplab = cv2.imread(dir_deeplab + name)
    img_EDA = cv2.imread(dir_EDA + name)
    img_tiny = cv2.imread(dir_tiny + name)

    concat = np.concatenate( (img_tiny, img_EDA, img_deeplab, img_gt ), axis= 1 )
    cv2.imwrite(dir_dst + name, concat)
