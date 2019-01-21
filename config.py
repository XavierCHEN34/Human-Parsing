import os

class Config_1():
    '''
    for 4 part label segmentation : head, upper-body, lower-body, shoes
    you need to merge annotation labels from 20 to 4 before training, see  merge_class.py
    '''
    def __init__(self):
        self.CUDA = 0
        self.BATCH_SIZE = 16
        self.POWER = 0.9
        self.Lr = 1e-2
        self.NUM_STEPS = 80000
        self.UP = 4000
        self.KEEP = 20000

        self.dataset_root = '/home/cx/Dataset/seg/LIP/'
        self.dir_train_imgs = self.dataset_root + 'train_images/'
        self.dir_val_imgs = self.dataset_root + 'val_images/'
        self.dir_train_gt = self.dataset_root + 'TrainVal_parsing_annotations/train_segmentations_4/'
        self.dir_val_gt = self.dataset_root + 'TrainVal_parsing_annotations/val_segmentations_4/'
        self.dir_val_gt_4 = self.dataset_root + 'TrainVal_parsing_annotations/val_segmentations_4/'

        self.train_img_lst = os.listdir(self.dir_train_imgs)
        self.val_img_lst = os.listdir(self.dir_val_imgs)

        self.w = 256
        self.h = 128


class Config_2():
    '''
    for 20 part label segmentation
    '''
    def __init__(self):
        self.CUDA = 0
        self.BATCH_SIZE = 16
        self.POWER = 0.9
        self.Lr = 1e-2
        self.NUM_STEPS = 120000
        self.UP = 4000
        self.KEEP = 40000

        self.dataset_root = '/home/cx/Dataset/seg/LIP/'
        self.dir_train_imgs = self.dataset_root + 'train_images/'
        self.dir_val_imgs = self.dataset_root + 'val_images/'
        self.dir_train_gt = self.dataset_root + 'TrainVal_parsing_annotations/train_segmentations/'
        self.dir_val_gt = self.dataset_root + 'TrainVal_parsing_annotations/val_segmentations/'
        self.dir_val_gt_4 = self.dataset_root + 'TrainVal_parsing_annotations/val_segmentations_4/'

        self.train_img_lst = os.listdir(self.dir_train_imgs)
        self.val_img_lst = os.listdir(self.dir_val_imgs)

        self.w = 256
        self.h = 128

config = Config_1()
CUDA = config.CUDA
BATCH_SIZE = config.BATCH_SIZE
POWER = config.POWER
Lr = config.Lr
NUM_STEPS = config.NUM_STEPS
UP = config.UP
KEEP = config.KEEP

dir_train_imgs = config.dir_train_imgs
dir_val_imgs = config.dir_val_imgs
dir_train_gt = config.dir_train_gt
dir_val_gt = config.dir_val_gt
dir_val_gt_4 = config.dir_val_gt_4

train_img_lst = config.train_img_lst
val_img_list = config.val_img_lst
print(len(train_img_lst), len(val_img_list))


